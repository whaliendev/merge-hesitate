# t5-hesitate-resolver.py
import subprocess
import tempfile
import numpy as np
from flask import Flask, request, jsonify
import os
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator # Use Accelerator for loading state
import logging
import time # Import time for request duration logging

try:
    from merge_hesitate.model import HesitateT5
    from merge_hesitate.confidence import ConfidenceCalibrator
except ImportError:
    print("Error: Ensure merge_hesitate package is installed or accessible in PYTHONPATH.")
    exit(1)

# --- Configuration ---
# TODO: Load these dynamically from saved config or env vars
MODEL_STATE_PATH = "OUTPUT/best_model_state" # Path relative to execution dir
CALIBRATOR_PATH = os.path.join(MODEL_STATE_PATH, "calibrator.pt")
TOKENIZER_PATH = "./tokenizer"
MODEL_NAME_OR_PATH = "./codet5-small" # Base model used for init, state will override
USE_FEATURES = False # IMPORTANT: Must match training! Set to True if features were used.
FEATURE_SIZE = 12   # IMPORTANT: Must match training! Ignored if USE_FEATURES is False.
MAX_SEQ_LENGTH = 512 # Max length for padded input, must match training data prep
MAX_RESOLVE_LEN = 256 # Max length for generated output tokens (like in evaluate)
NUM_BEAMS = 3 # Number of beams for generation, should match evaluation setting

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
app = Flask(__name__)
tokenizer = None
model = None
calibrator = None
device = None
space_token = "Ġ" # Assuming default space token for CodeT5 tokenizer

# --- Utility Class ---
class AResult:
    @staticmethod
    def success(data=None, msg=""):
        return jsonify({"code": "200", "msg": msg, "data": data})

    @staticmethod
    def error(code="400", msg="An error occurred", data=None):
        return jsonify({"code": code, "msg": msg, "data": data})

# --- Error Handling ---
@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("An error occurred during request processing:")
    # Return standardized error response but log the full traceback
    return AResult.error(code="500", msg=f"Internal Server Error: {type(e).__name__}"), 200


# --- Git Merge Helper ---
lbra_token = "<lbra>"
rbra_token = "<rbra>"
def execute_command(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()


def git_merge(tokens_base, tokens_a, tokens_b):
    global tokenizer # Access global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded")
    with tempfile.TemporaryDirectory() as merge_path:
        # Ensure correct file paths within the temporary directory
        # print(">>>>>>>", merge_path)
        base_file = os.path.join(merge_path, "base")
        a_file = os.path.join(merge_path, "a")
        b_file = os.path.join(merge_path, "b")

        try:
            with open(base_file, "w", encoding='utf-8') as f:
                 f.write("\n".join(tokens_base))
            with open(a_file, "w", encoding='utf-8') as f:
                 f.write("\n".join(tokens_a))
            with open(b_file, "w", encoding='utf-8') as f:
                 f.write("\n".join(tokens_b))
        except Exception as e:
            logger.error(f"Error writing temp files: {e}")
            raise

        final_tokens = []
        # Execute git merge-file and capture stdout
        merge_cmd = f"git merge-file -L a -L base -L b {a_file} {base_file} {b_file} --diff3 -p > {merge_path}/merge"
        execute_command(merge_cmd)
        logger.debug(f"Executing git merge command: {merge_cmd}")
        with open(f"{merge_path}/merge", "r") as f:
            merge_res = f.read().splitlines()
        merge_res = [x.strip() for x in merge_res]

        # Robust parsing logic for diff3 format
        idx = 0
        while idx < len(merge_res):
            line = merge_res[idx]
            if line.startswith("<<<<<<<"):
                final_tokens.append(lbra_token)
                idx += 1 # Move past '<<<<<<<' marker
                while idx < len(merge_res) and not merge_res[idx].startswith("|||||||"):
                    final_tokens.append(merge_res[idx])
                    idx += 1
                if idx < len(merge_res) and merge_res[idx].startswith("|||||||"):
                    final_tokens.append(tokenizer.sep_token)
                    idx += 1 # Move past '|||||||' marker
                    while idx < len(merge_res) and not merge_res[idx].startswith("======="):
                         final_tokens.append(merge_res[idx])
                         idx += 1
                    if idx < len(merge_res) and merge_res[idx].startswith("======="):
                         final_tokens.append(tokenizer.sep_token)
                         idx += 1 # Move past '=======' marker
                         while idx < len(merge_res) and not merge_res[idx].startswith(">>>>>>>"):
                              final_tokens.append(merge_res[idx])
                              idx += 1
                         if idx < len(merge_res) and merge_res[idx].startswith(">>>>>>>"):
                              final_tokens.append(rbra_token)
                              idx += 1 # Move past '>>>>>>>' marker
                         else:
                              logger.warning("Malformed diff3 block: missing '>>>>>>>'")
                              # Append remaining lines in block as part of 'b' for safety
                              while idx < len(merge_res):
                                   final_tokens.append(merge_res[idx])
                                   idx+=1
                              final_tokens.append(rbra_token) # Add closing bracket anyway

                    else:
                         logger.warning("Malformed diff3 block: missing '======='")
                         idx += 1 # Skip current line to avoid infinite loop maybe
                else:
                     logger.warning("Malformed diff3 block: missing '|||||||'")
                     idx += 1 # Skip current line

            else: # Normal context line
                final_tokens.append(line)
                idx += 1

        # Add BOS/EOS (Using actual tokens, not IDs yet)
        final_tokens = [tokenizer.bos_token] + final_tokens + [tokenizer.eos_token]
        # Return tokens, not IDs
        return final_tokens

# --- Input Padding ---
def pad_input(input_ids, conflict_length=MAX_SEQ_LENGTH - FEATURE_SIZE):
    """Pads or truncates input_ids (list of integers) to the specified conflict_length."""
    global tokenizer
    pad_id_actual = tokenizer.pad_token_id if tokenizer else 0 # Get actual pad ID

    if len(input_ids) <= conflict_length:
        padded_ids = input_ids + [pad_id_actual] * (conflict_length - len(input_ids))
    else:
        # Truncate from the end
        logger.warning(f"Input length {len(input_ids)} exceeds max_seq_length {conflict_length}. Truncating.")
        padded_ids = input_ids[:conflict_length]
    assert len(padded_ids) == conflict_length, f"Padding failed: expected {conflict_length}, got {len(padded_ids)}"
    return padded_ids

# --- Model Loading ---
def load_model_and_tokenizer():
    """Loads the HesitateT5 model, tokenizer, and calibrator."""
    global tokenizer, model, calibrator, device

    logger.info("Loading resources...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        # Ensure special tokens are known (needed for git_merge)
        special_tokens = ["<lbra>", "<rbra>"]
        tokenizer.add_tokens(special_tokens)
        logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

        logger.info(f"Initializing HesitateT5 model (base: {MODEL_NAME_OR_PATH})")
        # Instantiate the model structure first
        model_structure = HesitateT5(
            model_name_or_path=MODEL_NAME_OR_PATH,
            tokenizer=tokenizer, # Pass tokenizer for potential resizing
            confidence_threshold=0.88, # Initial threshold, will be loaded from state
            feature_size=FEATURE_SIZE,
            use_features=USE_FEATURES,
        )

        # Use a minimal Accelerator instance just for loading
        accelerator = Accelerator()

        # Prepare the model structure with Accelerator (required before load_state)
        model = accelerator.prepare(model_structure)

        if not os.path.exists(MODEL_STATE_PATH):
             raise FileNotFoundError(f"Model state directory not found: {MODEL_STATE_PATH}")

        logger.info(f"Loading model state from {MODEL_STATE_PATH}")
        accelerator.load_state(MODEL_STATE_PATH)
        logger.info("Model state loaded successfully.")

        # Keep the model unwrapped after loading for direct access
        model = accelerator.unwrap_model(model)
        model.eval() # Set to evaluation mode
        model.to(device) # Ensure model is on the correct device after loading

        logger.info(f"Model loaded. Current threshold: {model.confidence_threshold:.4f}")


        # Load the calibrator
        if os.path.exists(CALIBRATOR_PATH):
            logger.info(f"Loading calibrator from {CALIBRATOR_PATH}")
            try:
                # Load calibrator to CPU explicitly
                calibrator = torch.load(CALIBRATOR_PATH, map_location=torch.device('cpu'), weights_only=False)
                # Ensure any internal tensors (like temperature) ALSO remain on CPU
                if hasattr(calibrator, 'temperature') and isinstance(calibrator.temperature, torch.Tensor):
                     # Keep temperature on CPU to match calibrator loading device and inference input type
                     calibrator.temperature = calibrator.temperature.to(torch.device('cpu'))
                     logger.info(f"Calibrator temperature tensor explicitly kept on CPU.")
                logger.info(f"Calibrator loaded to CPU (Strategy: {getattr(calibrator, 'strategy', 'N/A')}).")
            except Exception as e:
                logger.error(f"Failed to load calibrator: {e}. Proceeding without calibration.")
                calibrator = None
        else:
            logger.warning(f"Calibrator file not found at {CALIBRATOR_PATH}. Proceeding without calibration.")
            calibrator = None

        logger.info("Resource loading complete.")

    except FileNotFoundError as e:
        logger.error(f"Error loading resources: {e}")
        raise e # Re-raise to stop the app if loading fails
    except Exception as e:
        logger.error(f"An unexpected error occurred during loading: {e}")
        raise e


# --- Core Resolution Logic ---
def resolve_with_hesitation(input_ids_array):
    """Resolves conflict using HesitateT5, applying calibration and threshold."""
    global model, tokenizer, calibrator, device

    if model is None or tokenizer is None:
        logger.error("Model or Tokenizer not loaded. Cannot resolve.")
        raise RuntimeError("Model/Tokenizer not loaded.")

    input_ids = torch.tensor(input_ids_array, dtype=torch.long).unsqueeze(0) # Add batch dim
    input_ids = input_ids.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

    # Prepare features if needed (assuming USE_FEATURES=False here)
    features = None
    if USE_FEATURES:
        logger.warning("USE_FEATURES is True, but feature extraction is NOT IMPLEMENTED in this resolver.")
        # features = extract_features_function(...) # Placeholder
        # features = features.to(device)

    resolved_content_str = None
    final_confidence = None
    hesitated = True # Default to hesitated

    with torch.no_grad():
        try:
            # Use the model's generate method
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "features": features, # Will be None if USE_FEATURES is False
                "num_beams": NUM_BEAMS,
                "max_resolve_len": MAX_RESOLVE_LEN,
            }
            # Remove features kwarg if not used by the loaded model
            if not hasattr(model, 'use_features') or not model.use_features:
                 if "features" in gen_kwargs:
                      gen_kwargs.pop("features")
            elif "features" not in gen_kwargs and USE_FEATURES: # Check global config too
                 # This case should not happen if USE_FEATURES is True globally and implemented
                 logger.error("Model expects features, but none were provided/implemented.")
                 raise ValueError("Feature mismatch: Model expects features.")


            generated_ids, raw_confidence_tensor = model.generate(**gen_kwargs)
            # raw_confidence_tensor shape: [1]

            raw_confidence = raw_confidence_tensor.item() # Get float value
            model_threshold = model.confidence_threshold # Get threshold loaded from state

            # --- Apply Calibration ---
            calibrated_confidence = raw_confidence # Default if no calibrator
            if calibrator is not None:
                try:
                    # Calibrator expects list of floats, returns list of floats
                    calibrated_list = calibrator.calibrate([raw_confidence])
                    calibrated_confidence = calibrated_list[0]
                    logger.info(f"Raw confidence: {raw_confidence:.4f}, Calibrated confidence: {calibrated_confidence:.4f}")
                except Exception as e:
                    logger.warning(f"Calibration failed during inference: {e}. Using raw confidence.")
                    calibrated_confidence = raw_confidence # Fallback to raw
            else:
                 logger.info(f"No calibrator loaded. Using raw confidence: {raw_confidence:.4f}")

            final_confidence = calibrated_confidence # Store the score used for decision

            # Decode the generated sequence (beam 0)
            output_ids_list = generated_ids[0].tolist() # Shape [max_resolve_len] -> list

            # Remove padding and EOS token robustly
            try:
                eos_index = output_ids_list.index(tokenizer.eos_token_id)
                output_ids_list = output_ids_list[:eos_index]
            except ValueError:
                pass # EOS not found, use full list
            # Remove padding tokens
            output_ids_list = [id_ for id_ in output_ids_list if id_ != tokenizer.pad_token_id]

            # Use tokenizer.decode for robust conversion
            resolved_content_str = tokenizer.decode(output_ids_list, skip_special_tokens=True)

            # --- Hesitation Check (AFTER decoding) ---
            logger.info(f"Comparing confidence {final_confidence:.4f} against threshold {model_threshold:.4f}")
            if final_confidence >= model_threshold:
                hesitated = False
                logger.info(f"Confidence >= threshold. Accepting resolution.")
            else:
                hesitated = True
                logger.info(f"Confidence < threshold. Hesitating (but returning generated content).)")
                # NOTE: We still return resolved_content_str even if hesitated

        except Exception as e:
            logger.error(f"Error during model generation or processing: {e}", exc_info=True)
            # Keep hesitated=True, return None content and confidence
            resolved_content_str = None # Set to None on generation error
            final_confidence = raw_confidence if 'raw_confidence' in locals() else None # Report raw if calibration failed early
            hesitated = True # Ensure hesitated is True on error

    # Return the generated content, confidence, and hesitation flag
    return resolved_content_str, final_confidence, hesitated


# --- Flask Routes ---
@app.route('/hello', methods=['GET'])
def hello():
    return AResult.success(msg="Hello from HesitateT5 Resolver!")

@app.route('/resolve_conflict', methods=['POST'])
def resolve_conflict():
    """API endpoint to handle conflict resolution using HesitateT5."""
    start_time = time.time()
    try:
        # Parse POST body
        json_data = request.get_json()
        if not json_data:
            logger.warning("Received empty or invalid JSON payload.")
            return AResult.error(code="400", msg="Invalid JSON payload"), 200 # Test script expects 200.
        raw_a = json_data.get('raw_a')
        raw_b = json_data.get('raw_b')
        raw_base = json_data.get('raw_base')

        if raw_a is None or raw_b is None or raw_base is None:
            logger.warning("Missing required parameters in JSON payload.")
            return AResult.error(code="400", msg="Missing required parameters: raw_a, raw_b, or raw_base"), 200

        # Tokenize input strings as lines for git_merge
        raw_base = " ".join(raw_base.split())
        raw_a = " ".join(raw_a.split())
        raw_b = " ".join(raw_b.split())
        tokens_base = tokenizer.tokenize(raw_base)
        tokens_a = tokenizer.tokenize(raw_a)
        tokens_b = tokenizer.tokenize(raw_b)

        # Create diff3 input tokens using git_merge helper
        tokens_input = git_merge(tokens_base, tokens_a, tokens_b) # Returns list of tokens

        # Convert tokens to IDs
        ids_input = tokenizer.convert_tokens_to_ids(tokens_input)

        # Pad input IDs
        padded_ids_list = pad_input(ids_input) # Returns list of integers

        # Convert to numpy array
        padded_ids_array = np.array(padded_ids_list)

        # Resolve with hesitation
        resolved_content, confidence, hesitated = resolve_with_hesitation(padded_ids_array)

        # --- Log Inputs and Resolution ---
        logger.info("="*15 + " Conflict Resolution Attempt " + "="*15)
        logger.info(f"INPUT Base:\n---\n{raw_base}\n---")
        logger.info(f"INPUT A:\n---\n{raw_a}\n---")
        logger.info(f"INPUT B:\n---\n{raw_b}\n---")
        logger.info("-"*40)
        logger.info(f"GENERATED Content:\n---\n{resolved_content if resolved_content is not None else '[Error during generation]'}\n---")
        confidence_str = f'{confidence:.4f}' if confidence is not None else 'N/A'
        logger.info(f"CONFIDENCE: {confidence_str}")
        logger.info(f"HESITATED: {hesitated}")
        logger.info("="*55)
        # ---------------------------------

        # Prepare response data
        response_data = {
            "resolved_content": resolved_content if not hesitated else None, # Return None *in response* if hesitated
            "confidence": confidence,
            "hesitated": hesitated
        }
        duration = time.time() - start_time
        logger.info(f"Request processed in {duration:.4f} seconds. Decision: {'Accepted' if not hesitated else 'Hesitated'}.")

        # Return success response with structured data
        return AResult.success(data=response_data), 200

    except Exception as e:
        # Let the global handler manage this, it logs the exception
        return handle_exception(e)

# --- Main Execution ---
if __name__ == '__main__':
    # Load model and tokenizer once at startup
    try:
        load_model_and_tokenizer()
        # Run Flask app
        logger.info("Starting Flask application on http://0.0.0.0:5000")
        # Use waitress or gunicorn for production instead of Flask dev server
        # For simplicity, using Flask's built-in server here
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
         logger.critical(f"Failed to initialize or run the application: {e}", exc_info=True)
