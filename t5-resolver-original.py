import subprocess
import tempfile

import numpy as np
from flask import Flask, request, jsonify
import os
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from run_mergegen import MergeT5, dotdict, model_type, beam_num, space_token
import torch

app = Flask(__name__)
tokenizer_type = "./tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

brackets_tokens = ["<lbra>", "<mbra>", "<rbra>"]
lbra_token = brackets_tokens[0]
rbra_token = brackets_tokens[2]

succeed_num = tokenizer.add_tokens(brackets_tokens)
assert succeed_num == len(brackets_tokens)
max_conflict_length = 500

args = dotdict(
    {
        "batch_size": 35,
        "test_batch_size": 30,
        "epoches": 100,
        "lr": 1e-4,
        "model_type": model_type,
        "max_conflict_length": 500,
        "max_resolve_length": 200,
    }
)
model = MergeT5(args)
model.load_state_dict(torch.load("best_model.pt"))

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class AResult:
    """A utility class to create standardized response formats."""

    @staticmethod
    def success(data=None, msg=""):
        return jsonify({
            "code": "200",
            "msg": msg,
            "data": data
        })

    @staticmethod
    def error(code="400", msg="An error occurred", data=None):
        return jsonify({
            "code": code,
            "msg": msg,
            "data": data
        })


# Error handling
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle exceptions and return standardized error response."""
    return AResult.error(code="500", msg=str(e)), 200


def execute_command(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()


def git_merge(tokens_base, tokens_a, tokens_b):
    with tempfile.TemporaryDirectory() as merge_path:
        with open(os.path.join(merge_path, "base"), "w") as f:
            f.write("\n".join(tokens_base))
        with open(os.path.join(merge_path, "a"), "w") as f:
            f.write("\n".join(tokens_a))
        with open(os.path.join(merge_path, "b"), "w") as f:
            f.write("\n".join(tokens_b))

        final_tokens = []
        execute_command(
            "git merge-file -L a -L base -L b %s/a %s/base %s/b --diff3 -p > %s/merge"
            % (merge_path, merge_path, merge_path, merge_path)
        )
        merge_res = open("%s/merge" % merge_path).read().splitlines()
        merge_res = [x.strip() for x in merge_res if x.strip()]
        format_ids = [
            k
            for k, x in enumerate(merge_res)
            if x == "<<<<<<< a"
               or x == ">>>>>>> b"
               or x == "||||||| base"
               or x == "======="
        ]
        # assert len(format_ids) % 4 == 0
        start = 0
        for k, x in enumerate(format_ids):
            if (
                    merge_res[format_ids[k]] == "<<<<<<< a"
                    and merge_res[format_ids[k + 1]] == "||||||| base"
                    and merge_res[format_ids[k + 2]] == "======="
                    and merge_res[format_ids[k + 3]] == ">>>>>>> b"
            ):
                context_tokens = merge_res[start: format_ids[k]]
                a_tokens = merge_res[format_ids[k] + 1: format_ids[k + 1]]
                base_tokens = merge_res[format_ids[k + 1] + 1: format_ids[k + 2]]
                b_tokens = merge_res[format_ids[k + 2] + 1: format_ids[k + 3]]
                start = format_ids[k + 3] + 1

                final_tokens += (
                        context_tokens
                        + [lbra_token]
                        + a_tokens
                        + [tokenizer.sep_token]
                        + base_tokens
                        + [tokenizer.sep_token]
                        + b_tokens
                        + [rbra_token]
                )

        if start != len(merge_res):
            final_tokens += merge_res[start:]
        final_tokens = (
                [tokenizer.bos_token] + final_tokens + [tokenizer.eos_token]
        )

        return final_tokens


def pad_input(input_ids, conflict_length=max_conflict_length, pad_id=tokenizer.pad_token_id):
    if len(input_ids) <= conflict_length:
        input_ids = input_ids + [pad_id] * (conflict_length - len(input_ids))
    else:
        input_ids = input_ids[:conflict_length]
    assert len(input_ids) == conflict_length
    return input_ids


def get_tensor(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)

    return tensor

def resolve(input_ids_array):
    # Ensure input_ids_array is a tensor of shape (1, seq_length)
    input_ids = get_tensor(input_ids_array)
    if len(input_ids.shape) == 1:
        # Add batch dimension
        input_ids = input_ids.unsqueeze(0)
    # Move input_ids to the appropriate device
    input_ids = input_ids.to(next(model.parameters()).device)
    # Prepare attention mask
    attention_mask = input_ids != tokenizer.pad_token_id
    with torch.no_grad():
        model.eval()
        # Encode the input_ids
        input_em = model.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )["last_hidden_state"]
        # Wrap the encoder outputs
        input_em = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=input_em
        )
        # Prepare decoder input ids
        decoder_input_ids = (
            torch.ones(len(input_ids), 1).long().to(input_ids.device)
            * tokenizer.bos_token_id
        )
        # Generate outputs using beam search
        beam_output = model.t5.generate(
            encoder_outputs=input_em,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            num_beams=beam_num,
            num_return_sequences=1,
            max_new_tokens=args.max_resolve_length * 2,
        )
        # Remove the initial bos_token_id from output
        output_ids = beam_output[:, 1:]
    # Convert output_ids to tokens and then to string
    output_ids_list = output_ids[0].tolist()
    # Remove tokens after eos_token_id if present
    if tokenizer.eos_token_id in output_ids_list:
        output_ids_list = output_ids_list[:output_ids_list.index(tokenizer.eos_token_id)]
    # Convert ids to tokens
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids_list)
    # Convert tokens to string
    output_string = "".join(output_tokens).replace(space_token, " ")
    return output_string

@app.route('/hello', methods=['GET'])
def hello():
    return "hello", 200


@app.route('/resolve_conflict', methods=['POST'])
def resolve_conflict():
    """API endpoint to handle conflict resolution between file versions."""
    try:
        # Parse POST body for raw_a, raw_b, raw_base
        raw_a = request.json.get('raw_a')
        raw_b = request.json.get('raw_b')
        raw_base = request.json.get('raw_base')

        if raw_a is None or raw_b is None or raw_base is None:
            return AResult.error(code="400", msg="Missing required parameters: raw_a, raw_b, or raw_base"), 400

        # Dummy processing - replace with actual conflict resolution logic
        raw_base = " ".join(raw_base.split())
        raw_a = " ".join(raw_a.split())
        raw_b = " ".join(raw_b.split())

        tokens_base = tokenizer.tokenize(raw_base)
        tokens_a = tokenizer.tokenize(raw_a)
        tokens_b = tokenizer.tokenize(raw_b)
        tokens_input = git_merge(tokens_base, tokens_a, tokens_b)
        ids_input = tokenizer.convert_tokens_to_ids(tokens_input)
        cur_input = np.array(pad_input(ids_input))

        result_data = resolve(cur_input)
        result_data = result_data.removeprefix(tokenizer.bos_token)

        # Return success response
        return AResult.success(data=result_data), 200

    except Exception as e:
        # In case of any error, it will be caught by the global error handler
        return handle_exception(e)


if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug=True)