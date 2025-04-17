import json
import os
import multiprocessing as mp
from tqdm import tqdm
import pickle
import numpy as np
import subprocess
from transformers import AutoTokenizer

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
except Exception as e:
    print(f"Tokenizer not found, fail to preprocess dataset: {e}")
    exit()

# Add special tokens
brackets_tokens = ["<lbra>", "<mbra>", "<rbra>"]
succeed_num = tokenizer.add_tokens(brackets_tokens)
assert succeed_num == len(brackets_tokens)

# Constants
# max input feature size is 512, we remain 12 for other features
MAX_CONFLICT_LENGTH = 500
MAX_RESOLVE_LENGTH = 256
LBRA_TOKEN = "<lbra>"
MBRA_TOKEN = "<mbra>"
RBRA_TOKEN = "<rbra>"


def process_chunk(start_idx, end_idx, data, data_name, result_queue):
    """Process a chunk of data"""
    try:
        # Unpack the data
        all_raw_base, all_raw_a, all_raw_b, all_raw_res = data

        # Get this chunk's data
        chunk_base = all_raw_base[start_idx:end_idx]
        chunk_a = all_raw_a[start_idx:end_idx]
        chunk_b = all_raw_b[start_idx:end_idx]
        chunk_res = all_raw_res[start_idx:end_idx]

        data_num = len(chunk_base)
        max_conflict_length = 0
        max_resolve_length = 0
        inputs = []
        outputs = []

        for i in tqdm(
            range(data_num), desc=f"Processing {data_name} {start_idx}-{end_idx}"
        ):
            global_idx = start_idx + i
            raw_base = chunk_base[i]
            raw_a = chunk_a[i]
            raw_b = chunk_b[i]
            raw_res = chunk_res[i]

            # Format raw text
            raw_base = " ".join(raw_base.split())
            raw_a = " ".join(raw_a.split())
            raw_b = " ".join(raw_b.split())
            raw_res = " ".join(raw_res.split())

            # Tokenize
            tokens_base = tokenizer.tokenize(raw_base)
            tokens_a = tokenizer.tokenize(raw_a)
            tokens_b = tokenizer.tokenize(raw_b)
            tokens_res = tokenizer.tokenize(raw_res)

            # Process with git merge
            tokens_input = git_merge(tokens_base, tokens_a, tokens_b, global_idx)

            # Convert to ids
            ids_input = tokenizer.convert_tokens_to_ids(tokens_input)
            ids_res = tokenizer.convert_tokens_to_ids(tokens_res)

            cur_input = ids_input
            cur_output = ids_res + [tokenizer.eos_token_id]

            max_conflict_length = max(max_conflict_length, len(cur_input))
            max_resolve_length = max(max_resolve_length, len(cur_output))

            # Pad sequences
            cur_input = pad_length(
                cur_input, MAX_CONFLICT_LENGTH, tokenizer.pad_token_id
            )
            cur_output = pad_length(
                cur_output, MAX_RESOLVE_LENGTH, tokenizer.pad_token_id
            )

            inputs.append(cur_input)
            outputs.append(cur_output)

        print(
            f"{data_name} {start_idx}-{end_idx} - max lengths:",
            max_conflict_length,
            max_resolve_length,
        )

        # Save this chunk's processed data
        chunk_data = [np.array(inputs), np.array(outputs)]
        chunk_path = f"PROCESSED/processed_{data_name}_{start_idx}_{end_idx}.pkl"
        with open(chunk_path, "wb") as f:
            pickle.dump(chunk_data, f)

        result_queue.put((start_idx, end_idx, chunk_path))

    except Exception as e:
        print(f"Error processing chunk {start_idx} to {end_idx}: {str(e)}")
        result_queue.put(None)


def git_merge(tokens_base, tokens_a, tokens_b, index):
    """Perform git merge on the tokens"""
    merge_path = f"GIT_MERGE_FILES/{index}"
    if not os.path.exists(merge_path):
        os.makedirs(merge_path)

    # Write tokens to files
    with open(f"{merge_path}/base", "w") as f:
        f.write("\n".join(tokens_base))
    with open(f"{merge_path}/a", "w") as f:
        f.write("\n".join(tokens_a))
    with open(f"{merge_path}/b", "w") as f:
        f.write("\n".join(tokens_b))

    # Run git merge
    cmd = f"git merge-file -L a -L base -L b {merge_path}/a {merge_path}/base {merge_path}/b --diff3 -p > {merge_path}/merge"
    execute_command(cmd)

    merge_res = open(f"{merge_path}/merge").read().splitlines()
    merge_res = [x.strip() for x in merge_res if x.strip()]

    # Find merge markers
    format_ids = [
        k
        for k, x in enumerate(merge_res)
        if x == "<<<<<<< a" or x == ">>>>>>> b" or x == "||||||| base" or x == "======="
    ]

    # print strange cases, as there may exists unresolved conflicts
    if len(format_ids) % 4 != 0:
        print(
            f"----------------- FORMAT_IDS_LEN: {len(format_ids)} -------------------"
        )

    # Process merge results
    final_tokens = []
    start = 0
    for k, x in enumerate(format_ids):
        if (
            k + 3 < len(format_ids)
            and merge_res[format_ids[k]] == "<<<<<<< a"
            and merge_res[format_ids[k + 1]] == "||||||| base"
            and merge_res[format_ids[k + 2]] == "======="
            and merge_res[format_ids[k + 3]] == ">>>>>>> b"
        ):
            context_tokens = merge_res[start : format_ids[k]]
            a_tokens = merge_res[format_ids[k] + 1 : format_ids[k + 1]]
            base_tokens = merge_res[format_ids[k + 1] + 1 : format_ids[k + 2]]
            b_tokens = merge_res[format_ids[k + 2] + 1 : format_ids[k + 3]]
            start = format_ids[k + 3] + 1

            final_tokens += (
                context_tokens
                + [LBRA_TOKEN]
                + a_tokens
                + [tokenizer.sep_token]
                + base_tokens
                + [tokenizer.sep_token]
                + b_tokens
                + [RBRA_TOKEN]
            )

    # Add remaining tokens
    if start < len(merge_res):
        final_tokens += merge_res[start:]

    # Add special tokens
    # final_tokens = [tokenizer.bos_token] + final_tokens + [tokenizer.eos_token]

    return final_tokens


def execute_command(cmd):
    """Execute a shell command"""
    p = subprocess.Popen(cmd, shell=True)
    p.wait()


def pad_length(tokens, max_length, pad_id):
    """Pad or truncate a sequence to the specified length"""
    if len(tokens) <= max_length:
        tokens = tokens + [pad_id] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]
    assert len(tokens) == max_length, "Tokens length must be equal to max_length"
    return tokens


def process_data_parallel(data, data_name):
    """
    Process data in parallel using multiprocessing

    data: [
      [], # base_list
      [], # ours_list
      [], # theirs_list
      [], # res_list
    ]
    """
    assert len(data) == 4, "Data must have 4 elements"
    assert len(data[0]) == len(data[1]) == len(data[2]) == len(data[3]), (
        "All datasets must have the same length"
    )

    interval = 1000  # Process in chunks of 1000 items
    total_items = len(data[0])

    # Create a list of start and end indices for each chunk
    chunks = [
        (i, min(i + interval, total_items)) for i in range(0, total_items, interval)
    ]

    # Create a queue for communication between processes
    manager = mp.Manager()
    result_queue = manager.Queue()

    # Create and start processes
    processes = []
    for start_idx, end_idx in chunks:
        p = mp.Process(
            target=process_chunk,
            args=(start_idx, end_idx, data, data_name, result_queue),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    results = []
    with tqdm(total=len(chunks), desc=f"Processing {data_name} chunks") as pbar:
        for _ in range(len(chunks)):
            result = result_queue.get()
            if result is not None:
                results.append(result)
            pbar.update(1)

    # Join all processes
    for p in processes:
        p.join()

    # Merge all processed data
    if results:
        print(f"Merging processed data for {data_name}...")
        merge_processed_data(results, data_name)


def merge_processed_data(chunks, data_name):
    """Merge all processed data chunks into a single file"""
    all_inputs = []
    all_outputs = []

    for start_idx, end_idx, chunk_path in sorted(chunks):
        if os.path.exists(chunk_path):
            try:
                with open(chunk_path, "rb") as f:
                    chunk_data = pickle.load(f)
                    all_inputs.extend(chunk_data[0])
                    all_outputs.extend(chunk_data[1])
            except Exception as e:
                print(f"Error loading {chunk_path}: {str(e)}")

    if all_inputs and all_outputs:
        merged_data = [np.array(all_inputs), np.array(all_outputs)]
        output_path = f"PREPROCESSED/processed_{data_name}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(merged_data, f)
        print(f"Saved merged data to {output_path}")


def main():
    """Main function to load and process all datasets"""
    with open("./RAW_DATA/split/train.json", "r") as f:
        train_data = json.load(f)

    with open("./RAW_DATA/split/val.json", "r") as f:
        val_data = json.load(f)

    with open("./RAW_DATA/split/test.json", "r") as f:
        test_data = json.load(f)

    # Process each dataset in parallel
    process_data_parallel(train_data, "train")
    process_data_parallel(val_data, "val")
    process_data_parallel(test_data, "test")


if __name__ == "__main__":
    if not os.path.exists("./RAW_DATA/split"):
        print("Please run split_dataset.py first")
        exit()

    # Create necessary directories
    if not os.path.exists("PROCESSED"):
        os.makedirs("PROCESSED")
    if not os.path.exists("PREPROCESSED"):
        os.makedirs("PREPROCESSED")
    if not os.path.exists("GIT_MERGE_FILES"):
        os.makedirs("GIT_MERGE_FILES")

    main()
