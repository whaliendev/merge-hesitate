import argparse
import json
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    with open(args.data_path, "r") as f:
        data = json.load(f)

    base, ours, theirs, res = data
    assert len(base) == len(ours) == len(theirs) == len(res), (
        "All datasets must have the same length"
    )

    tr, vr, _ = 0.8, 0.2, 0.0
    tr_size = int(len(base) * tr)
    vr_size = int(len(base) * vr)
    _ = len(base) - tr_size - vr_size

    indices = list(range(len(base)))
    random.shuffle(indices)

    tr_indices = indices[:tr_size]
    vr_indices = indices[tr_size : tr_size + vr_size]
    ts_indices = indices[tr_size + vr_size :]

    train_base = [base[i] for i in tr_indices]
    train_ours = [ours[i] for i in tr_indices]
    train_theirs = [theirs[i] for i in tr_indices]
    train_res = [res[i] for i in tr_indices]

    val_base = [base[i] for i in vr_indices]
    val_ours = [ours[i] for i in vr_indices]
    val_theirs = [theirs[i] for i in vr_indices]
    val_res = [res[i] for i in vr_indices]

    test_base = [base[i] for i in ts_indices]
    test_ours = [ours[i] for i in ts_indices]
    test_theirs = [theirs[i] for i in ts_indices]
    test_res = [res[i] for i in ts_indices]

    train_data = [train_base, train_ours, train_theirs, train_res]

    val_data = [val_base, val_ours, val_theirs, val_res]

    test_data = [test_base, test_ours, test_theirs, test_res]

    dump_data(train_data, os.path.join(args.output_dir, "train.json"))
    dump_data(val_data, os.path.join(args.output_dir, "val.json"))
    dump_data(test_data, os.path.join(args.output_dir, "test.json"))


def dump_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
    assert len(data) == 4, "Data must have 4 elements"
    assert len(data[0]) == len(data[1]) == len(data[2]) == len(data[3]), (
        "All datasets must have the same length"
    )
    print(f"Dumped {len(data[0])} data to {path}")


if __name__ == "__main__":
    # Create PREPROCESSED directory if it doesn't exist
    if not os.path.exists("./PREPROCESSED"):
        os.makedirs("./PREPROCESSED")
        print("Created PREPROCESSED directory")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=False, default="./RAW_DATA/cpp.json"
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, default="RAW_DATA/split"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    seed_everything()
    main(args)
