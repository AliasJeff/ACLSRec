#!/usr/bin/env python3
"""Convert SECL/HINT processed JSON sequences to RecBole .inter atomic files.

Reads <cache>/<category>/processed/all_item_seqs.json (SECL preprocessed
cache) and writes dataset/<category>/<category>.inter in RecBole atomic-file
format (user_id:token, item_id:token, timestamp:float). HINT item ids are
written verbatim as tokens so RecBole remaps them internally.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Convert SECL all_item_seqs.json to RecBole .inter format."
    )
    parser.add_argument(
        "--hint_cache_dir",
        default="/root/autodl-tmp/HINT/cache/AmazonReviews2014",
        help="SECL cache root directory (contains <category>/processed)",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Dataset category, e.g. Office_Products / Yelp",
    )
    parser.add_argument(
        "--out_dir",
        default="./dataset",
        help="Output directory for RecBole datasets (default: ./dataset)",
    )
    args = parser.parse_args()

    json_path = os.path.join(
        args.hint_cache_dir, args.category, "processed", "all_item_seqs.json"
    )
    out_dir = os.path.join(args.out_dir, args.category)
    os.makedirs(out_dir, exist_ok=True)

    inter_path = os.path.join(out_dir, f"{args.category}.inter")

    with open(json_path, "r", encoding="utf-8") as f:
        all_item_seqs = json.load(f)

    with open(inter_path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for user_id, item_seq in all_item_seqs.items():
            for index, item_id in enumerate(item_seq):
                f.write(f"{user_id}\t{item_id}\t{index}\n")

    print(f"Successfully saved RecBole .inter file to: {inter_path}")
    print(f"  users: {len(all_item_seqs)}, "
          f"interactions: {sum(len(s) for s in all_item_seqs.values())}")


if __name__ == "__main__":
    main()
