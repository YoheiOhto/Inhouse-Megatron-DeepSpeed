import sys
import argparse
import torch
import os
import re
import json
import numpy as np

from megatron import print_rank_0, get_args
from megatron.initialize import initialize_megatron
from megatron.training import build_train_valid_test_data_iterators

# このスクリプトは pretrain_bert.py と同じディレクトリ階層に置く必要があります

def add_extra_args(parser):
    return parser

def get_prefixes_and_weights_from_args(args):
    """
    args.data_pathから接頭辞と重みを取得する
    """
    data_path_list = " ".join(args.data_path).split()
    if len(data_path_list) % 2 != 0:
        raise ValueError("The --data-path is not in the correct format.")

    weights = [float(data_path_list[2 * i]) for i in range(len(data_path_list) // 2)]
    prefixes = [data_path_list[2 * i + 1] for i in range(len(data_path_list) // 2)]
    
    return prefixes, weights


def main():
    """
    データセットの実際のサイズに基づき、最適な学習設定を計算して表示する。
    """
    initialize_megatron(extra_args_provider=add_extra_args, args_defaults={'tokenizer_type': 'BertWordPieceCase'})
    args = get_args()
    # 必須の属性を初期化
    if not hasattr(args, 'iteration'):
        args.iteration = 0
    if not hasattr(args, 'consumed_train_samples'):
        args.consumed_train_samples = 0
    if not hasattr(args, 'consumed_valid_samples'):
        args.consumed_valid_samples = 0
    
    try:
        from pretrain_bert import train_valid_test_datasets_provider
    except ImportError:
        print("Error: pretrain_bert.py not found.", file=sys.stderr)
        sys.exit(1)

    print_rank_0("Building datasets to determine actual sample counts (this may take a moment)...")
    
    train_samples_demand = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples_demand,  eval_iters * args.global_batch_size, test_iters * args.global_batch_size]
    train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(train_val_test_num_samples)

    if not hasattr(train_ds, 'datasets'):
        print_rank_0("This checker is for BlendableDataset only. Exiting.")
        sys.exit(0)
        
    print_rank_0("\n" + "#" * 70)
    print_rank_0("### Megatron Setting Generator & Verifier ###")
    print_rank_0("#" * 70)

    prefixes = [os.path.basename(p) for p in args.data_path if not p.replace('.', '', 1).isdigit()]
    train_counts = [len(d) for d in train_ds.datasets]
    total_train_supply = sum(train_counts)

    print_rank_0("\n[1. Actual Sample Counts (Supply)]")
    header = f"{'Dataset':<25} | {'Train Samples':>15}"
    print_rank_0(header)
    print_rank_0("-" * len(header))
    for i in range(len(prefixes)):
        name = os.path.basename(prefixes[i]).split('_')[0]
        print_rank_0(f"{name:<25} | {train_counts[i]:>15,}")
    print_rank_0("-" * len(header))
    print_rank_0(f"{'TOTAL':<25} | {total_train_supply:>15,}")

    recommended_weights = [count / total_train_supply for count in train_counts] if total_train_supply > 0 else [0.0] * len(train_counts)
    recommended_iters = total_train_supply // args.global_batch_size
    
    print_rank_0("\n[2. Recommended Settings for Your .sh Script]")
    print_rank_0("\n# --- Copy and paste this block into your .sh script ---")
    print_rank_0(f"# Recommended settings for 1 epoch")
    print_rank_0(f"train_iters={recommended_iters}")
    print_rank_0(f"lr_decay_iters={recommended_iters}")
    print_rank_0("")
    print_rank_0("# Corrected weights based on actual data")
    for i in range(len(prefixes)):
        dataset_name_key = os.path.basename(prefixes[i]).split('_')[0]
        print_rank_0(f"weight_{dataset_name_key}={recommended_weights[i]:.4f}")
    print_rank_0("# ----------------------------------------------------")
    
    print_rank_0("\nSUCCESS: Settings generated. You can now use these in your main training script.")
    print_rank_0("No need for further verification as these settings are derived from the actual data.")


if __name__ == "__main__":
    main()