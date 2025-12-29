#!/usr/bin/env python3
"""
Experiment Runner: executes multiple configurations sequentially on a single GPU node.

Scenarios:
1) Head-only: Qwen (classifier only)
2) Head-only: DINO (classifier only)
3) LoRA: Qwen (LoRA adapters + classifier)
4) Backbone finetune: DINO (backbone + classifier)

Usage (env overrides):
  EPOCHS=1 BATCH_SIZE=2 NUM_WORKERS=2 python embedding/experiment_runner.py \
      --splits-dir data/splits --path-root data --dino-size small

Outputs: logs are printed to stdout; fine_tune.py handles metrics.
"""

import os
import sys
import subprocess
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINE_TUNE = os.path.join(THIS_DIR, "fine_tune.py")


def run(label, args_list):
    print("\n" + "="*80)
    print(f"RUN: {label}")
    print("="*80)
    print("Command:", "python", FINE_TUNE, *args_list)
    sys.stdout.flush()
    proc = subprocess.run([sys.executable, FINE_TUNE, *args_list], text=True)
    if proc.returncode != 0:
        print(f"❌ {label} failed with code {proc.returncode}")
    else:
        print(f"✅ {label} completed")
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--path-root", default=".")
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 1)))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 2)))
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", 2)))
    p.add_argument("--dino-size", choices=["small", "base"], default=os.environ.get("DINO_SIZE", "small"))
    args = p.parse_args()

    scenarios = [
        ("Qwen - head-only", [
            "--splits-dir", args.splits_dir,
            "--path-root", args.path_root,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--embedding-model", "qwen",
            "--strategy", "head-only",
        ]),
        ("DINO - head-only", [
            "--splits-dir", args.splits_dir,
            "--path-root", args.path_root,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--embedding-model", f"dino-{args.dino_size}",
            "--strategy", "head-only",
        ]),
        ("Qwen - LoRA", [
            "--splits-dir", args.splits_dir,
            "--path-root", args.path_root,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--embedding-model", "qwen",
            "--strategy", "lora",
        ]),
        ("DINO - backbone finetune", [
            "--splits-dir", args.splits_dir,
            "--path-root", args.path_root,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--embedding-model", f"dino-{args.dino_size}",
            "--strategy", "backbone",
        ]),
    ]

    failures = 0
    for label, cmd_args in scenarios:
        rc = run(label, cmd_args)
        failures += int(rc != 0)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if failures == 0:
        print("All scenarios completed successfully")
    else:
        print(f"{failures} scenario(s) failed")

    return failures


if __name__ == "__main__":
    sys.exit(main())
