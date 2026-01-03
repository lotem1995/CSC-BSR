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
from loguru import logger
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FINE_TUNE = os.path.join(THIS_DIR, "fine_tune.py")

#define logging to file
logger.add("experiment_runner_{time}.log", rotation="1 MB")

def run(label, args_list):
    logger.info("\n" + "="*80)
    logger.info(f"RUN: {label}")
    logger.info("="*80)
    logger.info("Command: python {} {}".format(FINE_TUNE, " ".join(args_list)))
    sys.stdout.flush()
    # Change to parent directory so imports work correctly
    parent_dir = os.path.dirname(THIS_DIR)
    proc = subprocess.run([sys.executable, FINE_TUNE, *args_list], text=True, cwd=parent_dir)
    if proc.returncode != 0:
        logger.error(f"❌ {label} failed with code {proc.returncode}")
    else:
        logger.info(f"✅ {label} completed")
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

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    if failures == 0:
        logger.info("All scenarios completed successfully")
    else:
        logger.error(f"{failures} scenario(s) failed")

    return failures


if __name__ == "__main__":
    sys.exit(main())
