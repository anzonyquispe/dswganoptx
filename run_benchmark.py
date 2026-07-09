#!/usr/bin/env python3
"""
Benchmark script comparing original vs optimized WGAN package performance.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Check for required packages
try:
    import torch
    import pandas as pd
    import numpy as np
    from time import time
    from copy import copy
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)

print("=" * 70)
print("         WGAN BENCHMARK: Original vs Optimized Package")
print("=" * 70)
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Device priority: MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# Load data
print("\n[1/6] Loading data...")
df = pd.read_feather('/Users/anzony.quisperojas/Documents/GitHub/python/ds-wgan/data/original_data/cps.feather')
df = df.drop(["u74", "u75"], axis=1)

np.random.seed(42)
df_balanced = df.sample(2*len(df), weights=(1-df.t.mean())*df.t+df.t.mean()*(1-df.t), replace=True)
print(f"Dataset shape: {df_balanced.shape}")

# Parameters
continuous_vars_0 = ["age", "education", "re74", "re75"]
continuous_lower_bounds_0 = {"re74": 0, "re75": 0}
categorical_vars_0 = ["black", "hispanic", "married", "nodegree"]
context_vars_0 = ["t"]

continuous_vars_1 = ["re78"]
continuous_lower_bounds_1 = {"re78": 0}
categorical_vars_1 = []
context_vars_1 = ["t", "age", "education", "re74", "re75", "black", "hispanic", "married", "nodegree"]

BATCH_SIZE = 4096
MAX_EPOCHS = 1000
CRITIC_LR = 1e-3
GENERATOR_LR = 1e-3
PRINT_EVERY = 200  # Less verbose output

# ============================================================
# ORIGINAL PACKAGE
# ============================================================
print("\n" + "=" * 70)
print("[2/6] BENCHMARKING ORIGINAL PACKAGE")
print("=" * 70)

sys.path = [p for p in sys.path if 'dswganoptx' not in p]
sys.path.insert(0, '/Users/anzony.quisperojas/Documents/GitHub/python/ds-wgan')

if 'wgan' in sys.modules:
    del sys.modules['wgan']
if 'wgan.wgan' in sys.modules:
    del sys.modules['wgan.wgan']

import wgan as wgan_original

torch.manual_seed(42)
data_wrappers_orig = [
    wgan_original.DataWrapper(df_balanced, continuous_vars_0, categorical_vars_0,
                              context_vars_0, continuous_lower_bounds_0),
    wgan_original.DataWrapper(df_balanced, continuous_vars_1, categorical_vars_1,
                              context_vars_1, continuous_lower_bounds_1)
]

specs_orig = [
    wgan_original.Specifications(dw, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                                 critic_lr=CRITIC_LR, generator_lr=GENERATOR_LR,
                                 print_every=PRINT_EVERY, device=DEVICE)
    for dw in data_wrappers_orig
]

generators_orig = [wgan_original.Generator(spec) for spec in specs_orig]
critics_orig = [wgan_original.Critic(spec) for spec in specs_orig]

# Train Model 0 - Original
print("\n--- Training Model 0 (X|t) with ORIGINAL package ---")
torch.manual_seed(42)
start = time()
x, context = data_wrappers_orig[0].preprocess(df_balanced)
wgan_original.train(generators_orig[0], critics_orig[0], x, context, specs_orig[0])
time_orig_model0 = time() - start

# Train Model 1 - Original
print("\n--- Training Model 1 (Y|X,t) with ORIGINAL package ---")
torch.manual_seed(42)
start = time()
x, context = data_wrappers_orig[1].preprocess(df_balanced)
wgan_original.train(generators_orig[1], critics_orig[1], x, context, specs_orig[1])
time_orig_model1 = time() - start

# Generate with Original
print("\n--- Generating 1M samples with ORIGINAL package ---")
torch.manual_seed(42)
start = time()
df_gen_orig = data_wrappers_orig[0].apply_generator(generators_orig[0], df.sample(int(1e6), replace=True, random_state=42))
df_gen_orig = data_wrappers_orig[1].apply_generator(generators_orig[1], df_gen_orig)
df_gen_cf = copy(df_gen_orig)
df_gen_cf["t"] = 1 - df_gen_cf["t"]
df_gen_orig["re78_cf"] = data_wrappers_orig[1].apply_generator(generators_orig[1], df_gen_cf)["re78"]
time_gen_orig = time() - start
att_orig = ((df_gen_orig.re78 - df_gen_orig.re78_cf) * (2*df_gen_orig.t - 1))[df_gen_orig.t == 1].mean()

# ============================================================
# OPTIMIZED PACKAGE
# ============================================================
print("\n" + "=" * 70)
print("[3/6] BENCHMARKING OPTIMIZED PACKAGE")
print("=" * 70)

sys.path = [p for p in sys.path if 'ds-wgan' not in p or 'dswganoptx' in p]
sys.path.insert(0, '/Users/anzony.quisperojas/Documents/GitHub/python/dswganoptx')

if 'wgan' in sys.modules:
    del sys.modules['wgan']
if 'wgan.wgan' in sys.modules:
    del sys.modules['wgan.wgan']

import wgan as wgan_optimized

torch.manual_seed(42)
data_wrappers_opt = [
    wgan_optimized.DataWrapper(df_balanced, continuous_vars_0, categorical_vars_0,
                               context_vars_0, continuous_lower_bounds_0),
    wgan_optimized.DataWrapper(df_balanced, continuous_vars_1, categorical_vars_1,
                               context_vars_1, continuous_lower_bounds_1)
]

specs_opt = [
    wgan_optimized.Specifications(dw, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                                  critic_lr=CRITIC_LR, generator_lr=GENERATOR_LR,
                                  print_every=PRINT_EVERY, device=DEVICE)
    for dw in data_wrappers_opt
]

generators_opt = [wgan_optimized.Generator(spec) for spec in specs_opt]
critics_opt = [wgan_optimized.Critic(spec) for spec in specs_opt]

# Train Model 0 - Optimized
print("\n--- Training Model 0 (X|t) with OPTIMIZED package ---")
torch.manual_seed(42)
start = time()
x, context = data_wrappers_opt[0].preprocess(df_balanced)
wgan_optimized.train(generators_opt[0], critics_opt[0], x, context, specs_opt[0])
time_opt_model0 = time() - start

# Train Model 1 - Optimized
print("\n--- Training Model 1 (Y|X,t) with OPTIMIZED package ---")
torch.manual_seed(42)
start = time()
x, context = data_wrappers_opt[1].preprocess(df_balanced)
wgan_optimized.train(generators_opt[1], critics_opt[1], x, context, specs_opt[1])
time_opt_model1 = time() - start

# Generate with Optimized
print("\n--- Generating 1M samples with OPTIMIZED package ---")
torch.manual_seed(42)
start = time()
df_gen_opt = data_wrappers_opt[0].apply_generator(generators_opt[0], df.sample(int(1e6), replace=True, random_state=42))
df_gen_opt = data_wrappers_opt[1].apply_generator(generators_opt[1], df_gen_opt)
df_gen_cf = copy(df_gen_opt)
df_gen_cf["t"] = 1 - df_gen_cf["t"]
df_gen_opt["re78_cf"] = data_wrappers_opt[1].apply_generator(generators_opt[1], df_gen_cf)["re78"]
time_gen_opt = time() - start
att_opt = ((df_gen_opt.re78 - df_gen_opt.re78_cf) * (2*df_gen_opt.t - 1))[df_gen_opt.t == 1].mean()

# ============================================================
# RESULTS SUMMARY
# ============================================================
speedup_m0 = time_orig_model0 / time_opt_model0
speedup_m1 = time_orig_model1 / time_opt_model1
speedup_train = (time_orig_model0 + time_orig_model1) / (time_opt_model0 + time_opt_model1)
speedup_gen = time_gen_orig / time_gen_opt

time_saved = (time_orig_model0 + time_orig_model1 + time_gen_orig) - (time_opt_model0 + time_opt_model1 + time_gen_opt)
pct_saved = time_saved / (time_orig_model0 + time_orig_model1 + time_gen_orig) * 100

print("\n")
print("#" * 70)
print("#" + " " * 68 + "#")
print("#" + "           BENCHMARK RESULTS SUMMARY".center(68) + "#")
print("#" + " " * 68 + "#")
print("#" * 70)

print(f"\n{'Device:':<30} {DEVICE}")
print(f"{'Epochs:':<30} {MAX_EPOCHS}")
print(f"{'Batch Size:':<30} {BATCH_SIZE}")

print("\n" + "-" * 70)
print("                      TRAINING TIME (seconds)")
print("-" * 70)
print(f"{'Component':<25} {'Original':>12} {'Optimized':>12} {'Speedup':>12}")
print("-" * 70)
print(f"{'Model 0 (X|t)':<25} {time_orig_model0:>12.2f} {time_opt_model0:>12.2f} {speedup_m0:>11.2f}x")
print(f"{'Model 1 (Y|X,t)':<25} {time_orig_model1:>12.2f} {time_opt_model1:>12.2f} {speedup_m1:>11.2f}x")
print("-" * 70)
print(f"{'TOTAL TRAINING':<25} {time_orig_model0+time_orig_model1:>12.2f} {time_opt_model0+time_opt_model1:>12.2f} {speedup_train:>11.2f}x")

print("\n" + "-" * 70)
print("                    GENERATION TIME (1M samples)")
print("-" * 70)
print(f"{'Data Generation':<25} {time_gen_orig:>12.2f} {time_gen_opt:>12.2f} {speedup_gen:>11.2f}x")

print("\n" + "-" * 70)
print("                        RESULTS VALIDATION")
print("-" * 70)
print(f"{'ATT (Original):':<30} {att_orig:.2f}")
print(f"{'ATT (Optimized):':<30} {att_opt:.2f}")
print(f"{'ATT Difference:':<30} {abs(att_orig - att_opt):.2f}")

print("\n" + "=" * 70)
print("                         FINAL SUMMARY")
print("=" * 70)
print(f"\n  Total Original Time:    {time_orig_model0 + time_orig_model1 + time_gen_orig:.2f} seconds")
print(f"  Total Optimized Time:   {time_opt_model0 + time_opt_model1 + time_gen_opt:.2f} seconds")
print(f"  Time Saved:             {time_saved:.2f} seconds ({pct_saved:.1f}% reduction)")
print(f"  Overall Speedup:        {(time_orig_model0 + time_orig_model1 + time_gen_orig) / (time_opt_model0 + time_opt_model1 + time_gen_opt):.2f}x faster")
print("\n" + "=" * 70)
