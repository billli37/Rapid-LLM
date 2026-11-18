#!/usr/bin/env python3
"""
Regression test harness for ECE 201C course projects.

Usage:
    python course/regress.py record
        Runs the course models on a100_80GB_1gpu and records the baseline outputs.

    python course/regress.py check
        Re-runs the models and verifies that the results match the recorded baselines
        byte-for-byte (zero regression). Exits with code 1 on mismatch.
"""

import sys
import os
import subprocess
import shutil
import tempfile
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable

HW_CONFIG = os.path.join(REPO_ROOT, "configs/hardware-config/course_project/a100_80GB_1gpu.yaml")
BASELINE_DIR = os.path.join(REPO_ROOT, "course", ".baselines")

MODELS = [
    ("Llama3.1-8B", os.path.join(REPO_ROOT, "configs/model-config/course_project/Llama3.1-8B_inf.yaml"), False),
    ("qwen3_30b_a3b", os.path.join(REPO_ROOT, "configs/model-config/course_project/qwen3_30b_a3b_inf.yaml"), False),
    ("mimo_7b", os.path.join(REPO_ROOT, "configs/model-config/course_project/mimo_7b_inf.yaml"), False),
    ("gemma_3_4b_dense", os.path.join(REPO_ROOT, "configs/model-config/course_project/gemma_3_4b_inf.yaml"), True),
]

def run_model(model_path, strip_window=False):
    """Run run_perf.py with the given model config and return results dict."""
    tmp_config = None
    actual_model_path = model_path
    
    if strip_window:
        with open(model_path, "r") as f:
            cfg = yaml.safe_load(f)
        if "window" in cfg.get("model_param", {}).get("attention", {}):
            del cfg["model_param"]["attention"]["window"]
        tf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.dump(cfg, tf)
        tf.close()
        actual_model_path = tf.name
        tmp_config = actual_model_path

    try:
        cmd = [
            PYTHON,
            os.path.join(REPO_ROOT, "run_perf.py"),
            "--hardware_config", HW_CONFIG,
            "--model_config", actual_model_path,
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error running model {model_path}:\n{proc.stderr}", file=sys.stderr)
            sys.exit(proc.returncode)

        results_file = os.path.join(REPO_ROOT, "output", "LLM", "LLM_inference_results.txt")
        memory_file = os.path.join(REPO_ROOT, "output", "LLM", "memory-summary", "memory_capacity_comparison.txt")
        
        results_content = ""
        memory_content = ""
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results_content = f.read()
        if os.path.exists(memory_file):
            with open(memory_file, "r") as f:
                memory_content = f.read()

        return {
            "results": results_content,
            "memory": memory_content,
            "stdout": proc.stdout,
        }
    finally:
        if tmp_config and os.path.exists(tmp_config):
            os.remove(tmp_config)


def record():
    """Run all models and save their outputs."""
    print("Recording baselines for course models on a100_80GB_1gpu...")
    os.makedirs(BASELINE_DIR, exist_ok=True)
    
    for name, path, strip_win in MODELS:
        print(f"  Recording {name}...")
        out = run_model(path, strip_window=strip_win)
        model_dir = os.path.join(BASELINE_DIR, name)
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, "LLM_inference_results.txt"), "w") as f:
            f.write(out["results"])
        with open(os.path.join(model_dir, "memory_capacity_comparison.txt"), "w") as f:
            f.write(out["memory"])
        with open(os.path.join(model_dir, "stdout.txt"), "w") as f:
            f.write(out["stdout"])
            
    print(f"Baselines successfully recorded in {BASELINE_DIR}")


def check():
    """Run all models and compare to recorded baselines."""
    if not os.path.exists(BASELINE_DIR):
        print(f"Error: Baseline directory {BASELINE_DIR} does not exist. Run 'record' first.", file=sys.stderr)
        sys.exit(1)

    print("Checking course models against recorded baselines...")
    all_passed = True
    
    for name, path, strip_win in MODELS:
        print(f"  Checking {name}...", end=" ", flush=True)
        model_dir = os.path.join(BASELINE_DIR, name)
        if not os.path.exists(model_dir):
            print(f"FAILED (no baseline found)")
            all_passed = False
            continue
            
        with open(os.path.join(model_dir, "LLM_inference_results.txt"), "r") as f:
            expected_results = f.read()
        with open(os.path.join(model_dir, "memory_capacity_comparison.txt"), "r") as f:
            expected_memory = f.read()

        out = run_model(path, strip_window=strip_win)
        
        diffs = []
        if out["results"] != expected_results:
            diffs.append("LLM_inference_results.txt mismatch")
        if out["memory"] != expected_memory:
            diffs.append("memory_capacity_comparison.txt mismatch")
            
        if diffs:
            print(f"FAILED: {', '.join(diffs)}")
            all_passed = False
        else:
            print("OK")

    if all_passed:
        print("Regression check PASSED: all models matched baselines byte-for-byte.")
        sys.exit(0)
    else:
        print("Regression check FAILED: output differed from baseline.", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "check"):
        print(__doc__)
        sys.exit(1)
        
    action = sys.argv[1]
    if action == "record":
        record()
    elif action == "check":
        check()

if __name__ == "__main__":
    main()
