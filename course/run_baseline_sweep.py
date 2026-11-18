#!/usr/bin/env python3
"""
Run standard autoregressive (AR) baseline evaluation across the required sweep matrix:
  - Hardware: a100_80GB_1gpu.yaml
  - Model: Llama3.1-8B_inf.yaml
  - Prompt length S = 512
  - Batch sizes B in {1, 16, 64}
  - Output lengths O in {128, 256}
"""

import os
import sys
import yaml
import tempfile
import subprocess
import json
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable

HW_CONFIG = os.path.join(REPO_ROOT, "configs/hardware-config/course_project/a100_80GB_1gpu.yaml")
BASE_MODEL_CONFIG = os.path.join(REPO_ROOT, "configs/model-config/course_project/Llama3.1-8B_inf.yaml")

SWEEP_BATCHES = [1, 16, 64]
SWEEP_OUTPUTS = [128, 256]
PROMPT_LEN = 512

def parse_run_output(stdout, results_text, memory_text):
    data = {}
    
    # Parse results_text
    m = re.search(r"Inference Time for batch:\s+([0-9.]+)s", results_text)
    if m: data["total_time_s"] = float(m.group(1))
    
    m = re.search(r"Prefill Time:\s+([0-9.]+)s", results_text)
    if m: data["prefill_time_s"] = float(m.group(1))
    
    m = re.search(r"Decode Time:\s+([0-9.]+)s", results_text)
    if m: data["decode_time_s"] = float(m.group(1))
    
    m = re.search(r"Time to First Token:\s+([0-9.]+)s", results_text)
    if m: data["ttft_s"] = float(m.group(1))
    
    m = re.search(r"Aggregate Decode Throughput Tok/s[^:]*:\s+start=([0-9.]+),\s+mid\([^)]+\)=([0-9.]+),\s+end=([0-9.]+)", results_text)
    if m:
        data["throughput_start"] = float(m.group(1))
        data["throughput_mid"] = float(m.group(2))
        data["throughput_end"] = float(m.group(3))
        
    # Parse memory text
    m = re.search(r"Prefill peak memory usage \(per gpu\):\s+([0-9.]+)\s+GiB", memory_text)
    if m: data["prefill_peak_mem_gib"] = float(m.group(1))
    
    m = re.search(r"Final decode peak memory usage \(per gpu\):\s+([0-9.]+)\s+GiB", memory_text)
    if m: data["decode_peak_mem_gib"] = float(m.group(1))
    
    m = re.search(r"Max peak memory usage \(per gpu\):\s+([0-9.]+)\s+GiB", memory_text)
    if m: data["max_peak_mem_gib"] = float(m.group(1))
    
    m = re.search(r"Remaining memory headroom:\s+([0-9.]+)\s+GiB", memory_text)
    if m: data["headroom_gib"] = float(m.group(1))

    # Parse stdout for KV cache info
    m = re.search(r"\[kv-cache\] prefill_store=([0-9.]+)\s+GiB,\s+decode_store=([0-9.]+)\s+GiB,\s+decode_fetch=([0-9.]+)\s+GiB", stdout)
    if m:
        data["kv_prefill_store_gib"] = float(m.group(1))
        data["kv_decode_store_gib"] = float(m.group(2))
        data["kv_decode_fetch_gib"] = float(m.group(3))

    return data

def run_single(batch_size, prompt_len, output_len):
    with open(BASE_MODEL_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg["model_param"]["global_batch_size"] = batch_size
    cfg["model_param"]["seq_len"] = prompt_len + output_len
    cfg["model_param"]["decode_len"] = output_len
    
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.dump(cfg, tf)
        tmp_model_path = tf.name

    try:
        cmd = [
            PYTHON,
            os.path.join(REPO_ROOT, "run_perf.py"),
            "--hardware_config", HW_CONFIG,
            "--model_config", tmp_model_path,
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error for B={batch_size}, O={output_len}:\n{proc.stderr}", file=sys.stderr)
            return None
            
        results_file = os.path.join(REPO_ROOT, "output", "LLM", "LLM_inference_results.txt")
        memory_file = os.path.join(REPO_ROOT, "output", "LLM", "memory-summary", "memory_capacity_comparison.txt")
        
        with open(results_file, "r") as f:
            results_text = f.read()
        with open(memory_file, "r") as f:
            memory_text = f.read()
            
        metrics = parse_run_output(proc.stdout, results_text, memory_text)
        metrics["batch_size"] = batch_size
        metrics["prompt_len"] = prompt_len
        metrics["output_len"] = output_len
        metrics["prefill_calls"] = 1
        metrics["decode_calls"] = output_len
        metrics["total_calls"] = 1 + output_len
        metrics["position_passes"] = 1 * prompt_len + output_len * 1
        
        # Calculate FLOPs analytically for Llama 3.1 8B
        # H=4096, L=32, n_heads=32, kv_heads=8, I=14336, V=128256
        H = 4096
        L = 32
        n_heads = 32
        kv_heads = 8
        d_k = H // n_heads # 128
        I = 14336
        V = 128256
        
        # Per-token linear GEMM FLOPs per layer:
        # QKV: 2 * (H*H + 2 * (kv_heads/n_heads)*H*H) = 2 * (4096*4096 + 2 * (1/4)*4096*4096) = 2 * 1.5 * H^2 = 3 * H^2
        # Out_proj: 2 * H^2
        # FFN (SwiGLU: gate, up, down): 2 * (2 * H * I + I * H) = 6 * H * I
        # Total per token per layer linear FLOPs = 5 * H^2 + 6 * H * I
        linear_flops_per_token_per_layer = 5 * (H**2) + 6 * H * I
        
        # Prefill FLOPs:
        # Linear: B * S * L * linear_flops_per_token_per_layer
        # Attention QK + AV (causal): B * L * 4 * n_heads * d_k * (S^2 / 2) = 2 * B * L * H * (S^2 / 2) = B * L * H * S^2
        # Unembedding: 2 * B * S * H * V
        prefill_flops = (
            batch_size * prompt_len * L * linear_flops_per_token_per_layer +
            batch_size * L * H * (prompt_len ** 2) +
            2 * batch_size * prompt_len * H * V
        )
        
        # Decode FLOPs for O tokens:
        # For each token step i from 0 to O-1, context length is S + i:
        # Linear: B * O * L * linear_flops_per_token_per_layer
        # Attention QK + AV: sum_{i=0}^{O-1} B * L * 4 * n_heads * d_k * (S + i) = 4 * B * L * H * sum_{i=0}^{O-1} (S + i)
        # Unembedding: 2 * B * O * H * V
        total_context_sum = sum(prompt_len + i for i in range(output_len))
        decode_flops = (
            batch_size * output_len * L * linear_flops_per_token_per_layer +
            4 * batch_size * L * H * total_context_sum +
            2 * batch_size * output_len * H * V
        )
        total_flops = prefill_flops + decode_flops
        
        metrics["prefill_tflops"] = prefill_flops / 1e12
        metrics["decode_tflops"] = decode_flops / 1e12
        metrics["total_tflops"] = total_flops / 1e12
        
        return metrics
    finally:
        if os.path.exists(tmp_model_path):
            os.remove(tmp_model_path)

def main():
    print("=" * 80)
    print("Running Autoregressive (AR) Decode Baseline Evaluation")
    print(f"Model: Llama3.1-8B | Hardware: 1x A100-80GB | Prompt Length S = {PROMPT_LEN}")
    print("=" * 80)
    
    results = []
    for B in SWEEP_BATCHES:
        for O in SWEEP_OUTPUTS:
            print(f"Running baseline B={B:2d}, S={PROMPT_LEN}, O={O:3d}...", end=" ", flush=True)
            res = run_single(B, PROMPT_LEN, O)
            if res:
                results.append(res)
                print(f"Done! Total time: {res['total_time_s']:.3f}s, TTFT: {res['ttft_s']:.3f}s, Max Mem: {res['max_peak_mem_gib']:.2f} GiB")
            else:
                print("FAILED!")
                
    output_json = os.path.join(REPO_ROOT, "course", "baseline_ar_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll baseline results saved to {output_json}")

if __name__ == "__main__":
    main()
