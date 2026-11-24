#!/usr/bin/env python3
"""
Block diffusion eval
Runs:
  1. Block Diffusion (b=8, R=4) vs AR decode across B in {1, 16, 64}, S = 512, O in {128, 256} on single A100-80GB.
  2. Case Study 1: Parameter sweep over (b, R) combinations.
  3. Case Study 2: Context length scaling S in {512, 2K, 8K, 32K}.
  4. Case Study 3: Batch scaling B in {1..256} up to memory boundary (Roofline crossover).
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

def parse_run_output(stdout, results_text, memory_text):
    data = {}
    
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
        
    m = re.search(r"Prefill peak memory usage \(per gpu\):\s+([0-9.]+)\s+GiB", memory_text)
    if m: data["prefill_peak_mem_gib"] = float(m.group(1))
    
    m = re.search(r"Final decode peak memory usage \(per gpu\):\s+([0-9.]+)\s+GiB", memory_text)
    if m: data["decode_peak_mem_gib"] = float(m.group(1))
    
    m = re.search(r"Max peak memory usage \(per gpu\):\s+([0-9.]+)\s+GiB", memory_text)
    if m: data["max_peak_mem_gib"] = float(m.group(1))
    
    m = re.search(r"Remaining memory headroom:\s+([0-9.]+)\s+GiB", memory_text)
    if m: data["headroom_gib"] = float(m.group(1))

    def _parse_val_unit(val_str, unit_str):
        v = float(val_str)
        if unit_str == "TiB":
            return v * 1024.0
        elif unit_str == "GiB":
            return v
        elif unit_str == "MiB":
            return v / 1024.0
        elif unit_str == "KiB":
            return v / (1024.0 * 1024.0)
        return v

    m = re.search(r"\[kv-cache\] prefill_store=([0-9.]+)\s*(\w+),\s+decode_store=([0-9.]+)\s*(\w+),\s+decode_fetch=([0-9.]+)\s*(\w+)", stdout)
    if m:
        data["kv_prefill_store_gib"] = _parse_val_unit(m.group(1), m.group(2))
        data["kv_decode_store_gib"] = _parse_val_unit(m.group(3), m.group(4))
        data["kv_decode_fetch_gib"] = _parse_val_unit(m.group(5), m.group(6))

    return data


def run_model_sim(batch_size, prompt_len, output_len, decode_schedule="autoregressive", block_size=1, refine_rounds=0):
    with open(BASE_MODEL_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg["model_param"]["global_batch_size"] = batch_size
    cfg["model_param"]["seq_len"] = prompt_len + output_len
    cfg["model_param"]["decode_len"] = output_len
    
    if decode_schedule == "block_diffusion":
        if "inference_param" not in cfg:
            cfg["inference_param"] = {}
        cfg["inference_param"]["decode_schedule"] = "block_diffusion"
        cfg["inference_param"]["block_diffusion"] = {
            "block_size": block_size,
            "refine_rounds": refine_rounds,
        }
    else:
        if "inference_param" in cfg and "block_diffusion" in cfg["inference_param"]:
            del cfg["inference_param"]["block_diffusion"]
        if "inference_param" in cfg:
            cfg["inference_param"]["decode_schedule"] = "autoregressive"

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
            print(f"Error for B={batch_size}, S={prompt_len}, O={output_len}, sched={decode_schedule}:\n{proc.stderr}", file=sys.stderr)
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
        metrics["decode_schedule"] = decode_schedule
        metrics["block_size"] = block_size
        metrics["refine_rounds"] = refine_rounds
        
        # Call count logic
        if decode_schedule == "block_diffusion":
            n_blocks = output_len // block_size
            decode_calls = n_blocks * (refine_rounds + 1)
        else:
            decode_calls = output_len
        metrics["prefill_calls"] = 1
        metrics["decode_calls"] = decode_calls
        metrics["total_calls"] = 1 + decode_calls
        
        # FLOPs calculation for Llama 3.1 8B
        H = 4096
        L = 32
        I = 14336
        V = 128256
        linear_flops_per_token_per_layer = 5 * (H**2) + 6 * H * I
        
        prefill_flops = (
            batch_size * prompt_len * L * linear_flops_per_token_per_layer +
            batch_size * L * H * (prompt_len ** 2) +
            2 * batch_size * prompt_len * H * V
        )
        
        if decode_schedule == "block_diffusion":
            n_blocks = output_len // block_size
            passes = refine_rounds + 1
            linear_decode = passes * batch_size * output_len * L * linear_flops_per_token_per_layer
            unembed_decode = passes * 2 * batch_size * output_len * H * V
            context_sum = sum(block_size * (prompt_len + (k + 1) * block_size) for k in range(n_blocks))
            attn_decode = passes * 4 * batch_size * L * H * context_sum
            decode_flops = linear_decode + attn_decode + unembed_decode
        else:
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


def run_required_baseline_evaluation():
    print("=" * 80)
    print("1. REQUIRED BASELINE EVALUATION: Block Diffusion (b=8, R=4) vs Standard AR")
    print("   Hardware: 1x A100-80GB | Prompt Length S = 512")
    print("   Sweep Matrix: B in {1, 16, 64}, O in {128, 256}")
    print("=" * 80)
    
    batches = [1, 16, 64]
    outputs = [128, 256]
    prompt_len = 512
    
    comparisons = []
    
    for B in batches:
        for O in outputs:
            print(f"--> Running B={B:2d}, S={prompt_len}, O={O:3d}...")
            print("    [AR] Running autoregressive baseline...", end=" ", flush=True)
            res_ar = run_model_sim(B, prompt_len, O, decode_schedule="autoregressive")
            print(f"Done! Time: {res_ar['total_time_s']:.3f}s, Calls: {res_ar['total_calls']}")
            
            print("    [BD] Running block diffusion (b=8, R=4)...", end=" ", flush=True)
            res_bd = run_model_sim(B, prompt_len, O, decode_schedule="block_diffusion", block_size=8, refine_rounds=4)
            print(f"Done! Time: {res_bd['total_time_s']:.3f}s, Calls: {res_bd['total_calls']}")
            
            speedup = res_ar["total_time_s"] / res_bd["total_time_s"]
            decode_speedup = res_ar["decode_time_s"] / res_bd["decode_time_s"]
            call_reduction = (1.0 - (res_bd["total_calls"] / res_ar["total_calls"])) * 100.0
            kv_reduction = (1.0 - (res_bd["kv_decode_fetch_gib"] / res_ar["kv_decode_fetch_gib"])) * 100.0
            
            cmp_item = {
                "batch_size": B,
                "prompt_len": prompt_len,
                "output_len": O,
                "ar": res_ar,
                "bd": res_bd,
                "speedup_total": round(speedup, 3),
                "speedup_decode": round(decode_speedup, 3),
                "call_reduction_pct": round(call_reduction, 1),
                "kv_reduction_pct": round(kv_reduction, 1),
            }
            comparisons.append(cmp_item)
            print(f"    ==> Speedup: {speedup:.2f}x total ({decode_speedup:.2f}x decode), Calls: -{call_reduction:.1f}%\n")
            
    out_path = os.path.join(REPO_ROOT, "course", "required_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"Saved required evaluation results to {out_path}")
    return comparisons


def run_case_study_param_sweep():
    print("=" * 80)
    print("2. CASE STUDY 1: (b, R) Parameter Sweep")
    print("   Fixed: B = 1, S = 512, O = 256 | Varying (b, R) with R < b")
    print("=" * 80)
    
    B = 1
    S = 512
    O = 256
    
    # Run AR baseline once for reference
    res_ar = run_model_sim(B, S, O, decode_schedule="autoregressive")
    
    param_pairs = [
        (4, 2),
        (8, 2),
        (8, 4),
        (16, 2),
        (16, 4),
        (16, 8),
        (32, 4),
        (32, 8),
    ]
    
    results = []
    for b, R in param_pairs:
        print(f"--> Running (b={b:2d}, R={R:2d})...", end=" ", flush=True)
        res = run_model_sim(B, S, O, decode_schedule="block_diffusion", block_size=b, refine_rounds=R)
        speedup = res_ar["total_time_s"] / res["total_time_s"]
        decode_speedup = res_ar["decode_time_s"] / res["decode_time_s"]
        item = {
            "block_size": b,
            "refine_rounds": R,
            "passes_per_token": (R + 1) / b,
            "result": res,
            "speedup_total": round(speedup, 3),
            "speedup_decode": round(decode_speedup, 3),
        }
        results.append(item)
        print(f"Done! Total: {res['total_time_s']:.3f}s (Decode: {res['decode_time_s']:.3f}s), Speedup: {speedup:.2f}x, Calls: {res['total_calls']}")
        
    out_path = os.path.join(REPO_ROOT, "course", "case_study_param_sweep.json")
    with open(out_path, "w") as f:
        json.dump({"ar_baseline": res_ar, "param_sweep": results}, f, indent=2)
    print(f"Saved Case Study 1 results to {out_path}\n")
    return results


def run_case_study_context_sweep():
    print("=" * 80)
    print("3. CASE STUDY 2: Long Context Prompt Scaling (S)")
    print("   Fixed: B = 1, O = 128 | b = 8, R = 4 vs AR")
    print("   Varying Prompt Length S in {512, 2048, 8192, 32768}")
    print("=" * 80)
    
    B = 1
    O = 128
    contexts = [512, 2048, 8192, 32768]
    
    results = []
    for S in contexts:
        print(f"--> Running Context Length S = {S}...")
        print("    [AR] Running...", end=" ", flush=True)
        res_ar = run_model_sim(B, S, O, decode_schedule="autoregressive")
        print(f"Done! Decode: {res_ar['decode_time_s']:.3f}s, KV fetch: {res_ar['kv_decode_fetch_gib']} GiB")
        
        print("    [BD] Running (b=8, R=4)...", end=" ", flush=True)
        res_bd = run_model_sim(B, S, O, decode_schedule="block_diffusion", block_size=8, refine_rounds=4)
        print(f"Done! Decode: {res_bd['decode_time_s']:.3f}s, KV fetch: {res_bd['kv_decode_fetch_gib']} GiB")
        
        speedup = res_ar["total_time_s"] / res_bd["total_time_s"]
        decode_speedup = res_ar["decode_time_s"] / res_bd["decode_time_s"]
        
        results.append({
            "prompt_len": S,
            "ar": res_ar,
            "bd": res_bd,
            "speedup_total": round(speedup, 3),
            "speedup_decode": round(decode_speedup, 3),
            "kv_traffic_reduction_pct": round((1.0 - res_bd["kv_decode_fetch_gib"] / res_ar["kv_decode_fetch_gib"]) * 100.0, 1),
        })
        print(f"    ==> Decode Speedup: {decode_speedup:.2f}x\n")
        
    out_path = os.path.join(REPO_ROOT, "course", "case_study_context_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Case Study 2 results to {out_path}\n")
    return results


def run_case_study_batch_scaling():
    print("=" * 80)
    print("4. CASE STUDY 3: Batch Scaling & Roofline Transition Analysis")
    print("   Fixed: S = 512, O = 128 | b = 8, R = 4 vs AR")
    print("   Varying Batch Size B in {1, 4, 8, 16, 32, 64, 128, 256}")
    print("=" * 80)
    
    S = 512
    O = 128
    batches = [1, 4, 8, 16, 32, 64, 128, 256]
    
    results = []
    for B in batches:
        print(f"--> Testing Batch Size B = {B:3d}...", end=" ", flush=True)
        res_ar = run_model_sim(B, S, O, decode_schedule="autoregressive")
        if res_ar is None:
            print("AR OOM or Failed!")
            break
        res_bd = run_model_sim(B, S, O, decode_schedule="block_diffusion", block_size=8, refine_rounds=4)
        if res_bd is None:
            print("BD OOM or Failed!")
            break
            
        speedup = res_ar["total_time_s"] / res_bd["total_time_s"]
        decode_speedup = res_ar["decode_time_s"] / res_bd["decode_time_s"]
        
        item = {
            "batch_size": B,
            "ar": res_ar,
            "bd": res_bd,
            "speedup_total": round(speedup, 3),
            "speedup_decode": round(decode_speedup, 3),
            "ar_tok_s": res_ar["throughput_mid"],
            "bd_tok_s": res_bd["throughput_mid"],
        }
        results.append(item)
        print(f"Done! AR: {res_ar['decode_time_s']:.3f}s, BD: {res_bd['decode_time_s']:.3f}s => Decode Speedup: {decode_speedup:.2f}x | Peak Mem: {res_bd['max_peak_mem_gib']:.1f} GiB")
        
    out_path = os.path.join(REPO_ROOT, "course", "case_study_batch_scaling.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Case Study 3 results to {out_path}\n")
    return results


def main():
    print("Starting Block Diffusion Evaluation Suite...")
    os.makedirs(os.path.join(REPO_ROOT, "course"), exist_ok=True)
    
    req_res = run_required_baseline_evaluation()
    p_res = run_case_study_param_sweep()
    c_res = run_case_study_context_sweep()
    b_res = run_case_study_batch_scaling()
    
    print("=" * 80)
    print("ALL EVALUATIONS AND CASE STUDIES COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
