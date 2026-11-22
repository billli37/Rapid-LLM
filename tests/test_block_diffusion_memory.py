import copy
from pathlib import Path
import pytest

import config
import llm_util
from inference_timing import TimeCalculationLLMInference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware-config" / "course_project" / "a100_80GB_1gpu.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model-config" / "course_project" / "Llama3.1-8B_inf.yaml"


@pytest.fixture
def a100_hw():
    return config.parse_config(str(HW_CONFIG_PATH), config_type="hardware")


@pytest.fixture
def llama_model():
    return config.parse_config(str(MODEL_CONFIG_PATH), config_type="LLM")


def test_neutral_memory_equivalence(a100_hw, llama_model):
    """Verify that peak memory for neutral block diffusion (b=1, R=0) matches AR decode."""
    tc_ar = TimeCalculationLLMInference(a100_hw, llama_model, "LLM")
    tc_ar.calc_time()
    peak_ar = tc_ar.memory_peak_gb

    model_bd1 = copy.deepcopy(llama_model)
    model_bd1.project_param = {
        "decode_schedule": "block_diffusion",
        "block_size": 1,
        "refine_rounds": 0,
    }
    tc_bd1 = TimeCalculationLLMInference(a100_hw, model_bd1, "LLM")
    tc_bd1.calc_time()
    peak_bd1 = tc_bd1.memory_peak_gb

    assert abs(peak_ar - peak_bd1) < 1e-4


def test_kv_cache_fetch_traffic_formula(a100_hw, llama_model):
    """Verify that decode KV cache fetch bytes follows the (R+1) block formula."""
    model = copy.deepcopy(llama_model)
    decode_len = 128
    prompt_len = 512
    block_size = 8
    refine_rounds = 4

    model.model_config.decode_len = decode_len
    model.model_config.seq_len = prompt_len + decode_len
    model.project_param = {
        "decode_schedule": "block_diffusion",
        "block_size": block_size,
        "refine_rounds": refine_rounds,
    }
    tc = TimeCalculationLLMInference(a100_hw, model, "LLM")
    res = tc.calc_total_inference_time()

    token_bytes = llm_util.attention_kv_cache_token_bytes(
        "mha",
        batch_size=1,
        kv_heads=8,
        head_dim=128,
        precision_bytes=tc.precision.kv_cache,
    )
    n_blocks = decode_len // block_size
    expected_fetch_tokens = (refine_rounds + 1) * (
        n_blocks * prompt_len + block_size * n_blocks * (n_blocks - 1) // 2
    )
    expected_bytes = token_bytes * 32 * expected_fetch_tokens

    assert res["kv_cache_decode_fetch_bytes"] == pytest.approx(expected_bytes)


def test_estimate_inference_memory_block_diffusion(a100_hw, llama_model):
    """Verify that llm_util.estimate_inference_memory runs cleanly with block diffusion."""
    model_bd = copy.deepcopy(llama_model)
    model_bd.project_param = {
        "decode_schedule": "block_diffusion",
        "block_size": 8,
        "refine_rounds": 4,
    }
    res = llm_util.estimate_inference_memory(
        a100_hw,
        model_bd,
        batch_size=1,
    )
    assert res["max_peak_gb"] > 0.0
    assert res["headroom_gb"] > 0.0
