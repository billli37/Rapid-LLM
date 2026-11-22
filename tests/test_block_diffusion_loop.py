import copy
from pathlib import Path
import pytest

import config
from inference_timing import TimeCalculationLLMInference
from simulate_inference_graph import InferenceConfig, DecodeGraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware-config" / "course_project" / "a100_80GB_1gpu.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model-config" / "course_project" / "Llama3.1-8B_inf.yaml"


@pytest.fixture
def a100_hw():
    return config.parse_config(str(HW_CONFIG_PATH), config_type="hardware")


@pytest.fixture
def llama_model():
    return config.parse_config(str(MODEL_CONFIG_PATH), config_type="LLM")


def test_neutral_equivalence_ar_vs_bd(a100_hw, llama_model):
    """Verify that neutral Block Diffusion (b=1, R=0) matches AR decode."""
    tc_ar = TimeCalculationLLMInference(a100_hw, llama_model, "LLM")
    res_ar = tc_ar.calc_total_inference_time()

    model_bd = copy.deepcopy(llama_model)
    model_bd.project_param = {
        "decode_schedule": "block_diffusion",
        "block_size": 1,
        "refine_rounds": 0,
    }
    tc_bd = TimeCalculationLLMInference(a100_hw, model_bd, "LLM")
    res_bd = tc_bd.calc_total_inference_time()

    assert abs(res_ar["total_inference_time"] - res_bd["total_inference_time"]) < 1e-6
    assert abs(res_ar["decode_time"] - res_bd["decode_time"]) < 1e-6
    assert abs(res_ar["time_to_first_token"] - res_bd["time_to_first_token"]) < 1e-6
    assert res_ar["kv_cache_decode_fetch_bytes"] == res_bd["kv_cache_decode_fetch_bytes"]

    rates_ar = res_ar["decode_tokens_per_s"]
    rates_bd = res_bd["decode_tokens_per_s"]
    for k in ["start", "midpoint", "end", "overall"]:
        assert abs(rates_ar[k] - rates_bd[k]) < 1e-6


@pytest.mark.parametrize(
    "decode_len,block_size,refine_rounds,expected_calls",
    [
        (128, 1, 0, 129),
        (128, 8, 4, 81),    # (128 / 8) * (4 + 1) + 1 = 16 * 5 + 1 = 81
        (256, 8, 4, 161),   # (256 / 8) * 5 + 1 = 32 * 5 + 1 = 161
        (128, 16, 2, 25),   # (128 / 16) * 3 + 1 = 8 * 3 + 1 = 25
        (256, 4, 2, 193),   # (256 / 4) * 3 + 1 = 64 * 3 + 1 = 193
    ],
)
def test_block_diffusion_call_counts(a100_hw, llama_model, decode_len, block_size, refine_rounds, expected_calls):
    """Verify that model_calls matches 1 + (decode_len / b) * (R + 1)."""
    model = copy.deepcopy(llama_model)
    model.model_config.decode_len = decode_len
    model.model_config.seq_len = 512 + decode_len
    model.project_param = {
        "decode_schedule": "block_diffusion",
        "block_size": block_size,
        "refine_rounds": refine_rounds,
    }
    tc = TimeCalculationLLMInference(a100_hw, model, "LLM")
    res = tc.calc_total_inference_time()
    assert res["model_calls"] == expected_calls
    expected_decode_calls = expected_calls - 1
    assert res["decode_calls"] == expected_decode_calls


def test_block_diffusion_sample_points(a100_hw, llama_model):
    """Verify that _generate_sample_points generates correct block indices."""
    cfg_ar = InferenceConfig(
        batch_size=1,
        seq_len=512,
        decode_len=128,
        hidden_dim=4096,
        num_heads=32,
        kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        num_layers=32,
        use_moe=False,
        num_experts=1,
        top_k=1,
        sample_every=32,
        decode_schedule="autoregressive",
        block_size=1,
        refine_rounds=0,
    )
    dg_ar = DecodeGraph(cfg_ar, a100_hw, llama_model, TimeCalculationLLMInference, False, 1, 1)
    pts_ar = dg_ar._generate_sample_points()

    # b=1 block diffusion should generate identical sample points
    cfg_bd1 = copy.deepcopy(cfg_ar)
    cfg_bd1.decode_schedule = "block_diffusion"
    dg_bd1 = DecodeGraph(cfg_bd1, a100_hw, llama_model, TimeCalculationLLMInference, False, 1, 1)
    pts_bd1 = dg_bd1._generate_sample_points()
    assert pts_ar == pts_bd1

    # b=8 block diffusion (16 blocks, interval 32 // 8 = 4)
    cfg_bd8 = copy.deepcopy(cfg_ar)
    cfg_bd8.decode_schedule = "block_diffusion"
    cfg_bd8.block_size = 8
    cfg_bd8.refine_rounds = 4
    dg_bd8 = DecodeGraph(cfg_bd8, a100_hw, llama_model, TimeCalculationLLMInference, False, 1, 1)
    pts_bd8 = dg_bd8._generate_sample_points()
    assert pts_bd8 == [0, 3, 7, 11, 15]


def test_block_diffusion_speedup_at_batch_1(a100_hw, llama_model):
    """Verify that block diffusion (b=8, R=4) achieves speedup over AR decode at B=1."""
    model_ar = copy.deepcopy(llama_model)
    model_ar.model_config.batch_size = 1
    model_ar.model_config.decode_len = 128
    model_ar.model_config.seq_len = 512 + 128
    tc_ar = TimeCalculationLLMInference(a100_hw, model_ar, "LLM")
    res_ar = tc_ar.calc_total_inference_time()

    model_bd = copy.deepcopy(model_ar)
    model_bd.project_param = {
        "decode_schedule": "block_diffusion",
        "block_size": 8,
        "refine_rounds": 4,
    }
    tc_bd = TimeCalculationLLMInference(a100_hw, model_bd, "LLM")
    res_bd = tc_bd.calc_total_inference_time()
    assert res_bd["decode_time"] < res_ar["decode_time"]
    speedup = res_ar["decode_time"] / res_bd["decode_time"]
    assert speedup > 1.25
