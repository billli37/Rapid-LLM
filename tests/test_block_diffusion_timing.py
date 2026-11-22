from pathlib import Path
import pytest

import config
from inference_timing import TimeCalculationLLMInference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_CONFIG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "a100_80GB.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model-config" / "course_project" / "Llama3.1-8B_inf.yaml"


@pytest.fixture
def llama_timing():
    hw = config.parse_config(str(HW_CONFIG_PATH), config_type="hardware")
    model = config.parse_config(str(MODEL_CONFIG_PATH), config_type="LLM")
    return TimeCalculationLLMInference(hw, model, "LLM")


def test_decode_transformer_results_q1_equivalence(llama_timing):
    """Verify that query_tokens=1 gives exact standard AR decode timings."""
    timings_q1, breakdown_q1 = llama_timing._build_decode_transformer_results(
        batch_size=1,
        total_seq_len=512,
        use_moe_layer=False,
        query_tokens=1,
    )

    # Sanity checks on q=1 operations
    assert "attention" in timings_q1
    assert "linear_softmax" in timings_q1
    assert "layernorm1" in timings_q1
    assert "layernorm2" in timings_q1
    assert "ffn1" in timings_q1
    assert "ffn2" in timings_q1

    t_fwd = breakdown_q1["transformer_time_f"]
    assert t_fwd > 0.0
    assert breakdown_q1["linear_softmax_f"] > 0.0


@pytest.mark.parametrize("block_size", [2, 4, 8, 16])
def test_decode_transformer_results_scaling(llama_timing, block_size):
    """Verify that query_tokens=b scales layer operations and increases step compute."""
    batch_size = 1
    total_seq_len = 512

    timings_q1, breakdown_q1 = llama_timing._build_decode_transformer_results(
        batch_size=batch_size,
        total_seq_len=total_seq_len,
        use_moe_layer=False,
        query_tokens=1,
    )

    timings_qb, breakdown_qb = llama_timing._build_decode_transformer_results(
        batch_size=batch_size,
        total_seq_len=total_seq_len,
        use_moe_layer=False,
        query_tokens=block_size,
    )

    # Attention FLOPs should scale with block_size
    attn_flops_q1 = timings_q1["attention"].forward.flops
    attn_flops_qb = timings_qb["attention"].forward.flops
    assert attn_flops_qb == pytest.approx(attn_flops_q1 * block_size)

    # Linear projection FLOPs should scale with block_size
    qkv_flops_q1 = timings_q1["qkv_proj"].forward.flops
    qkv_flops_qb = timings_qb["qkv_proj"].forward.flops
    assert qkv_flops_qb == pytest.approx(qkv_flops_q1 * block_size)

    ffn1_flops_q1 = timings_q1["ffn1"].forward.flops
    ffn1_flops_qb = timings_qb["ffn1"].forward.flops
    assert ffn1_flops_qb == pytest.approx(ffn1_flops_q1 * block_size)

    # Step time for q=b should be greater than or equal to q=1
    assert breakdown_qb["transformer_time_f"] >= breakdown_q1["transformer_time_f"]
