import pytest
from types import SimpleNamespace

import llm_util


def _build_dummy_ctx(
    use_moe=False,
    moe_num_experts=1,
    attention_type="mha",
    head_dim=128,
    swiglu_mlp=True,
    run_type="inference",
):
    return SimpleNamespace(
        use_moe=use_moe,
        moe_num_experts=moe_num_experts,
        attention_type=attention_type,
        head_dim=head_dim,
        swiglu_mlp=swiglu_mlp,
        run_type=run_type,
        model_type="llama",
    )


def test_autoregressive_decoder_gemm_q1_equivalence():
    """Verify that query_tokens=1 produces standard single-token decode shapes."""
    ctx = _build_dummy_ctx()
    batch_size = 2
    current_seq_len = 512
    d_model = 4096
    num_heads = 32
    kv_heads = 8
    intermediate_size = 14336
    vocab_size = 128256

    gemms_4d = llm_util.autoregressive_decoder_gemm(
        ctx,
        batch_size=batch_size,
        current_seq_len=current_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        kv_heads=kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        model_type="llama",
        query_tokens=1,
    )

    # 4D raw shapes
    assert gemms_4d["qkv_proj"] == (batch_size, 1, d_model, 4096 + 2 * 1024)
    # shared_heads = 32 // 8 = 4
    assert gemms_4d["attention_score"] == (batch_size * kv_heads, 4, 128, current_seq_len)
    assert gemms_4d["attention_output"] == (batch_size * kv_heads, 4, current_seq_len, 128)
    assert gemms_4d["output_proj"] == (batch_size, 1, 4096, d_model)
    assert gemms_4d["ffn1"] == (batch_size, 1, d_model, 2 * intermediate_size)
    assert gemms_4d["ffn2"] == (batch_size, 1, intermediate_size, d_model)
    assert gemms_4d["linear"] == (batch_size, 1, d_model, vocab_size)

    # 3D processed shapes
    processed = llm_util.process_decode_gemm_shapes(
        ctx,
        batch_size=batch_size,
        current_seq_len=current_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        kv_heads=kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        model_type="llama",
        query_tokens=1,
    )

    assert processed["qkv_proj"] == (batch_size * 1, d_model, 4096 + 2 * 1024)
    assert processed["attention_score"] == (batch_size * kv_heads, 4, 128, current_seq_len)
    assert processed["attention_output"] == (batch_size * kv_heads, 4, current_seq_len, 128)
    assert processed["output_proj"] == (batch_size * 1, 4096, d_model)
    assert processed["ffn1"] == (batch_size * 1, d_model, 2 * intermediate_size)
    assert processed["ffn2"] == (batch_size * 1, intermediate_size, d_model)
    assert processed["linear"] == (batch_size * 1, d_model, vocab_size)


@pytest.mark.parametrize("block_size", [2, 4, 8, 16, 32])
def test_block_diffusion_gemm_scaling(block_size):
    """Verify that query_tokens=b scales all M dimensions and attention query dims by b."""
    ctx = _build_dummy_ctx()
    batch_size = 4
    current_seq_len = 1024
    d_model = 4096
    num_heads = 32
    kv_heads = 8
    intermediate_size = 14336
    vocab_size = 128256

    processed = llm_util.process_decode_gemm_shapes(
        ctx,
        batch_size=batch_size,
        current_seq_len=current_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        kv_heads=kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        model_type="llama",
        query_tokens=block_size,
    )

    shared_heads = num_heads // kv_heads

    # Reshaped 3D GEMMs must have M = batch_size * block_size
    expected_m = batch_size * block_size
    assert processed["qkv_proj"][0] == expected_m
    assert processed["output_proj"][0] == expected_m
    assert processed["ffn1"][0] == expected_m
    assert processed["ffn2"][0] == expected_m
    assert processed["linear"][0] == expected_m

    # Attention score and output must have query dimension = block_size * shared_heads
    expected_attn_m = block_size * shared_heads
    assert processed["attention_score"] == (batch_size * kv_heads, expected_attn_m, 128, current_seq_len)
    assert processed["attention_output"] == (batch_size * kv_heads, expected_attn_m, current_seq_len, 128)


def test_invalid_query_tokens_raises():
    """Verify that non-positive or non-integer query_tokens raise AssertionError."""
    ctx = _build_dummy_ctx()
    kwargs = dict(
        batch_size=1,
        current_seq_len=128,
        d_model=4096,
        num_heads=32,
        kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        model_type="llama",
    )

    for invalid_q in [0, -1, -5, "8", 4.0, None]:
        with pytest.raises(AssertionError):
            llm_util.autoregressive_decoder_gemm(ctx, **kwargs, query_tokens=invalid_q)
