
from transformers import AutoConfig
import numpy as np
import argparse

def cache_bytes(config, batch_size, seq_len, dtype_size=2, num_layers=None):
    num_layers = config.num_hidden_layers if num_layers is None else num_layers
    hidden_size = config.head_dim * config.num_attention_heads // config.num_kv_groups
    return 2 * batch_size * seq_len * num_layers * hidden_size * dtype_size

def hidden_bytes(config, batch_size, seq_len, dtype_size=2):
    return batch_size * seq_len * config.hidden_size * dtype_size

def get_model_config(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model_name = model_path.split('/')[-1]
    hf_model_type = getattr(config, 'model_type', '')
    model_config = argparse.Namespace()
    if hf_model_type == 'qwen3_moe':
        model_config.model_type = 'qwen3_moe'
        model_config.attention_bias = config.attention_bias
        model_config.intermediate_size = config.intermediate_size
        model_config.moe_intermediate_size = config.moe_intermediate_size
        model_config.hidden_act = config.hidden_act
        assert config.max_position_embeddings >= 32768, f"{config.max_position_embeddings}"
        model_config.max_position_embeddings = config.max_position_embeddings
        model_config.tie_word_embeddings = config.tie_word_embeddings
        model_config.pad_token_id = config.pad_token_id if config.pad_token_id is not None else 151645
        model_config.rope_scaling = config.rope_scaling
        model_config.rope_theta = config.rope_theta
        model_config.num_experts = config.num_experts
        model_config.num_experts_per_tok = config.num_experts_per_tok
        model_config.norm_topk_prob = config.norm_topk_prob
        model_config.decoder_sparse_step = config.decoder_sparse_step
        model_config.mlp_only_layers = list(config.mlp_only_layers or [])
    elif 'opt' in model_name.lower():
        model_config.model_type = 'opt'
        model_config.max_position_embeddings = 2048
        model_config.pad_token_id = 1
    elif 'llama-3' in model_name.lower() or 'llama3' in model_name.lower():
        model_config.model_type = 'llama3'
        model_config.attention_bias = False
        model_config.intermediate_size = config.intermediate_size
        model_config.hidden_act = config.hidden_act
        assert config.max_position_embeddings >= 32768, f"{config.max_position_embeddings}"
        model_config.max_position_embeddings = config.max_position_embeddings
        model_config.tie_word_embeddings = config.tie_word_embeddings
        model_config.pad_token_id = 128001
        model_config.rope_theta = config.rope_theta
        model_config.rope_scaling = config.rope_scaling
    elif 'qwen2' in model_name.lower():
        model_config.model_type = 'qwen2'
        model_config.attention_bias = True
        model_config.intermediate_size = config.intermediate_size
        model_config.hidden_act = config.hidden_act
        assert config.max_position_embeddings >= 32768, f"{config.max_position_embeddings}"
        model_config.max_position_embeddings = config.max_position_embeddings
        model_config.tie_word_embeddings = config.tie_word_embeddings
        model_config.pad_token_id = 151643
        model_config.rope_scaling = None
        model_config.rope_theta = config.rope_theta
    elif hf_model_type == 'qwen3' or 'qwen3' in model_name.lower():
        model_config.model_type = 'qwen3'
        model_config.attention_bias = config.attention_bias
        model_config.intermediate_size = config.intermediate_size
        model_config.hidden_act = config.hidden_act
        assert config.max_position_embeddings >= 32768, f"{config.max_position_embeddings}"
        model_config.max_position_embeddings = config.max_position_embeddings
        model_config.tie_word_embeddings = config.tie_word_embeddings
        model_config.pad_token_id = 151645
        model_config.rope_scaling = None
        model_config.rope_theta = config.rope_theta
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    model_config.hidden_size = config.hidden_size
    model_config.dtype = np.float16
    model_config.num_attention_heads = config.num_attention_heads
    model_config.num_kv_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.num_attention_heads
    model_config.num_kv_groups = model_config.num_attention_heads // model_config.num_kv_heads
    model_config.num_hidden_layers = config.num_hidden_layers
    if hasattr(config, 'head_dim'):
        model_config.head_dim = config.head_dim
    else:
        model_config.head_dim = config.hidden_size // config.num_attention_heads
    model_config.vocab_size = config.vocab_size
    model_config.scaling = model_config.head_dim ** -0.5
    if hasattr(config, 'rms_norm_eps'):
        model_config.rms_norm_eps = config.rms_norm_eps
    return model_config
