SEED = 1993
MAX_SEQUENCE_LENGTH = 2048
TRAINING_BLOCK_SIZE = 1024
MONO_VOCAB_SIZE = 32_000
MULTI_VOCAB_SIZE = 150_000
MULTI_TOKENIZER_LINES_PER_LANGUAGE = 1_000_000

EPOCHS = 3
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-5
WARMUP_STEPS = 2
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPSILON = 1e-8
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
CHECKPOINT_STEPS = 1_000


def model_config(vocab_size: int) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 576,
        "initializer_range": 0.02,
        "intermediate_size": 1536,
        "max_position_embeddings": MAX_SEQUENCE_LENGTH,
        "model_type": "llama",
        "num_attention_heads": 9,
        "num_hidden_layers": 30,
        "num_key_value_heads": 3,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": vocab_size,
    }
