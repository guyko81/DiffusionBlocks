from dataclasses import dataclass, field


@dataclass
class LMConfig:
    # Vocabulary / sequence
    vocab_size: int = 50257
    max_seq_len: int = 512

    # Transformer core
    hidden_size: int = 768
    num_layers: int = 8
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0

    # MLA (Multi-head Latent Attention)
    num_heads: int = 12
    kv_lora_rank: int = 128
    q_lora_rank: int = 192
    qk_rope_head_dim: int = 32
    qk_nope_head_dim: int = 32
    v_head_dim: int = 64
    rope_theta: float = 10000.0

    # FFN (dense SwiGLU)
    intermediate_size: int = 2048

    # DiffusionBlocks
    num_blocks: int = 4
    gamma: float = 0.05
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    # Engram
    engram_enabled: bool = True
    engram_table_size: int = 1_000_000
    engram_num_hashes: int = 8
    engram_ngram_orders: tuple = (2, 3)
    engram_inject_iterations: tuple = (0, 2)
    engram_on_cpu: bool = True

    # Training
    tie_embeddings: bool = True

    @property
    def head_dim(self):
        return self.qk_rope_head_dim + self.qk_nope_head_dim
