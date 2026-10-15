from dataclasses import dataclass, field
from typing import Dict, List
from torch import nn

@dataclass
class ModelConfig: 

    d_model: int = 192
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_heads: int = 4  # * (48 Dim/Head) = 192 D_model
    d_ff: int = 768
    dropout: float = 0.1
    max_seq_len: int = 1024
    vocab_size: int = 10000
    n_task_types: int = 6          # 0=pretrain: Span denoising, 1= rename, 2= simplify, 3= deduplicate, 4= remove_dead_code, 5= extract_method
    use_copy: bool = True
    special_token_ids: Dict[str, int] = field(default_factory=lambda: {
        **{
            "pad": 0, "unk": 1, "bos": 2, "eos": 3, "mask": 4,
            "sep": 5, "indent": 6, "dedent": 7, "newline": 8,
            "rename": 9, "simplify": 10, "deduplicate": 11,
            "remove_dead_code": 12, "extract_method": 13,
            "search": 14, "replace": 15,
        },
        **{f"extra_id_{i}": 16 + i for i in range(64)},
    })
    tier_name: str = "TLM_S"
    target_param_count: int = 5_000_000
    use_weight_tying: bool = True
    use_prenorm: bool = True
    tokenizer_path: str = "tokenizer/tokenizer_tlm_small.json"
    
    def special_tokens_in_id_order(self) -> List[str]:
        return [tok for tok, _ in sorted(self.special_token_ids.items(), key=lambda kv: kv[1])]

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        if self.d_ff != 4 * self.d_model:
            print(f"[Config] d_ff={self.d_ff} != 4xd_model={4*self.d_model} (intentional for tiny model)")

    def estimated_params(self) -> int:
        embed      = self.vocab_size * self.d_model
        enc_attn   = self.n_encoder_layers * 4 * self.d_model**2          # q,k,v,out -- no bias
        dec_attn   = self.n_decoder_layers * 8 * self.d_model**2          # self + cross
        ffn_block  = self.d_model*self.d_ff + self.d_ff + self.d_ff*self.d_model + self.d_model  # fc1+bias, fc2+bias
        enc_ffn    = self.n_encoder_layers * ffn_block
        dec_ffn    = self.n_decoder_layers * ffn_block
        ln         = 2 * self.d_model                                      # weight + bias per LN
        enc_ln     = self.n_encoder_layers * 2 * ln + ln                   # 2/block + final_norm
        dec_ln     = self.n_decoder_layers * 3 * ln + ln                   # 3/block + final_norm
        task_emb   = self.n_encoder_layers * self.n_task_types * self.d_model
        copy_head  = (self.d_model + 1) if self.use_copy else 0            # gate Linear(d_model,1)
        head       = 0 if self.use_weight_tying else self.vocab_size * self.d_model

        total = embed + enc_attn + dec_attn + enc_ffn + dec_ffn + enc_ln + dec_ln + task_emb + copy_head + head
        print(f"[Config] Estimated params: embed={embed}, enc_attn={enc_attn}, dec_attn={dec_attn}, "
              f"enc_ffn={enc_ffn}, dec_ffn={dec_ffn}, enc_ln={enc_ln}, dec_ln={dec_ln}, "
              f"task_emb={task_emb}, copy_head={copy_head}, head={head}")
        return total

    def __str__(self):
        est = self.estimated_params() / 1e6
        return (
            f"ModelConfig [{self.tier_name}]\n"
            f"  d_model={self.d_model}, n_heads={self.n_heads}\n"
            f"  Encoder: {self.n_encoder_layers}L  Decoder: {self.n_decoder_layers}L  FFN: {self.d_ff}\n"
            f"  Estimated: {est:.2f}M params  (target: {self.target_param_count/1e6:.2f}M)\n"
            f"  Vocab: {self.vocab_size}  MaxSeq: {self.max_seq_len}  TaskTypes: {self.n_task_types}  WeightTie: {self.use_weight_tying}"
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
