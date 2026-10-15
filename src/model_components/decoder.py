import torch.nn as nn
from .feed_forward import FeedForwardLayer
from .multi_head_attention import MultiHeadAttention
from .rope import RotaryEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = MultiHeadAttention(c)
        self.cross_attn = MultiHeadAttention(c)
        self.ffn = FeedForwardLayer(c)
        self.norm1 = nn.LayerNorm(c.d_model)
        self.norm2 = nn.LayerNorm(c.d_model)
        self.norm3 = nn.LayerNorm(c.d_model)
    def forward(self, x, enc_output, rope, self_attention_mask=None,
                cross_attention_mask=None, return_cross_attn=False,
                self_past_kv=None, cross_past_kv=None, use_cache=False, position_offset=0):
        h = self.norm1(x)
        if use_cache:
            attn_out, self_present = self.self_attn(
                h, h, h, attention_mask=self_attention_mask, rope=rope,
                past_kv=self_past_kv, use_cache=True, position_offset=position_offset)
        else:
            attn_out = self.self_attn(h, h, h, attention_mask=self_attention_mask, rope=rope)
            self_present = None
        x = x + attn_out

        norm2 = self.norm2(x)
        if use_cache:
            # enc_output is only re-projected on the first decode step (cross_past_kv
            # None); every later step reuses the cached cross K/V unchanged.
            k_in = enc_output if cross_past_kv is None else None
            v_in = enc_output if cross_past_kv is None else None
            if return_cross_attn:
                ca_out, ca_w, cross_present = self.cross_attn(
                    norm2, k_in, v_in, attention_mask=cross_attention_mask, rope=None,
                    return_weights=True, past_kv=cross_past_kv, use_cache=True)
            else:
                ca_out, cross_present = self.cross_attn(
                    norm2, k_in, v_in, attention_mask=cross_attention_mask, rope=None,
                    past_kv=cross_past_kv, use_cache=True)
                ca_w = None
            x = x + ca_out
            x = x + self.ffn(self.norm3(x))
            return x, ca_w, self_present, cross_present

        if return_cross_attn:
            ca_out, ca_w = self.cross_attn(norm2, enc_output, enc_output,
                                            attention_mask=cross_attention_mask, rope=None,
                                            return_weights=True)
            x = x + ca_out
            x = x + self.ffn(self.norm3(x))
            return x, ca_w
        x = x + self.cross_attn(norm2, enc_output, enc_output,
                                 attention_mask=cross_attention_mask, rope=None)
        x = x + self.ffn(self.norm3(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(c) for _ in range(c.n_decoder_layers)])
        self.rope = RotaryEmbedding(c.d_model // c.n_heads, c.max_seq_len)
        self.final_norm = nn.LayerNorm(c.d_model)
    def forward(self, x, enc_output, self_attention_mask=None,
                cross_attention_mask=None, return_last_cross_attn=False,
                cache=None, position_offset=0):
        """
        cache: optional list of dicts (one per block, see make_decoder_cache), each with
        keys 'self_kv'/'cross_kv'. When given, mutated in place with the updated K/V so
        the same list can be passed back in on the next incremental decode step.
        """
        use_cache = cache is not None
        ca_w = None
        last = len(self.blocks) - 1
        for i, b in enumerate(self.blocks):
            want_ca = return_last_cross_attn and i == last
            if use_cache:
                blk_cache = cache[i]
                x, ca_w_i, self_present, cross_present = b(
                    x, enc_output, self.rope,
                    self_attention_mask=self_attention_mask,
                    cross_attention_mask=cross_attention_mask,
                    return_cross_attn=want_ca,
                    self_past_kv=blk_cache["self_kv"],
                    cross_past_kv=blk_cache["cross_kv"],
                    use_cache=True, position_offset=position_offset)
                blk_cache["self_kv"] = self_present
                blk_cache["cross_kv"] = cross_present
                if want_ca:
                    ca_w = ca_w_i
            elif want_ca:
                x, ca_w = b(x, enc_output, self.rope,
                            self_attention_mask=self_attention_mask,
                            cross_attention_mask=cross_attention_mask,
                            return_cross_attn=True)
            else:
                x = b(x, enc_output, self.rope,
                      self_attention_mask=self_attention_mask,
                      cross_attention_mask=cross_attention_mask)
        x = self.final_norm(x)
        if return_last_cross_attn: return x, ca_w
        return x


def make_decoder_cache(n_decoder_layers):
    """Fresh empty incremental-decode cache: one {'self_kv': None, 'cross_kv': None} per block."""
    return [{"self_kv": None, "cross_kv": None} for _ in range(n_decoder_layers)]