import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if model_path not in sys.path:
    sys.path.insert(0, model_path)


from src.model_components import (
    CopyPointerHead,
    EmbeddingLayer,
    EncoderLayer,
    DecoderLayer,
    make_decoder_cache,
)

class TLM_S(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.encoder = EncoderLayer(c)
        self.decoder = DecoderLayer(c)
        self.embedding = EmbeddingLayer(c)
        self.output_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.use_weight_tying:
            self.output_head.weight = self.embedding.token_embedding.weight
        self.copy_head = CopyPointerHead(c.d_model) if c.use_copy else None
        
        print(f"[tlm_s.py] Initialized TLM_S model variant with {sum(p.numel() for p in self.parameters())} parameters")
        
    @staticmethod
    def _causal(seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _causal_incremental(n_new, total_len, device):
        """Causal mask for `n_new` new query rows attending over `total_len` keys
        (past cached keys + the new ones). Past columns are fully visible; the new
        columns are causal among themselves. Reduces to `_causal` when total_len==n_new."""
        mask = torch.ones(n_new, total_len, device=device)
        new_start = total_len - n_new
        mask[:, new_start:] = torch.tril(torch.ones(n_new, n_new, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def encode(self, enc_ids, enc_attn=None, task_ids=None):
        enc_emb = self.embedding(enc_ids)
        if task_ids is None:
            task_ids = torch.zeros(enc_ids.size(0), dtype=torch.long, device=enc_ids.device)
        enc_mask = enc_attn.unsqueeze(1).unsqueeze(2) if enc_attn is not None else None
        enc_out = self.encoder(enc_emb, attention_mask=enc_mask, task_ids=task_ids)
        return enc_out, enc_mask

    def new_cache(self):
        return make_decoder_cache(self.config.n_decoder_layers)

    def decode_step(self, dec_ids_new, enc_out, enc_ids, cache, position_offset,
                     enc_mask=None, use_copy: bool = True):
        """Run the decoder for only the new token(s) `dec_ids_new` (B, Sq_new), reusing
        and updating `cache` (see make_decoder_cache) in place. `position_offset` is the
        number of tokens already generated/cached (0 on the first call). Mathematically
        identical to calling forward() on the full sequence so far and slicing the last
        Sq_new positions."""
        dec_emb = self.embedding(dec_ids_new)
        Sq_new = dec_ids_new.size(1)
        total_len = position_offset + Sq_new
        self_mask = self._causal_incremental(Sq_new, total_len, dec_ids_new.device)
        if use_copy and self.copy_head is not None:
            dec_out, ca_w = self.decoder(dec_emb, enc_out,
                                          self_attention_mask=self_mask,
                                          cross_attention_mask=enc_mask,
                                          return_last_cross_attn=True,
                                          cache=cache, position_offset=position_offset)
            vocab_logits = self.output_head(dec_out)
            log_probs, p_gen = self.copy_head(dec_out, vocab_logits, ca_w, enc_ids,
                                              pad_id=self.config.special_token_ids["pad"],
                                              vocab_size=self.config.vocab_size)
            return {"log_probs": log_probs, "p_gen": p_gen, "vocab_logits": vocab_logits, "ca_w": ca_w}
        else:
            dec_out = self.decoder(dec_emb, enc_out,
                                    self_attention_mask=self_mask,
                                    cross_attention_mask=enc_mask,
                                    cache=cache, position_offset=position_offset)
            logits = self.output_head(dec_out)
            return {"log_probs": F.log_softmax(logits, dim=-1),
                    "p_gen": None,
                    "vocab_logits": logits}

    def decode_full(self, enc_out, enc_mask, enc_ids, dec_ids, dec_attn=None, use_copy: bool = True):
        """Non-incremental (full teacher-forced) decoder pass reusing an already-computed
        `enc_out`/`enc_mask` (see `encode()`). Factored out of `forward()` so callers that
        need the SAME encoder output for more than one decoder pass (e.g.
        calibration_training.calibration_step running a rollout-replay pass and a gold-anchor
        pass for the same encoder input) can avoid re-running the encoder redundantly.
        `forward()` is exactly `encode()` followed by this."""
        dec_emb = self.embedding(dec_ids)
        cmask = self._causal(dec_ids.size(1), dec_ids.device)
        dec_mask = cmask if dec_attn is None else cmask * dec_attn.unsqueeze(1).unsqueeze(2)
        if use_copy and self.copy_head is not None:
            dec_out, ca_w = self.decoder(dec_emb, enc_out,
                                          self_attention_mask=dec_mask,
                                          cross_attention_mask=enc_mask,
                                          return_last_cross_attn=True)
            vocab_logits = self.output_head(dec_out)
            log_probs, p_gen = self.copy_head(dec_out, vocab_logits, ca_w, enc_ids,
                                              pad_id=self.config.special_token_ids["pad"],
                                              vocab_size=self.config.vocab_size)
            return {"log_probs": log_probs, "p_gen": p_gen, "vocab_logits": vocab_logits,
                    "ca_w": ca_w, "dec_out": dec_out}
        else:
            dec_out = self.decoder(dec_emb, enc_out,
                                    self_attention_mask=dec_mask,
                                    cross_attention_mask=enc_mask)
            logits = self.output_head(dec_out)
            return {"log_probs": F.log_softmax(logits, dim=-1),
                    "p_gen": None,
                    "vocab_logits": logits, "dec_out": dec_out}

    def forward(self, enc_ids, dec_ids, enc_attn=None, dec_attn=None,
                task_ids=None, use_copy: bool = True):
        enc_out, enc_mask = self.encode(enc_ids, enc_attn=enc_attn, task_ids=task_ids)
        return self.decode_full(enc_out, enc_mask, enc_ids, dec_ids, dec_attn=dec_attn, use_copy=use_copy)
            
            
print("[tlm_s.py] Loaded TLM_S model variant")
