import torch


def collate_fn(batch, pad_id: int):
    """Pad encoder / decoder sequences to the batch max; build attention masks."""
    enc_lists, dec_lists = zip(*batch)
    max_se = max(len(x) for x in enc_lists)
    max_sd = max(len(x) for x in dec_lists)

    B = len(batch)
    enc_ids  = torch.full((B, max_se), pad_id, dtype=torch.long)
    dec_ids  = torch.full((B, max_sd), pad_id, dtype=torch.long)
    enc_attn = torch.zeros((B, max_se), dtype=torch.long)

    for i, (e, d) in enumerate(zip(enc_lists, dec_lists)):
        enc_ids[i, :len(e)]  = torch.tensor(e, dtype=torch.long)
        dec_ids[i, :len(d)]  = torch.tensor(d, dtype=torch.long)
        enc_attn[i, :len(e)] = 1

    task_ids = torch.zeros(B, dtype=torch.long)  # 0 = pretrain
    return {"enc_ids": enc_ids, "enc_attn": enc_attn,
            "dec_ids": dec_ids, "task_ids": task_ids}
