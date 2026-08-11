import torch


def calculate_masks(
    input_ids,
    reset_ids=None,
    actual_seq_len=None,
    mask_compress=False,
    apply_mome=False,
    swa_layers=0,
    swa_sliding_window=3,
    swa_attention_sink=0,
    param_sink_number=0
):
    """
    Three-in-one mask calculation logic extracted from Sophon.
    Does not depend on global args and dataset collator, can be used independently for transformers adaptation.

    Note: In the output masks, True represents Mask (invisible/blocked), False represents Attend (visible/attention).
    """
    batch_size, seq_length = input_ids.size()
    device = input_ids.device

    attention_mask = None
    swa_mask = None
    mome_mask = None

    # ==========================================
    # 1. Calculate Attention Mask (Causal & EOD Reset)
    # ==========================================
    if mask_compress:
        # Compress mode: fixed 2048x2048 upper triangular matrix
        attention_mask = torch.triu(torch.ones([2048, 2048], dtype=torch.bool, device=device), diagonal=1)
    else:
        # Normal mode: [B, 1, S, S] causal upper triangular matrix
        attention_mask = torch.triu(torch.ones((batch_size, seq_length, seq_length), dtype=torch.bool, device=device), diagonal=1).unsqueeze(1)

        # Block cross-document attention when concatenating multiple documents using reset_ids
        if reset_ids is not None:
            # reset_ids is expected to be a list or tensor containing absolute position indices of EOD tokens in each batch
            for b in range(batch_size):
                # Filter invalid positions if passed a padded tensor
                eod_indices = reset_ids[b] if isinstance(reset_ids, list) else reset_ids[b].tolist()
                for index in eod_indices:
                    # Mask: tokens after current EOD cannot see current EOD and tokens before it
                    if index < seq_length - 1 and index >= 0:
                        attention_mask[b, 0, (index + 1):, :(index + 1)] = True

    # ==========================================
    # 2. Calculate SWA (Sliding Window Attention) Mask
    # ==========================================
    if swa_layers > 0:
        if mask_compress:
            swa_mask = attention_mask
        else:
            if swa_attention_sink == 0 or seq_length <= swa_attention_sink:
                general_swa_mask = (
                    torch.triu(torch.ones(seq_length, seq_length, device=device), 1) +
                    torch.tril(torch.ones(seq_length, seq_length, device=device), -swa_sliding_window - 1)
                ).bool()
            else:
                general_swa_mask = (
                    torch.triu(torch.ones(seq_length, seq_length, device=device), 1) +
                    torch.tril(torch.cat([
                        torch.zeros(seq_length, swa_attention_sink, device=device),
                        torch.ones(seq_length, seq_length - swa_attention_sink, device=device)
                    ], dim=1), -(swa_sliding_window - swa_attention_sink) - 1)
                ).bool()

            general_swa_mask = general_swa_mask.expand_as(attention_mask)
            # Apply SWA restrictions on top of the base attention_mask (which already includes EOD blocking)
            swa_mask = torch.logical_or(attention_mask, general_swa_mask)

    # ==========================================
    # 3. Calculate MOME (Memory-Optimized Masked Expansion) Mask
    # ==========================================
    if apply_mome and actual_seq_len is not None:
        num_tokens = batch_size * seq_length
        mome_list = list(actual_seq_len)  # Assume the input is a list of valid lengths/EOD positions

        for i in range(1, swa_sliding_window - 1):
            mome_list.extend([each + i for each in actual_seq_len])

        mome_list.extend([0, 1])
        mome_list.sort()

        mome_tensor = torch.tensor(mome_list, dtype=torch.long, device=device)
        valid_mask = (mome_tensor < num_tokens)
        mome_tensor = mome_tensor[valid_mask]

        # Initialize all True (all Masked), then scatter False at computed keep positions
        mome_mask_flat = torch.ones(num_tokens, dtype=torch.bool, device=device)
        mome_mask_flat.scatter_(0, mome_tensor, False)
        mome_mask = mome_mask_flat.view(batch_size, seq_length)

    # ==========================================
    # 4. Param Sink dimension concatenation padding (only in non-Compress mode)
    # ==========================================
    if param_sink_number > 0 and not mask_compress:
        add_mask_dim_2 = torch.zeros(
            (batch_size, 1, param_sink_number, attention_mask.size(3)),
            dtype=torch.bool, device=device
        )
        add_mask_dim_3 = torch.triu(torch.ones(
            (batch_size, 1, attention_mask.size(2) + param_sink_number, param_sink_number),
            dtype=torch.bool, device=device
        ), 1)

        attention_mask = torch.cat([add_mask_dim_2, attention_mask], dim=2)
        attention_mask = torch.cat([add_mask_dim_3, attention_mask], dim=3)

        if swa_layers > 0:
            swa_mask = torch.cat([add_mask_dim_2, swa_mask], dim=2)
            swa_mask = torch.cat([add_mask_dim_3, swa_mask], dim=3)

    return attention_mask, swa_mask, mome_mask