# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Small exhaustive DSA oracles: causal selection, sparse attention and student KL."""

from __future__ import annotations

from dataclasses import replace

import torch

from hyper_parallel.components import ops as operators
from tests.common.op_precision import PrecisionCase
from tests.common.op_precision_inputs import Inputs


def _compressed_bounds(layout, query, token):
    if layout.segments is None:
        return 0, (layout.query_offset + token + 1) // layout.compress_ratio
    for start, end, key_start, key_end, trailing in layout.segments:
        if start <= token < end:
            position = layout.compress_ratio * key_end + trailing - (end - token)
            return key_start, (position + 1) // layout.compress_ratio
    raise ValueError(f"No document for token={token}, query_shape={query.shape}")


def _compressed_indexer(x, key, weights, *, layout, sparse_count, query_chunk_size=256, reduce_sum=None):
    """Exhaustively rank visible compressed keys within each document."""
    del query_chunk_size
    result = torch.full((x.shape[0], x.shape[1], min(sparse_count, key.shape[1])), -1,
                        dtype=torch.int32, device=x.device)
    for batch in range(x.shape[0]):
        for token in range(x.shape[1]):
            low, high = _compressed_bounds(layout, x, token)
            candidates = list(range(low, min(high, key.shape[1])))
            if not candidates:
                continue
            scores = torch.stack([
                ((x[batch, token].float() * key[batch, index].float()).sum(-1).relu()
                 * weights[batch, token].float()).sum() for index in candidates
            ])
            if reduce_sum is not None:
                scores = reduce_sum(scores)
            selected = scores.argsort(descending=True)[:result.shape[-1]].tolist()
            indices = sorted(candidates[index] for index in selected)
            result[batch, token, :len(indices)] = torch.tensor(indices, dtype=torch.int32, device=x.device)
    return result


def _packed_indexer(x, key, weights, query_ends, key_ends, sparse_count):
    """Select document-local keys and canonicalize unordered TopK results."""
    query, keys, merge = (tensor.flatten(0, 1) for tensor in (x, key, weights))
    result = torch.full((query.shape[0], 1, sparse_count), -1, dtype=torch.int32, device=x.device)
    qs = ks = 0
    with torch.no_grad():
        for qe, ke in zip(query_ends.tolist(), key_ends.tolist()):
            for token in range(qs, qe):
                visible = token - qs + ke - ks - (qe - qs) + 1
                scores = torch.stack([
                    ((query[token].float() * keys[ks + index, 0].float()).sum(-1).relu()
                     * merge[token].float()).sum() for index in range(visible)
                ])
                selected = scores.argsort(descending=True)[:sparse_count]
                result[token, 0, :selected.numel()] = selected.to(torch.int32)
            qs, ks = qe, ke
    # The contract leaves TopK order unspecified; compare the selected sets.
    return result.sort(-1).values, query, keys, merge


def _canonical_packed_indexer(*args):
    indices, query, key, weights = operators.dsa_indexer_packed(*args)
    return indices.sort(-1).values, query, key, weights


def _compressed_attention(x, raw, compressed, indices, sinks, *, scale, window, layout,
                          global_sequence_length, raw_key_offset=0, main_key_offset=0, global_key_length=None):
    """Reconstruct visible raw and compressed banks using an explicit mask."""
    del global_sequence_length, global_key_length
    bank = raw[:, 0] if compressed is None else torch.cat((raw[:, 0], compressed), dim=1)
    allowed = torch.zeros((x.shape[2], bank.shape[1]), dtype=torch.bool, device=x.device)
    for token in range(x.shape[2]):
        position = layout.query_offset + token
        origin = 0
        for start, end, key_start, key_end, trailing in layout.segments or ():
            if start <= token < end:
                origin = layout.query_offset + end - (key_end - key_start) * layout.compress_ratio - trailing
        for index in range(max(origin, position - window + 1, raw_key_offset), position + 1):
            allowed[token, index - raw_key_offset] = True
        if compressed is not None:
            for index in set(indices[0, token].tolist()) - {-1}:
                allowed[token, raw.shape[2] + index - main_key_offset] = True
    # Batched contractions avoid rounding shared-bank gradients once per token.
    logits = torch.einsum("bhqd,bkd->bhqk", x, bank) * scale
    logits = logits.masked_fill(~allowed, -torch.inf)
    probabilities = torch.cat((logits.float(), sinks[None, :, None, None].expand(
        x.shape[0], -1, x.shape[2], -1,
    )), dim=-1).softmax(-1)
    return torch.einsum("bhqk,bkd->bqhd", probabilities[..., :-1].to(x.dtype), bank)


def _absorbed(x, key, q_rope, k_rope, indices, scale, query_ends, key_ends,
              *, sink_key=None, sink_value=None, dropout=0.0):
    """Evaluate packed sparse attention and optional parameter sinks token by token."""
    if dropout:
        raise ValueError("Strict sparse attention fixtures require dropout=0")
    # Cast shared banks once so token-wise oracle gradients accumulate in FP32.
    query, keys, q_rotary, k_rotary = (tensor.flatten(0, 1).float() for tensor in (x, key, q_rope, k_rope))
    sink_key = None if sink_key is None else sink_key.float()
    sink_value = None if sink_value is None else sink_value.float()
    outputs, maxima, totals = [], [], []
    qs = ks = 0
    for qe, ke in zip(query_ends.tolist(), key_ends.tolist()):
        for token in range(qs, qe):
            visible = token - qs + ke - ks - (qe - qs)
            selected = [index for index in indices[token, 0].tolist() if 0 <= index <= visible]
            values = keys[torch.tensor(selected, device=x.device) + ks, 0].float()
            rotary = k_rotary[torch.tensor(selected, device=x.device) + ks, 0].float()
            logits = (query[token].float() @ values.T + q_rotary[token].float() @ rotary.T) * scale
            maximum = logits.amax(-1, keepdim=True)
            maxima.append(maximum.detach())
            totals.append((logits - maximum).exp().sum(-1, keepdim=True).detach())
            padded = torch.nn.functional.pad(values, (0, q_rope.shape[-1]))
            if sink_key is not None:
                full_query = torch.cat((query[token], q_rotary[token]), dim=-1).float()
                sink_logits = torch.einsum("hd,shd->hs", full_query, sink_key[0].float()) * scale
                logits = torch.cat((logits, sink_logits), dim=-1)
                padded = torch.cat((padded[:, None].expand(-1, query.shape[1], -1), sink_value[0].float()), dim=0)
                output = torch.einsum("hs,shd->hd", logits.softmax(-1), padded)
            else:
                output = logits.softmax(-1) @ padded
            outputs.append(output.to(x.dtype))
        qs, ks = qe, ke
    return (torch.stack(outputs).reshape(*x.shape[:3], -1),
            torch.stack(maxima).squeeze(-1).unsqueeze(0), torch.stack(totals).squeeze(-1).unsqueeze(0))


def _kl_term(student, target):
    return (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - student.log_softmax(-1))).sum()


def _compressed_kl(x, key, weights, teacher_query, teacher_key, indices, sinks, *, attention_scale, loss_coeff,
                   query_chunk_size=256, reduce_sum=None, layout=None):
    """Compute compressed-teacher KL with detached teacher probabilities."""
    del query_chunk_size, reduce_sum
    loss = (x.float().sum() + key.float().sum() + weights.float().sum()) * 0.0
    for batch in range(x.shape[0]):
        for token in range(x.shape[1]):
            low, high = (0, key.shape[1]) if layout is None else _compressed_bounds(layout, x, token)
            selected = [index for index in indices[batch, token].tolist() if low <= index < high]
            if not selected:
                continue
            student_dots = (x[batch, token].float() @ key[batch, selected].float().T).relu()
            student = (student_dots * weights[batch, token].float()[:, None]).sum(0)
            with torch.no_grad():
                logits = teacher_query[batch, :, token].float() @ teacher_key[batch, selected].float().T
                logits = torch.cat((logits * attention_scale, sinks.float()[:, None]), dim=-1)
                target = logits.softmax(-1)[:, :-1].sum(0)
                target = target / target.sum()
            loss = loss + _kl_term(student, target)
    return loss * (loss_coeff / (x.shape[0] * x.shape[1]))


def _packed_kl(x, index_key, weights, query, key, indices, maximum, total, q_rope, k_rope,
               query_ends, key_ends, scale, loss_coeff):
    del maximum, total
    return _packed_kl_values(x, index_key, weights, query, key, q_rope, k_rope,
                             query_ends, key_ends, scale, loss_coeff, indices)


def _dense_kl(x, index_key, weights, query, key, maximum, total, q_rope, k_rope,
              query_ends, key_ends, *, scale, loss_coeff):
    del maximum, total
    return _packed_kl_values(x, index_key, weights, query, key, q_rope, k_rope,
                             query_ends, key_ends, scale, loss_coeff, None)


def _packed_kl_values(x, index_key, weights, query, key, q_rope, k_rope,
                      query_ends, key_ends, scale, loss_coeff, indices):
    """Reconstruct packed dense or sparse teacher distributions and student KL."""
    loss = (x.float().sum() + index_key.float().sum() + weights.float().sum()) * 0.0
    qs = ks = 0
    for qe, ke in zip(query_ends.tolist(), key_ends.tolist()):
        for token in range(qs, qe):
            visible = token - qs + ke - ks - (qe - qs)
            selected = list(range(visible + 1)) if indices is None else [
                index for index in indices[token, 0].tolist() if 0 <= index <= visible
            ]
            rows = torch.tensor(selected, device=x.device) + ks
            with torch.no_grad():
                teacher = (query[token].float() @ key[rows, 0].float().T
                           + q_rope[token].float() @ k_rope[rows, 0].float().T) * scale
                target = teacher.softmax(-1).mean(0)
            scores = (x[token].float() @ index_key[rows, 0].float().T).relu()
            student = (scores * weights[token].float()[:, None]).sum(0)
            loss = loss + _kl_term(student, target)
        qs, ks = qe, ke
    return loss * (loss_coeff / x.shape[0])


def dsa_cases(dtype: torch.dtype, hardware: bool = False) -> list[PrecisionCase]:
    """Cover every DSA variant with padding, packed resets and trainable students/teachers."""
    inputs = Inputs(dtype)
    cases = []
    width, heads, length = (128 if hardware else 8), (32 if hardware else 2), 6
    layout = operators.DsaQueryLayout(compress_ratio=2, segments=((0, 3, 0, 1, 1), (3, 6, 1, 2, 1)))
    student = (inputs.tensor(1, length, heads, width) + 0.5).requires_grad_()
    index_key = (inputs.tensor(1, 2, width) + 0.5).requires_grad_()
    weights = inputs.tensor(1, length, heads, dtype=torch.float32)
    indices = torch.tensor([[[-1, -1], [0, -1], [0, -1], [-1, -1], [1, -1], [1, -1]]], dtype=torch.int32)
    cases.append(PrecisionCase("dsa_indexer", "compressed_causal_topk", "two_documents", operators.dsa_indexer,
                               _compressed_indexer, (student, index_key, weights),
                               {"layout": layout, "sparse_count": 2, "query_chunk_size": 2}, dtype))
    selection_key = torch.cat((torch.full((1, 3, width), -1.0),
                               torch.ones(1, 1, width), torch.full((1, 1, width), 2.0)), dim=1).to(dtype)
    selection_weights = weights[:, :1].abs() + 0.25
    cases.append(PrecisionCase("dsa_indexer", "compressed_causal_topk", "competitive_topk_with_prefix",
                               operators.dsa_indexer, _compressed_indexer,
                               (student[:, :1], selection_key, selection_weights),
                               {"layout": operators.DsaQueryLayout(
                                   compress_ratio=1, query_offset=4, segments=((0, 1, 0, 5, 0),)),
                                "sparse_count": 2}, dtype))
    attention_width = 512 if hardware else 8
    teacher_query = inputs.tensor(1, heads, length, attention_width)
    teacher_key = inputs.tensor(1, 2, attention_width)
    sinks = inputs.tensor(heads, dtype=torch.float32)
    # A2/A3 sparse MLA requires a physical TopK width of 512 or 1024,
    # even when a short document has only one valid compressed candidate.
    attention_indices = torch.nn.functional.pad(indices, (0, 510), value=-1) if hardware else indices
    attention_key = torch.cat((teacher_key, inputs.tensor(1, 510, attention_width)), dim=1) if hardware else teacher_key
    cases.append(PrecisionCase("deepseek_sparse_attention", "compressed_banks", "window_and_compressed",
                               operators.deepseek_sparse_attention_compressed_banks, _compressed_attention,
                               (teacher_query, inputs.tensor(1, 1, length, attention_width),
                                attention_key, attention_indices, sinks),
                               {"layout": layout, "scale": attention_width ** -0.5, "window": 128 if hardware else 2,
                                "global_sequence_length": length, "global_key_length": attention_key.shape[1]}, dtype))
    # A longer single document supplies multiple candidates, so the KL and its
    # three student derivatives are nonzero rather than a one-choice softmax.
    kl_layout = operators.DsaQueryLayout(compress_ratio=2, segments=((0, 6, 0, 3, 0),))
    kl_indices = torch.tensor([[[-1, -1, -1], [0, -1, -1], [0, -1, -1],
                                [0, 1, -1], [0, 1, -1], [0, 1, 2]]], dtype=torch.int32)
    cases.append(PrecisionCase("dsa_indexer_kl", "compressed_teacher", "nonzero_student_loss", operators.dsa_indexer_kl,
                               _compressed_kl,
                               (student, inputs.tensor(1, 3, width), weights, teacher_query,
                                inputs.tensor(1, 3, attention_width), kl_indices, sinks),
                               {"layout": kl_layout, "attention_scale": attention_width ** -0.5,
                                "loss_coeff": 0.7, "query_chunk_size": 2}, dtype))
    cases.extend(_packed_dsa_cases(inputs, hardware, student, selection_weights))
    return cases


def _absorbed_cases(
    inputs: Inputs, teacher: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    topk: torch.Tensor, scale: float, ends: torch.Tensor,
) -> list[PrecisionCase]:
    """Build sparse attention cases with and without trainable parameter sinks."""
    query, key, q_rope, k_rope = teacher
    heads = query.shape[2]
    dtype = inputs.dtype
    cases = []
    for with_sink in (False, True):
        operation = (operators.deepseek_sparse_attention_with_sink if with_sink
                     else operators.deepseek_sparse_attention_absorbed)
        variant = "absorbed_with_parameter_sink" if with_sink else "absorbed_with_stats"
        kwargs = {} if not with_sink else {
            "sink_key": inputs.tensor(1, 2, heads, query.shape[-1] + q_rope.shape[-1]),
            "sink_value": inputs.tensor(1, 2, heads, query.shape[-1] + q_rope.shape[-1]),
        }
        cases.append(PrecisionCase("deepseek_sparse_attention", variant, "packed_two_documents", operation, _absorbed,
                                   (query, key, q_rope, k_rope, topk, scale, ends, ends), kwargs, dtype))
    return cases


def _packed_dsa_cases(
    inputs: Inputs, hardware: bool, student: torch.Tensor, selection_weights: torch.Tensor,
) -> list[PrecisionCase]:
    """Build packed indexer, sparse attention and teacher-KL boundary cases."""
    dtype = inputs.dtype
    length, heads, width = student.shape[1:]
    cases = []
    ends = torch.tensor([3, 6], dtype=torch.int32)
    packed_key = (inputs.tensor(1, length, 1, width, scale=0.05)
                  + torch.linspace(0.25, 1.0, length)[None, :, None, None]).to(dtype)
    merge = (inputs.tensor(1, length, heads, dtype=torch.float32, scale=0.05) + 0.25).requires_grad_()
    cases.append(PrecisionCase("dsa_indexer", "packed_causal_topk", "local_indices", _canonical_packed_indexer,
                               _packed_indexer, (student, packed_key, merge, ends, ends, 512 if hardware else 4),
                               precision_dtype=dtype))
    top_count = 512 if hardware else 2
    key_bank = torch.cat((torch.full((1, 2, 1, width), -1.0), torch.ones(1, top_count, 1, width)), dim=1)
    cases.append(PrecisionCase("dsa_indexer", "packed_causal_topk", "competitive_right_aligned_topk",
                               _canonical_packed_indexer, _packed_indexer,
                               (student[:, :1], key_bank.to(dtype), selection_weights,
                                torch.tensor([1], dtype=torch.int32),
                                torch.tensor([top_count + 2], dtype=torch.int32), top_count), precision_dtype=dtype))
    content_width, rope_width = (512, 64) if hardware else (8, 4)
    query = inputs.tensor(1, length, heads, content_width)
    key = inputs.tensor(1, length, 1, content_width)
    q_rope = inputs.tensor(1, length, heads, rope_width)
    k_rope = inputs.tensor(1, length, 1, rope_width)
    topk = torch.tensor([[[0, -1, -1]], [[0, 1, -1]], [[0, 1, 2]],
                         [[0, -1, -1]], [[0, 1, -1]], [[0, 1, 2]]], dtype=torch.int32)
    if hardware:
        topk = torch.nn.functional.pad(topk, (0, 512 - topk.shape[-1]), value=-1)
    scale = (content_width + rope_width) ** -0.5
    cases.extend(_absorbed_cases(inputs, (query, key, q_rope, k_rope), topk, scale, ends))
    flat_student, flat_key, flat_weights = (tensor.flatten(0, 1) for tensor in (student, packed_key, merge))
    flat_query, flat_teacher_key, flat_q_rope, flat_k_rope = (
        tensor.flatten(0, 1) for tensor in (query, key, q_rope, k_rope)
    )
    # Stats are real detached teacher outputs; fused KL consumes them whereas
    # the exhaustive oracle reconstructs the distributions independently.
    _, maximum, total = _absorbed(query, key, q_rope, k_rope, topk, scale, ends, ends)
    cases.append(PrecisionCase("dsa_indexer_kl", "packed_sparse_teacher", "detached_sparse_teacher",
                               operators.dsa_indexer_kl_packed, _packed_kl,
                               (flat_student, flat_key, flat_weights, flat_query, flat_teacher_key,
                                torch.nn.functional.pad(topk, (0, 512), value=-1) if hardware else topk,
                                maximum, total, flat_q_rope, flat_k_rope, ends, ends, scale, 0.7),
                               precision_dtype=dtype))
    if hardware:
        _, maximum, total = _absorbed(query[..., :128], key[..., :128], q_rope, k_rope, topk, scale, ends, ends)
    cases.append(PrecisionCase("dsa_indexer_kl", "packed_dense_teacher", "detached_dense_teacher",
                               operators.dsa_indexer_kl_dense, _dense_kl,
                               (flat_student, flat_key, flat_weights, flat_query[..., :128] if hardware else flat_query,
                                flat_teacher_key[..., :128].expand(-1, heads, -1) if hardware else flat_teacher_key,
                                maximum[0].T[..., None], total[0].T[..., None], flat_q_rope,
                                flat_k_rope.expand(-1, heads, -1) if hardware else flat_k_rope, ends, ends),
                               {"scale": scale, "loss_coeff": 0.7}, dtype))
    for case in cases[-2:]:
        # Keep nontrivial score differences comparable across CPU/ST dimensions.
        moderate_student = (inputs.tensor(*flat_student.shape, scale=0.02)
                            + 16.0 / (width * heads)).to(dtype)
        cases.append(replace(case, name="moderate_student", args=(
            moderate_student, *case.args[1:],
        )))
    return cases
