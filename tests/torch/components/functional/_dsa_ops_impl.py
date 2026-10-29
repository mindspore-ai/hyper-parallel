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
"""Independent FP32 references for the causal, uncompressed DSA CANN path."""

import pytest
import torch
import torch_npu

from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_kl_loss import dsa_sparse_indexer_kl_loss
from hyper_parallel.components.functional.dsa_dense import dsa_dense_indexer_kl_loss
from hyper_parallel.components.functional.dsa_ops import (
    _dense_indexer_kl_loss_grad,
    _dense_indexer_lse_v2,
    _sparse_attention,
    _lightning_indexer,
    _sparse_indexer_kl_loss_grad,
)
from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention


_SCALE = 576 ** -.5


def _data(layout, dtype=torch.bfloat16):
    if not torch_npu.npu.is_available():
        pytest.skip("DSA precision tests require an Ascend NPU")
    generator = torch.Generator().manual_seed(83)
    prefix = (2, 32) if layout == "BSND" else (45,)
    shapes = {"query": (*prefix, 64, 512), "key": (*prefix, 1, 512),
              "value": (*prefix, 1, 512), "query_rope": (*prefix, 64, 64),
              "key_rope": (*prefix, 1, 64), "query_index": (*prefix, 8, 128),
              "key_index": (*prefix, 1, 128), "weights": (*prefix, 8)}
    data = {name: (torch.randn(shape, generator=generator) * (.01 if name == "weights" else .5)).to(dtype)
            for name, shape in shapes.items()}
    return data


def _documents(data, layout):
    if layout == "BSND":
        return [{name: tensor[batch] for name, tensor in data.items()} for batch in range(2)]
    return [{name: tensor[start:end] for name, tensor in data.items()} for start, end in ((0, 13), (13, 45))]


def _options(layout):
    lengths = torch.tensor([13, 45], dtype=torch.int64) if layout == "TND" else None
    return {"layout": layout, "actual_seq_qlen": lengths, "actual_seq_klen": lengths}


def _indices(data, layout):
    """Select min(causal length, TopK) keys, as required by the gradient kernels."""
    tensors = []
    for document in _documents(data, layout):
        size = document["query"].size(0)
        indices = torch.full((size, 1, 2048), -1, dtype=torch.int32)
        for token in range(size):
            end = document["key"].size(0) - size + token + 1
            selected = torch.arange(max(0, end - 2048), end)
            indices[token, 0, :selected.numel()] = selected
        tensors.append(indices)
    return torch.stack(tensors) if layout == "BSND" else torch.cat(tensors)


def _reference(data, layout, indices=None):
    """Use explicit causal masks and average teacher probabilities over heads."""
    result = {name: [] for name in (
        "output", "maximum", "total", "index_max", "index_sum", "scores", "teacher_target",
    )}
    losses = []
    for batch, document in enumerate(_documents(data, layout)):
        query = document["query"]
        key = document["key"].expand(-1, query.size(1), -1)
        size = query.size(0)
        valid = torch.ones(size, key.size(0), dtype=torch.bool).tril(diagonal=key.size(0) - size)
        if indices is not None:
            selected = indices[batch] if layout == "BSND" else indices[0:13] if batch == 0 else indices[13:]
            valid.zero_()
            for token, row in enumerate(selected[:, 0]):
                valid[token, row[row >= 0].long()] = True
        scores = (torch.einsum("shd,thd->hst", query, key)
                  + torch.einsum("shr,thr->hst", document["query_rope"],
                                 document["key_rope"].expand(-1, query.size(1), -1))) * _SCALE
        scores = scores.masked_fill(~valid, -torch.inf)
        maximum = scores.amax(-1)
        total = (scores - maximum[..., None]).exp().sum(-1)
        probabilities = scores.softmax(-1)
        result["output"].append(torch.einsum("hst,td->shd", probabilities, document["value"][:, 0]))
        result["maximum"].append(maximum.T)
        result["total"].append(total.T)
        index_scores = (torch.einsum("shd,td->sht", document["query_index"], document["key_index"][:, 0])
                        .relu() * document["weights"][..., None]).sum(1).masked_fill(~valid, -torch.inf)
        index_max = index_scores.amax(-1)
        result["index_max"].append(index_max)
        result["index_sum"].append((index_scores - index_max[:, None]).exp().sum(-1))
        result["scores"].append(index_scores)
        teacher = probabilities.detach().mean(0)
        teacher_target = torch.zeros((size, 1, key.size(0)), device=query.device)
        teacher_target[:, 0] = teacher
        result["teacher_target"].append(teacher_target)
        log_index = index_scores.log_softmax(-1).masked_fill(~valid, 0)
        losses.append((teacher * (teacher.clamp_min(1e-30).log() - log_index)).sum())
    for name in ("output", "maximum", "total", "index_max", "index_sum"):
        result[name] = torch.stack(result[name]) if layout == "BSND" else torch.cat(result[name])
    for name in ("maximum", "total", "index_max", "index_sum"):
        result[name] = result[name].unsqueeze(1 if layout == "BSND" else 0)
    result["loss"] = torch.stack(losses).sum()
    max_key_length = max(document["key"].size(0) for document in _documents(data, layout))
    result["teacher_target"] = torch.cat([
        torch.nn.functional.pad(value, (0, max_key_length - value.shape[-1]))
        for value in result["teacher_target"]
    ])
    return result


def _device_data(data, requires_grad=False):
    device = {name: tensor.npu().requires_grad_(requires_grad) for name, tensor in data.items()}
    if data["value"] is data["key"]:
        device["value"] = device["key"]
    return device


def _close(actual, expected, *, rtol=.03, atol=.003):
    torch.testing.assert_close(actual.detach().float().cpu(), expected.detach().float(), rtol=rtol, atol=atol)


def _gradient_close(actual, expected):
    """Bound BF16 accumulation error, including cancellation around zero."""
    actual, expected = actual.detach().float().cpu(), expected.detach().float()
    error = actual - expected
    relative_error = error.norm() / expected.norm().clamp_min(1e-10)
    max_error = error.abs().max()
    max_allowed = .04 * expected.abs().max() + 1e-5
    assert relative_error <= .02, f"Gradient relative L2 error={relative_error}, allowed={.02}"
    assert max_error <= max_allowed, f"Gradient max absolute error={max_error}, allowed={max_allowed}"


def check_indexer(layout: str, dtype: str) -> None:
    """Compare selected indices and scores with independent FP32."""
    data = _data(layout, getattr(torch, dtype))
    device = _device_data(data)
    reference = _reference({name: tensor.float() for name, tensor in data.items()}, layout)
    indices, values = _lightning_indexer(device["query_index"], device["key_index"], device["weights"],
                                           **_options(layout))
    for batch, scores in enumerate(reference["scores"]):
        index = indices.cpu()[batch] if layout == "BSND" else indices.cpu()[:13] if batch == 0 else indices.cpu()[13:]
        value = values.cpu()[batch] if layout == "BSND" else values.cpu()[:13] if batch == 0 else values.cpu()[13:]
        for token in range(scores.size(0)):
            count = token + 1
            torch.testing.assert_close(index[token, 0, :count].sort().values, torch.arange(count, dtype=torch.int32))
            _close(value[token, 0, :count], scores[token, index[token, 0, :count].long()], atol=.008)
            torch.testing.assert_close(index[token, 0, count:], torch.full_like(index[token, 0, count:], -1))


def check_dense_indexer_lse(layout: str, dtype: str) -> None:
    """Compare dense indexer LSE V2 with an independent FP32 reference."""
    data = _data(layout, getattr(torch, dtype))
    reference = _reference({name: tensor.float() for name, tensor in data.items()}, layout)
    index_lse = _dense_indexer_lse_v2(
        data["query_index"].npu(), data["key_index"].npu(), data["weights"].npu(), **_options(layout),
    )
    expected_lse = reference["index_max"] + reference["index_sum"].log()
    _close(index_lse, expected_lse)


def check_topk_selection() -> None:
    """Compare ranking at the selection boundary, with temporary view conversions."""
    if not torch_npu.npu.is_available():
        pytest.skip("DSA precision tests require an Ascend NPU")
    generator = torch.Generator().manual_seed(128)
    query = (torch.randn(1, 2304, 8, 128, generator=generator) * .5).to(torch.bfloat16)
    key = (torch.randn(1, 2304, 1, 128, generator=generator) * .5).to(torch.bfloat16)
    weights = (torch.randn(1, 2304, 8, generator=generator) * .01).to(torch.bfloat16)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        # The views have a storage offset and a non-contiguous last dimension.
        device_query = torch.nn.functional.pad(query.npu().transpose(-1, -2), (1, 1))[..., 1:-1].transpose(-1, -2)
        indices, values = _lightning_indexer(device_query, key.npu(), weights.npu())
    torch.npu.current_stream().wait_stream(stream)
    indices, values = indices.cpu()[0, -8:, 0], values.cpu()[0, -8:, 0]
    scores = (torch.einsum("shd,td->sht", query.float()[0, -8:], key.float()[0, :, 0]).relu()
              * weights.float()[0, -8:, :, None]).sum(1)
    for row in range(8):
        scores[row, 2304 - 8 + row + 1:] = -torch.inf
    selected = scores.gather(1, indices.long())
    _close(values, selected, atol=.003)
    _close(selected.sort(-1, descending=True).values, scores.topk(2048).values, atol=.003)
    differences = indices.sort(-1).values.diff(dim=-1)
    assert bool((differences > 0).all()), f"TopK includes duplicate indices: minimum difference={differences.min()}"


def check_attention(layout: str) -> None:
    """Compare genuinely sparse attention and Q/shared-KV/RoPE gradients."""
    data = _data(layout)
    if layout == "BSND":
        for name in ("query", "query_rope", "query_index", "weights"):
            data[name] = data[name][:, :8].contiguous()
        generator = torch.Generator().manual_seed(71)
        for name in ("key", "key_rope", "key_index"):
            data[name] = (torch.randn(2, 2112, *data[name].shape[-2:], generator=generator) * .5).to(torch.bfloat16)
    data["value"] = data["key"]
    device = _device_data(data, requires_grad=True)
    reference_data = {name: tensor.float().requires_grad_() for name, tensor in data.items()}
    reference_data["value"] = reference_data["key"]
    indices = _indices(data, layout)
    reference = _reference(reference_data, layout, indices)
    output, maximum, total = _sparse_attention(
        device["query"], device["key"], device["value"], indices.npu(), _SCALE,
        query_rope=device["query_rope"], key_rope=device["key_rope"], **_options(layout))
    _close(output, reference["output"])
    _close(maximum, reference["maximum"])
    _close(total, reference["total"])
    upstream = torch.randn(output.shape, generator=torch.Generator().manual_seed(52)).to(output.dtype)
    output.backward(upstream.npu())
    reference["output"].backward(upstream.float())
    for name in ("query", "key", "query_rope", "key_rope"):
        _gradient_close(device[name].grad, reference_data[name].grad)


def check_kl(layout: str, sparse: bool) -> None:
    """Compare dense/sparse summed KL and each unnormalized indexer gradient."""
    data = _data(layout)
    device = _device_data(data)
    reference_data = {name: tensor.float().requires_grad_() for name, tensor in data.items()}
    indices = _indices(data, layout) if sparse else None
    reference = _reference(reference_data, layout, indices)
    arguments = [device[name] for name in ("query", "key", "query_index", "key_index", "weights")]
    stats = [reference[name].detach().contiguous().npu() for name in ("maximum", "total")]
    options = {"query_rope": device["query_rope"], "key_rope": device["key_rope"], **_options(layout)}
    if sparse:
        gradients = _sparse_indexer_kl_loss_grad(
            *arguments, indices.npu(), *stats, _SCALE, **options,
        )
    else:
        if layout != "TND":
            pytest.skip("dense package KL autograd integration is validated through TND CP")
        index_lse = _dense_indexer_lse_v2(
            *arguments[2:], **_options(layout),
        )
        target = reference["teacher_target"].detach().npu()
        gradients = _dense_indexer_kl_loss_grad(
            arguments[2], arguments[3], arguments[4], target, index_lse,
            actual_seq_qlen=_options(layout)["actual_seq_qlen"].npu(),
            actual_seq_klen=_options(layout)["actual_seq_klen"].npu(), layout=layout,
        )
    _close(gradients[-1].squeeze(), reference["loss"], rtol=.03, atol=.003)
    reference["loss"].backward()
    for actual, name in zip(gradients[:3], ("query_index", "key_index", "weights")):
        _gradient_close(actual, reference_data[name].grad)
    if layout == "TND":
        for tensor in device.values():
            tensor.requires_grad_()
        lengths = _options(layout)["actual_seq_qlen"]
        if sparse:
            loss = dsa_sparse_indexer_kl_loss(
                device["query_index"], device["key_index"], device["weights"], device["query"], device["key"],
                indices.npu(), *stats, device["query_rope"], device["key_rope"],
                lengths, lengths, _SCALE, .7,
            )
        else:
            loss = dsa_dense_indexer_kl_loss(
                device["query_index"], device["key_index"], device["weights"], target, index_lse,
                query_lengths=lengths, key_lengths=lengths, layout="TND", loss_coefficient=.7,
            )
        _close(loss, reference["loss"] * (.7 / 45))
        (loss * 2.5).backward()
        for name in ("query_index", "key_index", "weights"):
            _gradient_close(device[name].grad, reference_data[name].grad * (.7 * 2.5 / 45))
        for name in ("query", "key", "query_rope", "key_rope"):
            assert device[name].grad is None, f"Teacher {name} unexpectedly received gradient {device[name].grad}"


def check_training_chain() -> None:
    """Exercise public adapters and loss normalization with a shared MLA K/V."""
    data = _data("BSND")
    data["value"] = data["key"]
    device = _device_data(data, requires_grad=True)
    reference_data = {name: tensor.float().requires_grad_() for name, tensor in data.items()}
    reference_data["value"] = reference_data["key"]
    reference = _reference(reference_data, "BSND")
    lengths = torch.tensor([32, 64], dtype=torch.int32, device="npu")
    indices, iq, ik, weights = dsa_indexer(device["query_index"], device["key_index"], device["weights"],
                                         lengths, lengths, 2048)
    output, maximum, total = dsa_sparse_attention(
        device["query"], device["key"], device["query_rope"], device["key_rope"], indices, _SCALE, lengths, lengths)
    _close(output[..., :512], reference["output"])
    torch.testing.assert_close(output[..., 512:], torch.zeros_like(output[..., 512:]))
    loss = dsa_sparse_indexer_kl_loss(iq, ik, weights, device["query"].flatten(0, 1), device["key"].flatten(0, 1),
                       indices, maximum, total, device["query_rope"].flatten(0, 1),
                       device["key_rope"].flatten(0, 1), lengths, lengths, _SCALE, .7)
    _close(loss, reference["loss"] / 64 * .7)
    loss.backward(retain_graph=True)
    for name in ("query", "key", "query_rope", "key_rope"):
        assert device[name].grad is None, f"Teacher {name} unexpectedly received gradient {device[name].grad}"
    upstream = torch.randn(reference["output"].shape, generator=torch.Generator().manual_seed(91)).to(torch.bfloat16)
    (output[..., :512] * upstream.npu()).sum().backward()
    ((reference["output"] * upstream.float()).sum() + reference["loss"] / 64 * .7).backward()
    for name in ("query", "key", "query_rope", "key_rope", "query_index", "key_index", "weights"):
        _gradient_close(device[name].grad, reference_data[name].grad)
