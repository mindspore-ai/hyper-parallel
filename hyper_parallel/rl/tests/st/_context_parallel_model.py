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
"""Compare Actor logits, parameter gradients and real updates across CP degrees."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
import yaml

from _context_parallel_gradients import ParameterGradientComparison
from rl.data.contracts import ExperienceBatch
from rl.trainer.build import build_trainer


def create_checkpoint(path, family):
    """Write one logical CPU checkpoint reused unchanged by all topology runs."""
    if (path / 'config.json').exists():
        return
    torch.manual_seed(1207)
    options = {'vocab_size': 512, 'hidden_size': 256, 'intermediate_size': 512, 'num_hidden_layers': 2,
               'num_attention_heads': 8, 'num_key_value_heads': 4, 'head_dim': 128,
               'max_position_embeddings': 16384, 'attention_dropout': 0.0, 'tie_word_embeddings': False}
    if family == 'qwen3_moe':
        options.update(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=256)
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**options)).bfloat16()
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**options)).bfloat16()
    model.save_pretrained(path)
    if family == 'qwen3_moe':
        # Keep the logical expert-count field used by supported published checkpoints.
        saved = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        saved['num_experts'] = options['num_experts']
        (path / 'config.json').write_text(json.dumps(saved, indent=2) + '\n')


def recipe(args):
    """Keep the production training setup while selecting a small shared checkpoint."""
    base = Path('hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml')
    config = yaml.safe_load(base.read_text(encoding='utf-8'))
    if args.family == 'qwen3':
        fixture_path = base.with_name('qwen3_4b_rl.yaml')
        fixture = yaml.safe_load(fixture_path.read_text(encoding='utf-8'))
        config['train']['plan_overrides'] = fixture['train']['plan_overrides']
    config['model'].update(name=args.family, registry_name='cp_model_test', weights_path=str(args.checkpoint),
                           tokenizer_path='/models/Qwen3-MoE-4L')
    train = config['train']
    prompt_batch = math.ceil(args.batch_size / config['rollout']['num_return_sequences'])
    train.update(prompt_batch_size=prompt_batch, micro_batch_size=1, response_mini_batch_size=args.batch_size)
    train['accelerator'].update(tp=args.tp, cp=args.cp, ep=args.ep, dp_shard=args.dp,
                                cpu_offload=True, sequence_parallel=args.sp)
    if args.no_comm_fusion:
        train['accelerator']['comm_fusion'] = False
    train['optimizer']['lr'] = .001
    train['checkpoint']['output_dir'] = str(args.output / 'checkpoints')
    vllm = config['rollout']['vllm']
    world = args.tp * args.cp * args.dp
    vllm.update(tensor_parallel_size=args.tp, data_parallel_size=world // args.tp,
                expert_parallel_size=args.ep, context_parallel_size=1)
    config['logging']['experiment_name'] = 'cp_model_test'
    if args.reference_role:
        config['algorithm']['kl_coef'] = .01
    return config


def local_parameter(parameter):
    """Observe owned FSDP storage without invoking an unregistered DTensor equal op."""
    return parameter.to_local() if hasattr(parameter, 'to_local') else parameter


def check_logits(actor, ids, mask, args, parallel_dims):
    """Compare every valid vocabulary logit in its logical CP interval."""
    captured = []

    def capture(unused_module, unused_inputs, output):
        del unused_module, unused_inputs
        captured.append(local_parameter(output).detach().cpu())

    handle = actor.actor_model.lm_head.register_forward_hook(capture)
    try:
        with torch.no_grad():
            actor.sequence_log_probs(ids, mask)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise AssertionError('Model logit probe must observe exactly one LM head invocation')
    actual = captured[0]
    mesh = getattr(actor.actor_model, 'unified_cp_mesh', None)
    start = mesh.get_local_rank() * actual.shape[1] if mesh is not None else 0
    end = min(start + actual.shape[1], ids.shape[1])
    valid = mask[:, start:end].cpu().bool()
    actual = actual[:, :end - start]
    if args.reference:
        expected = torch.load(args.reference / f'dp{parallel_dims.dp_rank}-logits.pt', weights_only=True)[:, start:end]
        if args.replicated_batch:
            expected = expected[:1].expand(args.batch_size, -1, -1)
        if not torch.equal(actual[valid].contiguous().view(torch.uint8),
                           expected[valid].contiguous().view(torch.uint8)):
            raise AssertionError('Valid full-vocabulary logits changed across CP')
    elif parallel_dims.cp_rank == 0 and parallel_dims.tp_rank == 0:
        torch.save(actual, args.output / f'dp{parallel_dims.dp_rank}-logits.pt')


def check_packed_model(model, args, device, parallel_dims):
    """Check decoder metadata propagation against independent document forwards."""
    lengths = (3, 126, 128)
    total = sum(lengths)
    multiple = model.unified_sequence_multiple
    padded = total + (-total) % multiple
    tokens = torch.arange(1, padded + 1, device=device).remainder(500).unsqueeze(0)
    mask = (torch.arange(padded, device=device) < total).unsqueeze(0)
    documents = (*lengths[:-1], padded - sum(lengths[:-1]))
    positions = torch.cat([torch.arange(length, device=device) for length in documents]).unsqueeze(0)
    local = padded // args.cp
    start = parallel_dims.cp_rank * local
    with torch.no_grad():
        actual = model(input_ids=tokens[:, start:start + local], attention_mask=mask,
                       position_ids=positions[:, start:start + local], use_cache=False,
                       unified_document_lengths=documents).logits.detach().cpu()
        if args.cp == 1:
            independent = []
            offset = 0
            for length in lengths:
                width = length + (-length) % multiple
                request = torch.nn.functional.pad(tokens[:, offset:offset + length], (0, width - length))
                request_mask = (torch.arange(width, device=device) < length).unsqueeze(0)
                request_positions = torch.arange(width, device=device).unsqueeze(0)
                result = model(input_ids=request, attention_mask=request_mask,
                               position_ids=request_positions, use_cache=False).logits
                independent.append(result[:, :length].detach().cpu())
                offset += length
            expected = torch.cat(independent, dim=1)
            if not torch.equal(actual[:, :total].contiguous().view(torch.uint8),
                               expected.contiguous().view(torch.uint8)):
                raise AssertionError('Packed model differs from independently executed documents')
            if parallel_dims.tp_rank == 0:
                torch.save(expected, args.output / f'dp{parallel_dims.dp_rank}-packed.pt')
        else:
            if args.reference is None:
                raise ValueError('Packed CP model comparison requires its CP1 reference')
            expected = torch.load(args.reference / f'dp{parallel_dims.dp_rank}-packed.pt', weights_only=True)
            end = min(start + local, total)
            if not torch.equal(actual[:, :end - start].contiguous().view(torch.uint8),
                               expected[:, start:end].contiguous().view(torch.uint8)):
                raise AssertionError('Packed decoder output bytes changed across CP')


def main():
    """Load the actual HP/FSDP/PIK recipe and compare the same logical tokens."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('qwen3', 'qwen3_moe'), default='qwen3_moe')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--create-only', action='store_true')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--cp', type=int, default=2)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--ep', type=int, default=2)
    parser.add_argument('--sp', action='store_true')
    parser.add_argument('--length', type=int, default=129)
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-role', action='store_true')
    parser.add_argument('--packed-check', action='store_true')
    parser.add_argument('--compare-gradients', action='store_true')
    parser.add_argument('--empty-query-ranks', action='store_true')
    parser.add_argument('--no-comm-fusion', action='store_true')
    parser.add_argument('--save-activations-on-cpu', action='store_true')
    parser.add_argument('--replicated-batch', action='store_true')
    args = parser.parse_args()
    if args.create_only:
        create_checkpoint(args.checkpoint, args.family)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(recipe(args))
    try:
        actor = trainer.actor
        trainer.residency.move(offload=False)
        torch.manual_seed(517 if args.replicated_batch else 517 + trainer.parallel_dims.dp_rank)
        rows = 1 if args.replicated_batch else args.batch_size
        ids = torch.randint(1, 500, (rows, args.length), device=trainer.device)
        if args.replicated_batch:
            ids = ids.expand(args.batch_size, -1).clone()
        mask = torch.ones_like(ids)
        mask[0, -7:] = 0
        if args.replicated_batch:
            mask[:, -7:] = 0
        elif args.batch_size > 1:
            mask[1, :3] = 0
        if args.empty_query_ranks:
            mask[:, 17:] = 0
        actions = mask.clone()
        actions[:, :4 if args.empty_query_ranks else max(4, args.length // 3)] = 0
        with torch.no_grad():
            old = actor.sequence_log_probs(ids, mask)
        valid = actions[:, 1:].bool()
        probabilities = old[valid].cpu().contiguous()
        if args.reference:
            expected = torch.load(args.reference / f'dp{trainer.parallel_dims.dp_rank}-probabilities.pt',
                                  weights_only=True)
            if args.replicated_batch:
                reference_case = json.loads((args.reference / 'rank0.json').read_text(encoding='utf-8'))
                if (not reference_case.get('replicated_batch') or args.dp * args.batch_size
                        != reference_case['dp'] * reference_case['batch_size']):
                    raise ValueError('Replicated-batch comparisons must preserve the global sample count')
                expected = expected.reshape(reference_case['batch_size'], -1)[0].repeat(args.batch_size)
            if not torch.equal(probabilities.view(torch.uint8), expected.view(torch.uint8)):
                raise AssertionError(f'Model CP bytes differ: max_abs={(probabilities - expected).abs().max()}')
        check_logits(actor, ids, mask, args, trainer.parallel_dims)
        if args.packed_check:
            check_packed_model(actor.actor_model, args, trainer.device, trainer.parallel_dims)
        reference_actor = trainer.targets.reference_actor
        reference_probs = None
        if reference_actor is not None:
            with torch.no_grad():
                reference_probs = reference_actor.sequence_log_probs(ids, mask)
            if not torch.equal(reference_probs[valid], old[valid]):
                raise AssertionError('Initial frozen Reference and Actor probabilities differ')
        rewards = (torch.ones(args.batch_size, device=trainer.device) if args.replicated_batch else
                   torch.tensor([1. if index % 2 == 0 else -1. for index in range(args.batch_size)],
                                device=trainer.device))
        experience = ExperienceBatch(
            trajectories=(), sequences=ids, attention_mask=mask, action_mask=actions,
            rewards=rewards, old_log_probs=old,
            responses=('',) * args.batch_size, generation_seconds=0.,
            advantages=rewards[:, None].expand_as(old).clone(),
            reference_log_probs=reference_probs,
        )
        before = {}
        for name, param in actor.actor_model.named_parameters():
            value = local_parameter(param).detach()
            before[name] = (value.cpu().clone() if (args.compare_gradients or args.save_activations_on_cpu
                                                  or args.replicated_batch)
                            else value.clone())
        comparison = None
        if args.compare_gradients:
            comparison = ParameterGradientComparison(actor.actor_model, args.output, args.reference, trainer.device)
            comparison.initial()
            torch.npu.empty_cache()
        saved_tensors = (torch.autograd.graph.save_on_cpu(device_type=trainer.device.type)
                         if args.save_activations_on_cpu else nullcontext())
        with saved_tensors, comparison.observe() if comparison is not None else nullcontext():
            metrics = actor.update(experience)
        torch.npu.synchronize()
        if comparison is not None:
            comparison.finish()
        changed = any(not torch.equal(before[name], local_parameter(param).detach().to(before[name].device))
                      for name, param in actor.actor_model.named_parameters())
        if not changed or not math.isfinite(metrics.gradient_norm) or metrics.gradient_norm <= 0:
            raise AssertionError('CP Actor must execute a finite nonzero gradient and change weights')
        report = {'tp': args.tp, 'cp': args.cp, 'dp': args.dp, 'ep': args.ep, 'sp': args.sp, 'length': args.length,
                  'comm_fusion': not args.no_comm_fusion,
                  'saved_tensors_on_cpu': args.save_activations_on_cpu,
                  'replicated_batch': args.replicated_batch,
                  'checkpoint': str(args.checkpoint), 'batch_size': args.batch_size,
                  'forward_exact': args.reference is not None,
                  'packed_exact': args.packed_check,
                  'weights_changed': changed, 'metrics': asdict(metrics)}
        if reference_actor is not None:
            with torch.no_grad():
                frozen_after = reference_actor.sequence_log_probs(ids, mask)
            if not torch.equal(frozen_after[valid], reference_probs[valid]):
                raise AssertionError('Frozen CP Reference changed after Actor optimization')
            report['reference_unchanged'] = True
        if args.reference:
            reference = json.loads((args.reference / 'rank0.json').read_text(encoding='utf-8'))
            reference_norm = reference['metrics']['gradient_norm']
            report['gradient_norm_relative_error'] = abs(metrics.gradient_norm - reference_norm) / reference_norm
            if report['gradient_norm_relative_error'] > .02:
                raise AssertionError(f'CP gradient scaling differs: {report}')
        if trainer.parallel_dims.cp_rank == 0 and trainer.parallel_dims.tp_rank == 0:
            torch.save(probabilities, args.output / f'dp{trainer.parallel_dims.dp_rank}-probabilities.pt')
        (args.output / f'rank{dist.get_rank()}.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        trainer.lifecycle.close()


if __name__ == '__main__':
    main()
