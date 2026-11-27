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
"""Exact CPU oracles for document-subring balance plus pruned Ring transport."""

from itertools import product
import unittest

from hyper_parallel.distributed.context_parallel.query_balance import query_task_plan_in_document_rings
from hyper_parallel.distributed.context_parallel.ring_layout import pruned_ring_plan


def _boundaries(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return tuple(result)


def _global_tokens(tasks, boundaries):
    return tuple(
        token
        for task in tasks
        for token in range(boundaries[task.document] + task.start, boundaries[task.document] + task.end)
    )


def _block_edges(blocks, queries, keys):
    edges = []
    for block in blocks:
        q_length = block.query_end - block.query_start
        k_length = block.key_end - block.key_start
        for q_index in range(block.query_start, block.query_end):
            for k_index in range(block.key_start, block.key_end):
                if not block.causal or k_index <= q_index + k_length - q_length:
                    edges.append((queries[q_index], keys[k_index]))
    return edges


class TestRingPruneBalance(unittest.TestCase):
    """Prove active-subring ownership, attention parity and exact remote KV spans."""

    def test_balanced_queries_stay_inside_each_documents_source_subring(self):
        cases = (((5, 11), 2), ((3, 5, 16), 4), ((7, 13, 4), 6))
        for (lengths, size), mode in product(cases, ("none", "zigzag")):
            with self.subTest(lengths=lengths, size=size, mode=mode):
                assignments = query_task_plan_in_document_rings(
                    lengths, size, mode,
                )
                boundaries = _boundaries(lengths)
                owners = {}
                for rank, tasks in enumerate(assignments):
                    for task in tasks:
                        owners.setdefault(task.document, set()).add(rank)
                for document, length in enumerate(lengths):
                    begin, end = boundaries[document:document + 2]
                    source_owners = set(range(begin // (sum(lengths) // size),
                                              (end - 1) // (sum(lengths) // size) + 1))
                    self.assertEqual(owners[document], source_owners)
                actual = sorted(token for tasks in assignments for token in _global_tokens(tasks, boundaries))
                self.assertEqual(actual, list(range(sum(lengths))))

    def test_combined_plan_preserves_every_attention_pair_and_prunes_exact_keys(self):
        cases = (((24,), 4), ((5, 11), 2), ((3, 5, 16), 4), ((7, 13, 4), 6))
        for (lengths, size), mode, causal in product(
                cases, ("none", "zigzag"), (False, True)):
            with self.subTest(lengths=lengths, size=size, mode=mode, causal=causal):
                assignments = query_task_plan_in_document_rings(
                    lengths, size, mode,
                )
                boundaries = _boundaries(lengths)
                global_tokens = [
                    (document, position)
                    for document, length in enumerate(lengths)
                    for position in range(length)
                ]
                expected_edges = {
                    (query, key)
                    for query, (q_document, q_position) in enumerate(global_tokens)
                    for key, (k_document, k_position) in enumerate(global_tokens)
                    if q_document == k_document and (not causal or k_position <= q_position)
                }
                actual_edges = []
                for rank in range(size):
                    plan = pruned_ring_plan(lengths, size, rank, causal, assignments)
                    query_tokens = _global_tokens(assignments[rank], boundaries)
                    local_key_tokens = _global_tokens(assignments[rank], boundaries)
                    actual_edges.extend(_block_edges(plan.local_blocks, query_tokens, local_key_tokens))
                    for owner in range(size):
                        if owner == rank:
                            continue
                        key_tokens = _global_tokens(plan.key_tasks[owner], boundaries)
                        actual_edges.extend(_block_edges(plan.blocks[owner], query_tokens, key_tokens))
                self.assertEqual(len(actual_edges), len(set(actual_edges)))
                self.assertEqual(set(actual_edges), expected_edges)

                for owner, task_list in enumerate(assignments):
                    owner_tokens = set(_global_tokens(task_list, boundaries))
                    remote_queries = [
                        (worker, task)
                        for worker, tasks in enumerate(assignments) if worker != owner
                        for task in tasks
                    ]
                    expected_remote = {
                        token for token in owner_tokens
                        if any(
                            global_tokens[token][0] == task.document
                            and (not causal or global_tokens[token][1] < task.end)
                            for _, task in remote_queries
                        )
                    }
                    plan = pruned_ring_plan(lengths, size, 0, causal, assignments)
                    actual_remote = {
                        token for begin, end in plan.spans[owner] for token in range(begin, end)
                    }
                    self.assertEqual(actual_remote, expected_remote)
                    owner_compact = _global_tokens(task_list, boundaries)
                    actual_compact = {
                        owner_compact[token] for begin, end in plan.source_spans[owner]
                        for token in range(begin, end)
                    }
                    self.assertEqual(actual_compact, expected_remote)

    def test_invalid_subring_plans_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal contiguous"):
            query_task_plan_in_document_rings((5, 10), 4, "zigzag")
        with self.assertRaisesRegex(ValueError, "none or zigzag"):
            query_task_plan_in_document_rings((8,), 2, "greedy")

    def test_attention_batches_separate_repeated_query_rows(self):
        """A query's KV fragments must merge in separate native FA calls."""
        assignments = query_task_plan_in_document_rings((5, 11), 2, "zigzag")
        plan = pruned_ring_plan((5, 11), 2, 1, True, assignments)
        for batches in (*plan.batches, plan.local_batches):
            for blocks in batches:
                query_rows = [
                    row for block in blocks for row in range(block.query_start, block.query_end)
                ]
                self.assertEqual(len(query_rows), len(set(query_rows)))


if __name__ == "__main__":
    unittest.main()
