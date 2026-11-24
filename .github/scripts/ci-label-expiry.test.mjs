// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { evaluate_labels, LabelClient, PASS_LABELS, reconcile_labels, STATUS_CONTEXT, TTL_MS } from './ci-label-expiry.mjs';

const NOW = Date.parse('2026-10-30T12:00:00Z');
const [NORMAL, OVERTIME] = PASS_LABELS;
function pull(names = [NORMAL], overrides = {}) {
  return { number: 1, state: 'open', head: { sha: 'head' }, merge_commit_sha: 'merge',
    labels: names.map(name => ({ name })), ...overrides };
}
function event(name = NORMAL, age = TTL_MS, id = 1, action = 'labeled') {
  return { id, event: action, label: { name }, created_at: new Date(NOW - age).toISOString() };
}
class FakeClient {
  constructor(names = [NORMAL]) {
    this.pull = pull(names);
    this.events = names.map((name, index) => event(name, TTL_MS, index + 1));
    this.statuses = { head: [], merge: [] };
    this.writes = [];
  }
  list_pulls() { return [structuredClone(this.pull)]; }
  get_pull() { return structuredClone(this.pull); }
  get_events() { return structuredClone(this.events); }
  get_statuses(sha) { return structuredClone(this.statuses[sha] ?? []); }
  set_status(sha, state) {
    this.writes.push(['status', sha, state]);
    this.statuses[sha] = [{ context: STATUS_CONTEXT, state },
      ...(this.statuses[sha] ?? []).filter(status => status.context !== STATUS_CONTEXT)];
  }
  remove_label(number, name) {
    this.writes.push(['remove', number, name]);
    this.pull.labels = this.pull.labels.filter(label => label.name !== name);
  }
}
function sweep(client, options = {}) {
  return reconcile_labels(client, { write: true, now: () => NOW, completed_head: 'head', ...options });
}

test('24 hours is expired; one millisecond earlier is valid for either label', () => {
  for (const name of PASS_LABELS) {
    assert.equal(evaluate_labels(pull([name]), [event(name)], NOW)[0].expired, true);
    assert.equal(evaluate_labels(pull([name]), [event(name)], NOW - 1)[0].expired, false);
  }
});
test('latest re-addition resets only that label, even with out-of-order events', () => {
  const events = [event(NORMAL, 1000, 4), event(OVERTIME, TTL_MS, 2),
    event(NORMAL, TTL_MS * 2, 1), event(NORMAL, 2000, 3, 'unlabeled')];
  const result = evaluate_labels(pull(PASS_LABELS), events, NOW);
  assert.deepEqual(result.map(label => label.expired), [false, true]);
  assert.equal(result[0].event_id, 4);
});
test('unrelated activity and PR updated_at do not renew labels', () => {
  const value = pull([NORMAL, 'bug'], { updated_at: new Date(NOW).toISOString() });
  const result = evaluate_labels(value, [event(), event('bug', 0, 2)], NOW);
  assert.equal(result.length, 1);
  assert.equal(result[0].expired, true);
});
test('unattached labels are ignored; missing, removed, invalid or future additions fail closed', () => {
  assert.deepEqual(evaluate_labels(pull([]), [event()], NOW), []);
  for (const events of [[], [event(NORMAL, 0, 2, 'unlabeled')],
    [{ ...event(), created_at: 'invalid' }], [event(NORMAL, -1000)]]) {
    assert.throws(() => evaluate_labels(pull(), events, NOW), /Cannot establish/);
  }
});
test('both expired labels are removed only after failing head and merge commit statuses', () => {
  const client = new FakeClient(PASS_LABELS);
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['status', 'merge', 'failure'],
    ['remove', 1, NORMAL], ['remove', 1, OVERTIME]]);
  assert.deepEqual(client.pull.labels, []);
});
test('a still-valid alternative label keeps the validity status passing', () => {
  const client = new FakeClient(PASS_LABELS);
  client.events[1] = event(OVERTIME, 1000, 2);
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, [['status', 'head', 'success'], ['status', 'merge', 'success'], ['remove', 1, NORMAL]]);
});
test('dry runs never write, and closed PRs are untouched', () => {
  const client = new FakeClient();
  assert.equal(sweep(client, { write: false }).reports[0].labels[0].expired, true);
  assert.deepEqual(client.writes, []);
  client.pull.state = 'closed';
  assert.deepEqual(sweep(client).reports, []);
  assert.deepEqual(client.writes, []);
});
test('a re-added label between evaluation and removal is preserved', () => {
  const client = new FakeClient(PASS_LABELS);
  client.events[1] = event(OVERTIME, 1000, 2);
  let reads = 0;
  client.get_events = () => ++reads === 1 ? client.events : [event(NORMAL, 1000, 3), client.events[1]];
  assert.equal(sweep(client).errors.length, 1);
  assert.equal(client.writes.some(write => write[0] === 'remove' || write[2] === 'success'), false);
  assert.equal(client.pull.labels.length, 2);
});
test('a changed head or closed PR does not receive an outdated mutation', () => {
  for (const change of [{ head: { sha: 'new-head' } }, { state: 'closed' }]) {
    const client = new FakeClient();
    client.get_pull = () => ({ ...client.pull, ...change });
    assert.equal(sweep(client).errors.length, 1);
    assert.equal(client.writes.some(write => write[1] === 'head'), false);
    assert.equal(client.writes.some(write => write[0] === 'remove' || write[2] === 'success'), false);
  }
});
test('status publication failure prevents removal and is retried next sweep', () => {
  const client = new FakeClient();
  client.set_status = () => { throw new Error('permission denied'); };
  assert.match(sweep(client).errors[0], /permission denied/);
  assert.equal(client.pull.labels.length, 1);
  client.set_status = FakeClient.prototype.set_status;
  assert.equal(sweep(client).errors.length, 0);
  assert.equal(client.pull.labels.length, 0);
});
test('repairs stale successes even when a prior sweep already removed labels', () => {
  const client = new FakeClient([]);
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['status', 'merge', 'failure']]);
  client.writes = [];
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, []);
});
test('unrelated status contexts are preserved and a missing merge SHA is ignored', () => {
  const client = new FakeClient([]);
  client.pull.merge_commit_sha = null;
  client.statuses.head = [{ context: 'unit-test', state: 'success' }];
  sweep(client);
  assert.deepEqual(client.writes, [['status', 'head', 'failure']]);
  assert.deepEqual(client.statuses.head.find(status => status.context === 'unit-test'),
    { context: 'unit-test', state: 'success' });
});
test('label deletion errors are surfaced unless another actor already removed it', () => {
  const client = new FakeClient();
  client.remove_label = () => { throw new Error('network error'); };
  assert.match(sweep(client).errors[0], /network error/);
  client.remove_label = () => { client.pull.labels = []; throw new Error('already removed'); };
  assert.equal(sweep(client).errors.length, 0);
});
test('API read failure does not delete labels and does not stop other PRs', () => {
  const client = new FakeClient();
  client.list_pulls = () => [pull(), pull([], { number: 2 })];
  client.get_events = () => { throw new Error('API unavailable'); };
  client.get_pull = number => pull([], { number });
  assert.equal(sweep(client).errors.length, 1);
  assert.equal(client.writes.some(item => item[0] === 'remove'), false);
});
test('adapter publishes commit statuses without patching immutable Actions check runs', () => {
  const calls = [];
  const client = new LabelClient('owner/repo', (command, args, options) => {
    calls.push({ command, args, options });
    assert.equal(args.includes('PATCH'), false, 'Actions-owned check conclusions cannot be patched');
    if (args.includes('POST')) return '';
    if (args.some(arg => arg.includes('/status?'))) {
      return JSON.stringify([{ statuses: [{ context: 'unit-test', state: 'success' }] },
        { statuses: [{ context: STATUS_CONTEXT, state: 'failure' }] }]);
    }
    return JSON.stringify([[event()], [event(NORMAL, 1000, 2)]]);
  });
  assert.equal(client.get_events(1).length, 2);
  assert.equal(client.get_statuses('head').length, 2);
  client.set_status('head', 'failure', 1);
  assert.ok(calls[0].args.includes('--paginate'));
  assert.ok(calls[0].args.includes('--slurp'));
  assert.ok(calls[1].args.includes('--paginate'));
  assert.ok(calls[2].args.includes('repos/owner/repo/statuses/head'));
  assert.deepEqual(JSON.parse(calls[2].options.input), {
    context: 'label-check', state: 'failure',
    description: 'An open PR using this commit needs a fresh CI pass label. Run CI again.',
    target_url: 'https://github.com/owner/repo/pull/1',
  });
  assert.throws(() => new LabelClient('bad repo'), /Invalid GH_REPO/);
});

test('scheduled scans skip unlabeled PRs; completion signals only reconcile their head', () => {
  const client = new FakeClient([]);
  client.get_statuses = () => { throw new Error('Unexpected status query'); };
  assert.deepEqual(sweep(client, { completed_head: undefined }).reports, []);
  assert.deepEqual(sweep(client, { completed_head: 'another-head' }).reports, []);
  assert.deepEqual(client.writes, []);
});

test('a fresh CI result restores expired statuses and unchanged sweeps do not append statuses', () => {
  const client = new FakeClient();
  assert.deepEqual(sweep(client).errors, []);
  client.pull.labels = [{ name: OVERTIME }];
  client.events.push(event(OVERTIME, 1000, 2));
  client.writes = [];
  assert.deepEqual(sweep(client).errors, []);
  assert.deepEqual(client.writes, [['status', 'head', 'success'], ['status', 'merge', 'success']]);
  client.writes = [];
  assert.deepEqual(sweep(client).errors, []);
  assert.deepEqual(client.writes, []);
});

test('a renewal race conservatively blocks both commits without removing labels, then recovers', () => {
  const client = new FakeClient();
  client.set_status = (sha, state) => {
    FakeClient.prototype.set_status.call(client, sha, state);
    client.events.push(event(NORMAL, 1000, 2));
  };
  assert.match(sweep(client).errors[0], /changed before updating the status/);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['status', 'merge', 'failure']]);
  assert.equal(client.pull.labels.length, 1);
  client.set_status = FakeClient.prototype.set_status;
  assert.deepEqual(sweep(client).errors, []);
  assert.equal(client.statuses.head[0].state, 'success');
  assert.equal(client.statuses.merge[0].state, 'success');
});

test('a label expiring during status evaluation cannot publish success', () => {
  const client = new FakeClient();
  let reads = 0;
  assert.match(sweep(client, { now: () => ++reads === 1 ? NOW - 1 : NOW }).errors[0], /expired|changed/);
  assert.equal(client.writes.some(write => write[2] === 'success'), false);
});

test('failure on the second SHA keeps labels and a retry completes the status barrier', () => {
  const client = new FakeClient();
  client.set_status = (sha, state) => {
    if (sha === 'merge') throw new Error('merge status unavailable');
    FakeClient.prototype.set_status.call(client, sha, state);
  };
  assert.match(sweep(client).errors[0], /merge status unavailable/);
  assert.equal(client.pull.labels.length, 1);
  client.set_status = FakeClient.prototype.set_status;
  client.writes = [];
  assert.deepEqual(sweep(client).errors, []);
  assert.deepEqual(client.writes, [['status', 'merge', 'failure'], ['remove', 1, NORMAL]]);
});

test('head and merge using the same SHA publish only one status', () => {
  const client = new FakeClient();
  client.pull.merge_commit_sha = 'head';
  assert.deepEqual(sweep(client).errors, []);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['remove', 1, NORMAL]]);
});

test('a fresh sibling PR cannot clear the failure status of an expired PR sharing its head', () => {
  const client = new FakeClient();
  const sibling = pull([NORMAL], { number: 2, merge_commit_sha: 'sibling-merge' });
  const original_get_pull = client.get_pull.bind(client);
  const original_get_events = client.get_events.bind(client);
  client.statuses['sibling-merge'] = [];
  client.list_pulls = () => [original_get_pull(), structuredClone(sibling)];
  client.get_pull = number => number === 2 ? structuredClone(sibling) : original_get_pull();
  client.get_events = number => number === 2 ? [event(NORMAL, 1000, 2)] : original_get_events();
  assert.deepEqual(sweep(client).errors, []);
  assert.equal(client.statuses.head[0].state, 'failure');
  assert.equal(client.statuses.merge[0].state, 'failure');
  assert.equal(client.statuses['sibling-merge'][0].state, 'success');
  assert.equal(client.pull.labels.length, 0);
  client.pull.state = 'closed';
  assert.deepEqual(sweep(client).errors, []);
  assert.equal(client.statuses.head[0].state, 'success');
});

test('a new PR sharing the SHA prevents publication of an outdated success', () => {
  const client = new FakeClient();
  client.events = [event(NORMAL, 1000)];
  let reads = 0;
  client.list_pulls = () => ++reads === 1 ? [structuredClone(client.pull)] :
    [structuredClone(client.pull), pull([], { number: 2 })];
  assert.match(sweep(client).errors[0], /PRs sharing this commit changed/);
  assert.equal(client.writes.some(write => write[2] === 'success'), false);
});

test('expiry workflow grants commit status writes instead of check-run writes', () => {
  const workflow = readFileSync(new URL('../workflows/expire-ci-pass-labels.yml', import.meta.url), 'utf8');
  assert.match(workflow, /^  statuses: write$/m);
  assert.doesNotMatch(workflow, /^  checks: write$/m);
  assert.match(workflow, /ref: \$\{\{ github.event.repository.default_branch \}\}/);
});

test('scheduled and completion runs serialize writes to the same commit statuses', () => {
  const workflow = readFileSync(new URL('../workflows/expire-ci-pass-labels.yml', import.meta.url), 'utf8');
  assert.match(workflow, /^  group: expire-ci-pass-labels$/m);
  assert.match(workflow, /^  cancel-in-progress: false$/m);
});

test('a label expiring during the final membership read cannot publish success', () => {
  const client = new FakeClient();
  let clock = NOW - 1;
  let reads = 0;
  client.list_pulls = () => {
    if (++reads > 1) clock = NOW;
    return [structuredClone(client.pull)];
  };
  assert.ok(sweep(client, { now: () => clock }).errors.length > 0);
  assert.equal(client.writes.some(write => write[2] === 'success'), false);
  assert.equal(client.statuses.head[0].state, 'failure');
  assert.equal(client.statuses.merge[0].state, 'failure');
});

test('an unverifiable label revokes earlier successes when status writes remain available', () => {
  for (const unreadable of [false, true]) {
    const client = new FakeClient();
    client.statuses.head = [{ context: STATUS_CONTEXT, state: 'success' }];
    client.statuses.merge = [{ context: STATUS_CONTEXT, state: 'success' }];
    client.events = [];
    if (unreadable) client.get_events = () => { throw new Error('events unavailable'); };
    assert.ok(sweep(client).errors.length > 0);
    assert.equal(client.statuses.head[0].state, 'failure');
    assert.equal(client.statuses.merge[0].state, 'failure');
    assert.equal(client.pull.labels.length, 1);
  }
});

test('a failed head status write still attempts to block the merge commit', () => {
  const client = new FakeClient();
  client.statuses.head = [{ context: STATUS_CONTEXT, state: 'success' }];
  client.statuses.merge = [{ context: STATUS_CONTEXT, state: 'success' }];
  client.set_status = (sha, state) => {
    if (sha === 'head') throw new Error('head status unavailable');
    FakeClient.prototype.set_status.call(client, sha, state);
  };
  assert.ok(sweep(client).errors.length > 0);
  assert.equal(client.statuses.merge[0].state, 'failure');
  assert.equal(client.pull.labels.length, 1);
});

test('expiry alone does not revoke success until a write reconciliation runs', () => {
  const client = new FakeClient();
  assert.deepEqual(sweep(client, { now: () => NOW - 1 }).errors, []);
  assert.equal(client.statuses.head[0].state, 'success');
  assert.equal(sweep(client, { write: false, now: () => NOW + 1 }).reports[0].labels[0].expired, true);
  assert.equal(client.statuses.head[0].state, 'success');
  assert.deepEqual(sweep(client, { now: () => NOW + 1 }).errors, []);
  assert.equal(client.statuses.head[0].state, 'failure');
});

test('a status API outage cannot revoke a success already stored on GitHub', () => {
  const client = new FakeClient();
  client.statuses.head = [{ context: STATUS_CONTEXT, state: 'success' }];
  client.statuses.merge = [{ context: STATUS_CONTEXT, state: 'success' }];
  client.set_status = () => { throw new Error('status service unavailable'); };
  assert.ok(sweep(client).errors.length > 0);
  assert.equal(client.statuses.head[0].state, 'success');
  assert.equal(client.pull.labels.length, 1);
});

test('an unchanged success is revoked if the label expires while its status is read', () => {
  const client = new FakeClient();
  client.statuses.head = [{ context: STATUS_CONTEXT, state: 'success' }];
  client.statuses.merge = [{ context: STATUS_CONTEXT, state: 'success' }];
  let clock = NOW - 1;
  client.get_statuses = sha => {
    clock = NOW;
    return FakeClient.prototype.get_statuses.call(client, sha);
  };
  assert.ok(sweep(client, { now: () => clock }).errors.length > 0);
  assert.equal(client.statuses.head[0].state, 'failure');
  assert.equal(client.statuses.merge[0].state, 'failure');
});
