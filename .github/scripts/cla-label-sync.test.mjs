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
import {
  CLAClient, CLA_CONTEXT, CLA_LABELS, LEGACY_LABELS, SignatureService, evaluate_cla, event_pulls, sync_cla_labels,
} from './cla-label-sync.mjs';

const REPO = 'mindspore-ai/hyper-parallel';
const SHA = 'a'.repeat(40);
const AUTHOR = { id: 123, login: 'pr-creator' };
function pull(overrides = {}) {
  return { number: 1, state: 'open', head: { sha: SHA }, user: AUTHOR, labels: [{ name: 'bug' }], ...overrides };
}
function signature(overrides = {}) {
  return { user_id: AUTHOR.id, login: AUTHOR.login, signed: true, agreement: 'https://gist.github.com/cla', ...overrides };
}
function response(body, status = 200) { return { ok: status === 200, status, json: async () => body }; }
function service_fixture(replies = [response({ gist: signature().agreement }), response({ signed: true })]) {
  const calls = [];
  const service = new SignatureService('test-secret', async (url, options) => {
    calls.push({ url, ...options, body: JSON.parse(options.body) });
    return replies.shift();
  });
  return { service, calls };
}
class FakeClient {
  constructor() {
    this.repository = REPO;
    this.pull = pull();
    this.signature = signature();
    this.writes = [];
    this.ensure_count = 0;
  }
  ensure_labels() { this.ensure_count += 1; }
  get_pull() { return structuredClone(this.pull); }
  async get_signature() { return structuredClone(this.signature); }
  list_pulls() { return [this.get_pull()]; }
  pulls_for_sha() { return [this.get_pull()]; }
  add_label(number, name) { this.writes.push(['add', number, name]); this.pull.labels.push({ name }); }
  remove_label(number, name) {
    this.writes.push(['remove', number, name]);
    this.pull.labels = this.pull.labels.filter(label => label.name !== name);
  }
}
function sync(client, options = {}) { return sync_cla_labels(client, [{ number: 1 }], { write: true, ...options }); }

test('only the identified creator signature determines the exact new label names', () => {
  assert.equal(evaluate_cla(pull(), signature()).label, 'cla-signed');
  assert.equal(evaluate_cla(pull(), signature({ signed: false })).label, 'cla-unsigned');
  for (const value of [{ user_id: 456 }, { login: 'another-user' }, { signed: 'true' }, { agreement: '' }]) {
    assert.throws(() => evaluate_cla(pull(), signature(value)), /does not match/);
  }
});
test('personal query sends the creator identity, never a PR number or aggregate result', async () => {
  const { service, calls } = service_fixture();
  assert.deepEqual(await service.lookup_author(AUTHOR, REPO), signature());
  assert.equal(calls[0].url, 'https://cla-assistant.io/api/cla/getLinkedItem');
  assert.equal(calls[0].headers['x-token'], undefined);
  assert.deepEqual(calls[0].body, { owner: 'mindspore-ai', repo: 'hyper-parallel' });
  assert.equal(calls[1].url, 'https://cla-assistant.io/api/cla/hasSignature');
  assert.deepEqual(calls[1].body, { owner: 'mindspore-ai', repo: 'hyper-parallel', user: AUTHOR.login, userId: AUTHOR.id });
  assert.equal(calls[1].headers['x-token'], 'test-secret');
  assert.equal(calls[1].redirect, 'error');
  assert.ok(calls[1].signal instanceof AbortSignal);
});
test('explicit unsigned response is accepted without coercing missing or string values', async () => {
  for (const signed of [false, true]) {
    const { service } = service_fixture([response({ gist: 'agreement' }), response({ signed })]);
    assert.equal((await service.lookup_author(AUTHOR, REPO)).signed, signed);
  }
  for (const body of [{}, { signed: 'false' }, { signed: null }]) {
    const { service } = service_fixture([response({ gist: 'agreement' }), response(body)]);
    await assert.rejects(service.lookup_author(AUTHOR, REPO), /signature decision/);
  }
});
test('missing agreement cannot become a signed label through the service null-CLA behavior', async () => {
  for (const linked of [null, {}, { gist: '' }, { gist: ' ' }]) {
    const { service, calls } = service_fixture([response(linked)]);
    await assert.rejects(service.lookup_author(AUTHOR, REPO), /No current CLA/);
    assert.equal(calls.length, 1);
  }
});
test('missing credentials, invalid identities and other repositories fail before making requests', async () => {
  const { service, calls } = service_fixture();
  for (const author of [null, {}, { id: -1, login: 'user' }, { id: '123', login: 'user' }]) {
    await assert.rejects(service.lookup_author(author, REPO), /Invalid PR creator/);
  }
  await assert.rejects(service.lookup_author(AUTHOR, 'another/repo'), /restricted/);
  service.token = '';
  await assert.rejects(service.lookup_author(AUTHOR, REPO), /CLA_ASSISTANT_TOKEN/);
  assert.deepEqual(calls, []);
});
test('authorization, service, network and malformed response errors never become unsigned', async () => {
  for (const status of [401, 403, 429, 500]) {
    const { service } = service_fixture([response({ gist: 'agreement' }), response({}, status)]);
    await assert.rejects(service.lookup_author(AUTHOR, REPO), new RegExp(`HTTP ${status}`));
  }
  const { service } = service_fixture([{ ok: true, json: async () => { throw new Error('bad JSON'); } }]);
  await assert.rejects(service.lookup_author(AUTHOR, REPO), /invalid JSON/);
  service.fetch = async () => { throw new Error('test-secret'); };
  await assert.rejects(service.lookup_author(AUTHOR, REPO), error =>
    error.message.includes('failed or timed out') && !error.message.includes('test-secret'));
});
test('signing removes conflicting and all obsolete names, preserving unrelated labels', async () => {
  const client = new FakeClient();
  const removed = [CLA_LABELS[1], ...LEGACY_LABELS];
  client.pull.labels.push(...removed.map(name => ({ name })));
  assert.equal((await sync(client)).errors.length, 0);
  assert.deepEqual(client.writes, [...removed.map(name => ['remove', 1, name]), ['add', 1, CLA_LABELS[0]]]);
  assert.deepEqual(client.pull.labels.map(label => label.name), ['bug', CLA_LABELS[0]]);
});
test('unsigned creator result removes an existing signed label', async () => {
  const client = new FakeClient();
  client.signature.signed = false;
  client.pull.labels.push({ name: CLA_LABELS[0] });
  assert.equal((await sync(client)).errors.length, 0);
  assert.deepEqual(client.writes, [['remove', 1, CLA_LABELS[0]], ['add', 1, CLA_LABELS[1]]]);
});
test('signatures belong to the creator even when another contributor makes aggregate CLA fail', async () => {
  const identities = [];
  const client = new CLAClient(REPO, undefined, { lookup_author: async (user, repository) => {
    identities.push({ user, repository });
    return signature();
  } });
  client.get_pull = () => pull({ author: { login: 'different-committer' }, statuses: [{ state: 'failure' }] });
  const result = await sync(client, { write: false });
  assert.equal(result.reports[0].label, CLA_LABELS[0]);
  assert.deepEqual(identities, [{ user: AUTHOR, repository: REPO }]);
});
test('repeated synchronization is idempotent and dry-run never mutates labels', async () => {
  const client = new FakeClient();
  assert.equal((await sync(client, { write: false })).reports[0].label, CLA_LABELS[0]);
  assert.equal(client.ensure_count, 0);
  assert.deepEqual(client.writes, []);
  await sync(client);
  client.writes = [];
  await sync(client);
  assert.deepEqual(client.writes, []);
});
test('closed PRs do not query signatures; writes to another repository are rejected', async () => {
  const client = new FakeClient();
  client.pull.state = 'closed';
  client.get_signature = () => { throw new Error('Unexpected signature query'); };
  assert.deepEqual((await sync(client)).reports, []);
  assert.deepEqual(client.writes, []);
  client.repository = 'another/repo';
  await assert.rejects(sync(client), /restricted/);
});
test('head changes or closing during recheck prevent writes', async () => {
  for (const change of [{ head: { sha: 'b'.repeat(40) } }, { state: 'closed' }]) {
    const client = new FakeClient();
    let reads = 0;
    client.get_pull = () => ++reads === 1 ? structuredClone(client.pull) : { ...client.pull, ...change };
    assert.equal((await sync(client)).errors.length, 1);
    assert.deepEqual(client.writes, []);
  }
});
test('creator identity changes during reconciliation prevent stale writes', async () => {
  const client = new FakeClient();
  let reads = 0;
  client.get_pull = () => ++reads === 1 ? pull() : pull({ user: { id: 456, login: 'different-user' } });
  client.get_signature = async item => signature({ user_id: item.user.id, login: item.user.login });
  assert.equal((await sync(client)).errors.length, 1);
  assert.deepEqual(client.writes, []);
});
test('signature or agreement changes during reconciliation prevent stale writes', async () => {
  for (const change of [{ signed: false }, { agreement: 'new-agreement' }]) {
    const client = new FakeClient();
    let reads = 0;
    client.get_signature = async () => ++reads === 1 ? signature() : signature(change);
    assert.equal((await sync(client)).errors.length, 1);
    assert.deepEqual(client.writes, []);
  }
});
test('failed queries leave labels untouched and do not classify the author as unsigned', async () => {
  const client = new FakeClient();
  client.pull.labels.push({ name: CLA_LABELS[0] });
  client.get_signature = async () => { throw new Error('API unavailable'); };
  const result = await sync(client);
  assert.match(result.errors[0], /API unavailable/);
  assert.deepEqual(result.reports, []);
  assert.deepEqual(client.writes, []);
});
test('failed removal blocks a conflicting addition, while already-removed labels are tolerated', async () => {
  const client = new FakeClient();
  client.pull.labels.push({ name: CLA_LABELS[1] });
  client.remove_label = () => { throw new Error('denied'); };
  assert.match((await sync(client)).errors[0], /denied/);
  assert.deepEqual(client.writes, []);
  client.remove_label = () => {
    client.pull.labels = client.pull.labels.filter(label => label.name !== CLA_LABELS[1]);
    throw new Error('already absent');
  };
  assert.equal((await sync(client)).errors.length, 0);
  assert.deepEqual(client.writes, [['add', 1, CLA_LABELS[0]]]);
});
test('status events only locate open PRs on the current head, independently of aggregate outcome', () => {
  const client = new FakeClient();
  client.pulls_for_sha = () => [pull(), pull({ number: 2, state: 'closed' }),
    pull({ number: 3, head: { sha: 'b'.repeat(40) } })];
  for (const state of ['success', 'pending', 'failure']) {
    const event = { context: CLA_CONTEXT, sha: SHA, state };
    assert.deepEqual(event_pulls(client, 'status', event).map(item => item.number), [1]);
  }
  assert.deepEqual(event_pulls(client, 'status', { context: 'test', sha: SHA }), []);
  assert.throws(() => event_pulls(client, 'status', { context: CLA_CONTEXT, sha: 'invalid' }), /Invalid/);
  assert.deepEqual(event_pulls(client, 'pull_request_target', { pull_request: { number: 5 } }), [{ number: 5 }]);
  assert.equal(event_pulls(client, 'workflow_dispatch', {}).length, 1);
  assert.equal(event_pulls(client, 'schedule', {}).length, 1);
});
test('API adapter paginates PRs and adds/removes only individual labels', () => {
  const calls = [];
  const client = new CLAClient(REPO, (command, args, options) => {
    calls.push({ command, args, options });
    return args.includes('--paginate') ? JSON.stringify([[pull()], [pull({ number: 2 })]]) : '[]';
  });
  assert.equal(client.pulls_for_sha(SHA).length, 2);
  client.add_label(1, CLA_LABELS[0]);
  client.remove_label(1, 'mindspore-cla/no');
  assert.ok(calls[0].args.includes('--paginate'));
  assert.ok(calls[1].args.includes('POST'));
  assert.deepEqual(JSON.parse(calls[1].options.input), { labels: [CLA_LABELS[0]] });
  assert.ok(calls[2].args.includes('DELETE'));
  assert.ok(calls[2].args.some(value => value.endsWith('mindspore-cla%2Fno')));
});
test('repository label provisioning is idempotent and tolerates concurrent creation', () => {
  const client = new CLAClient(REPO);
  let reads = 0;
  let creates = 0;
  client.api = (path, options) => {
    if (options.method === 'POST') { creates += 1; throw new Error('already exists'); }
    return ++reads === 1 ? [] : CLA_LABELS.map(name => ({ name }));
  };
  client.ensure_labels();
  assert.equal(creates, 2);
  creates = 0;
  client.ensure_labels();
  assert.equal(creates, 0);
});
test('an already-synchronized sweep uses one personal query without unnecessary rechecks', async () => {
  const client = new FakeClient();
  client.pull.labels.push({ name: CLA_LABELS[0] });
  let reads = 0;
  client.get_pull = () => { throw new Error('Unexpected repeated PR read'); };
  client.get_signature = async () => { reads += 1; return signature(); };
  assert.equal((await sync_cla_labels(client, [client.pull], { write: true })).errors.length, 0);
  assert.equal(reads, 1);
  assert.deepEqual(client.writes, []);
});
test('a managed label added concurrently prevents a conflicting signed label', async () => {
  const client = new FakeClient();
  let reads = 0;
  client.get_pull = () => ++reads === 1 ? pull() : pull({ labels: [{ name: CLA_LABELS[1] }] });
  assert.equal((await sync(client)).errors.length, 1);
  assert.deepEqual(client.writes, []);
});
