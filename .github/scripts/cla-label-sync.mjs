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

import { appendFileSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { GitHubClient } from './reviewer-approval-gate.mjs';

export const CLA_CONTEXT = 'license/cla';
export const CLA_LABELS = ['cla-signed', 'cla-unsigned'];
export const LEGACY_LABELS = ['mindspore-cla/yes', 'mindspore-cla/no', 'signed-CLA', 'unsigned-CLA'];
const MANAGED_LABELS = [...CLA_LABELS, ...LEGACY_LABELS];
const REPOSITORY = 'mindspore-ai/hyper-parallel';

/** Query the hosted service for one GitHub user's current signature. */
export class SignatureService {
  constructor(token, fetch_impl = fetch) {
    this.token = token;
    this.fetch = fetch_impl;
  }

  async request(method, args, authenticated = false) {
    const headers = { 'Content-Type': 'application/json', Accept: 'application/json' };
    if (authenticated) headers['x-token'] = this.token;
    let response;
    try {
      response = await this.fetch(`https://cla-assistant.io/api/cla/${method}`, {
        method: 'POST', headers, body: JSON.stringify(args), redirect: 'error', signal: AbortSignal.timeout(20000),
      });
    } catch {
      // Do not expose credentials, request headers, or arbitrary upstream errors.
      throw new Error('CLA Assistant request failed or timed out');
    }
    if (!response.ok) throw new Error(`CLA Assistant ${method} returned HTTP ${response.status}`);
    try { return await response.json(); } catch { throw new Error('CLA Assistant returned invalid JSON'); }
  }

  async lookup_author(author, repository) {
    if (!this.token) throw new Error('Configure the CLA_ASSISTANT_TOKEN repository secret for author queries');
    if (!Number.isSafeInteger(author?.id) || author.id <= 0 || !author.login) throw new Error('Invalid PR creator');
    if (repository !== REPOSITORY) throw new Error('CLA queries are restricted to the configured repository');
    const [owner, repo] = repository.split('/');
    const linked = await this.request('getLinkedItem', { owner, repo });
    // The service treats a missing agreement as truthy internally; do not call
    // that a signed agreement. Require an actual configured CLA first.
    if (typeof linked?.gist !== 'string' || !linked.gist.trim()) throw new Error('No current CLA is configured');
    const result = await this.request('hasSignature', { owner, repo, user: author.login, userId: author.id }, true);
    if (typeof result?.signed !== 'boolean') throw new Error('CLA Assistant did not return a signature decision');
    return { user_id: author.id, login: author.login, signed: result.signed, agreement: linked.gist };
  }
}

/** Only the PR creator's verified individual result controls the label. */
export function evaluate_cla(pull, signature) {
  if (signature.user_id !== pull.user.id || signature.login !== pull.user.login ||
      typeof signature.signed !== 'boolean' || !signature.agreement) {
    throw new Error('Signature evidence does not match the PR creator');
  }
  return { label: CLA_LABELS[signature.signed ? 0 : 1], signature,
    reason: `PR creator ${pull.user.login} ${signature.signed ? 'has signed' : 'has not signed'} the current CLA` };
}

/** Adapt GitHub APIs without replacing unrelated PR labels. */
export class CLAClient extends GitHubClient {
  constructor(repository, run, service = new SignatureService(process.env.CLA_ASSISTANT_TOKEN)) {
    super(repository, run);
    this.service = service;
  }
  get_pull(number) { return this.api(`pulls/${number}`); }
  get_signature(pull) { return this.service.lookup_author(pull.user, this.repository); }
  pulls_for_sha(sha) { return this.api(`commits/${sha}/pulls?per_page=100`, { paginate: true }); }
  remove_label(number, name) {
    this.api(`issues/${number}/labels/${encodeURIComponent(name)}`, { method: 'DELETE' });
  }
  add_label(number, name) {
    this.api(`issues/${number}/labels`, { method: 'POST', body: { labels: [name] } });
  }
  ensure_labels() {
    const labels = this.api('labels?per_page=100', { paginate: true });
    const specs = [
      { name: CLA_LABELS[0], color: '0e8a16', description: 'The PR creator has signed the current CLA.' },
      { name: CLA_LABELS[1], color: 'd93f0b', description: 'The PR creator has not signed the current CLA.' },
    ];
    for (const spec of specs) {
      if (labels.some(label => label.name === spec.name)) continue;
      try { this.api('labels', { method: 'POST', body: spec }); } catch (error) {
        const current = this.api('labels?per_page=100', { paginate: true });
        if (!current.some(label => label.name === spec.name)) throw error;
      }
    }
  }
}

async function snapshot(client, number, listed_pull) {
  const pull = listed_pull ?? client.get_pull(number);
  const decision = pull.state === 'open' ? evaluate_cla(pull, await client.get_signature(pull)) : null;
  return { pull, decision };
}

async function recheck(client, before) {
  const after = await snapshot(client, before.pull.number);
  const managed = value => MANAGED_LABELS.filter(name => value.pull.labels.some(label => label.name === name));
  if (JSON.stringify(managed(before)) !== JSON.stringify(managed(after))) {
    throw new Error('Managed CLA labels changed concurrently; retry with current state');
  }
  if (after.pull.state !== 'open' || after.pull.head.sha !== before.pull.head.sha ||
      after.pull.user.id !== before.pull.user.id || after.pull.user.login !== before.pull.user.login ||
      after.decision.signature.agreement !== before.decision.signature.agreement ||
      after.decision.label !== before.decision.label) {
    throw new Error('PR creator or signature changed; retry with current state');
  }
  return after;
}

/** Synchronize creator labels and remove obsolete names; dry-run is the default. */
export async function sync_cla_labels(client, pulls, { write = false } = {}) {
  const reports = [];
  const errors = [];
  if (write && client.repository !== REPOSITORY) throw new Error('Writes are restricted to the configured repository');
  if (write && pulls.length) client.ensure_labels();
  for (const item of pulls) {
    try {
      const listed_pull = item.head && item.user && Array.isArray(item.labels) ? item : undefined;
      let current = await snapshot(client, item.number, listed_pull);
      if (current.pull.state !== 'open') continue;
      const { label, reason } = current.decision;
      if (write) {
        for (const name of MANAGED_LABELS.filter(name => name !== label)) {
          if (!current.pull.labels.some(value => value.name === name)) continue;
          current = await recheck(client, current);
          try { client.remove_label(item.number, name); } catch (error) {
            if (client.get_pull(item.number).labels.some(value => value.name === name)) throw error;
          }
          current.pull.labels = current.pull.labels.filter(value => value.name !== name);
        }
        if (!current.pull.labels.some(value => value.name === label)) {
          current = await recheck(client, current);
          client.add_label(item.number, label);
        }
      }
      reports.push({ number: item.number, label, reason });
    } catch (error) {
      errors.push(`PR #${item.number}: ${error.message}`);
    }
  }
  return { reports, errors };
}

/** Status events trigger a fresh personal query; the aggregate result is never used. */
export function event_pulls(client, event_name, event) {
  if (event_name === 'pull_request_target') return [{ number: event.pull_request.number }];
  if (event_name === 'status') {
    if (event.context !== CLA_CONTEXT) return [];
    if (!/^[a-f0-9]{40}$/.test(event.sha)) throw new Error('Invalid status SHA');
    return client.pulls_for_sha(event.sha).filter(pull => pull.state === 'open' && pull.head.sha === event.sha);
  }
  if (!event_name || ['schedule', 'workflow_dispatch'].includes(event_name)) return client.list_pulls();
  throw new Error(`Unsupported event: ${event_name}`);
}

async function main() {
  const args = process.argv.slice(2);
  if (args.some((arg, index) => !['--write', '--dry-run', '--pr'].includes(arg) && args[index - 1] !== '--pr') ||
      (args.includes('--write') && args.includes('--dry-run'))) throw new Error('Use --write or --dry-run [--pr NUMBER]');
  const number = args.includes('--pr') ? Number(args[args.indexOf('--pr') + 1]) : undefined;
  if (number !== undefined && (!Number.isSafeInteger(number) || number <= 0 || args.includes('--write'))) {
    throw new Error('--pr requires a positive number and is only available in dry-run mode');
  }
  if (!process.env.CLA_ASSISTANT_TOKEN) throw new Error('Configure CLA_ASSISTANT_TOKEN before running author queries');
  const client = new CLAClient(process.env.GH_REPO ?? REPOSITORY);
  const event = process.env.GITHUB_EVENT_PATH ? JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8')) : {};
  const pulls = number ? [{ number }] : event_pulls(client, process.env.GITHUB_EVENT_NAME, event);
  const result = await sync_cla_labels(client, pulls, { write: args.includes('--write') });
  const lines = result.reports.map(report => `PR #${report.number}: ${report.label} — ${report.reason}`);
  console.log(`${args.includes('--write') ? 'Creator label synchronization' : 'Read-only preview'}\n${lines.join('\n')}`);
  for (const error of result.errors) console.error(error);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY,
      `## PR creator CLA labels\n\n${lines.map(line => `- ${line}`).join('\n')}\n\nErrors: ${result.errors.length}\n`);
  }
  if (result.errors.length) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
