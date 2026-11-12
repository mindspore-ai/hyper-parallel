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

import { execFileSync } from 'node:child_process';
import { appendFileSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const PASS_LABELS = ['ci-pipeline-passed', 'ci-pipeline-passed-but-overtime'];
export const TTL_MS = 24 * 60 * 60 * 1000;
export const STATUS_CONTEXT = 'label-check';

/** Evaluate each currently attached label using its latest label transition. */
export function evaluate_labels(pull, events, now = Date.now()) {
  const latest = new Map();
  for (const event of [...events].sort((left, right) => left.id - right.id)) {
    if (PASS_LABELS.includes(event.label?.name) && ['labeled', 'unlabeled'].includes(event.event)) {
      latest.set(event.label.name, event);
    }
  }
  return PASS_LABELS.filter(name => pull.labels.some(label => label.name === name)).map(name => {
    const event = latest.get(name);
    const added = Date.parse(event?.created_at);
    if (event?.event !== 'labeled' || !Number.isFinite(added) || added > now) {
      throw new Error(`Cannot establish the current addition time for ${name} on PR #${pull.number}`);
    }
    return { name, event_id: event.id, added_at: event.created_at, expired: now - added >= TTL_MS };
  });
}

/** GitHub CLI adapter with explicit pagination and no shell interpolation. */
export class LabelClient {
  constructor(repository, run = execFileSync) {
    if (!/^[a-z\d_.-]+\/[a-z\d_.-]+$/i.test(repository)) throw new Error('Invalid GH_REPO');
    this.repository = repository;
    this.run = run;
  }

  api(path, { method = 'GET', body, paginate = false } = {}) {
    const args = ['api', '--hostname', 'github.com', '--method', method, `repos/${this.repository}/${path}`];
    if (paginate) args.push('--paginate', '--slurp');
    if (body) args.push('--input', '-');
    const output = this.run('gh', args, {
      encoding: 'utf8', input: body ? JSON.stringify(body) : undefined,
      maxBuffer: 32 * 1024 * 1024, timeout: 60000, stdio: ['pipe', 'pipe', 'pipe'],
    });
    return output.trim() ? JSON.parse(output) : null;
  }

  list_pulls() { return this.api('pulls?state=open&per_page=100', { paginate: true }).flat(); }
  get_pull(number) { return this.api(`pulls/${number}`); }
  get_events(number) { return this.api(`issues/${number}/events?per_page=100`, { paginate: true }).flat(); }
  get_statuses(sha) {
    return this.api(`commits/${sha}/status?per_page=100`,
      { paginate: true }).flatMap(page => page.statuses);
  }
  set_status(sha, state, number) {
    this.api(`statuses/${sha}`, {
      method: 'POST', body: {
        context: STATUS_CONTEXT, state,
        description: state === 'success' ? 'Every open PR using this commit has a fresh CI pass label.' :
          'An open PR using this commit needs a fresh CI pass label. Run CI again.',
        target_url: `https://github.com/${this.repository}/pull/${number}`,
      },
    });
  }
  remove_label(number, name) {
    this.api(`issues/${number}/labels/${encodeURIComponent(name)}`, { method: 'DELETE' });
  }
}

function snapshot(client, number, now, listed_pull) {
  const pull = listed_pull ?? client.get_pull(number);
  const events = pull.state === 'open' && pull.labels.some(label => PASS_LABELS.includes(label.name)) ?
    client.get_events(number) : [];
  return { pull, labels: pull.state === 'open' ? evaluate_labels(pull, events, now()) : [] };
}

function identity(value) {
  return JSON.stringify([value.pull.state, value.pull.head.sha, value.pull.merge_commit_sha,
    value.labels.map(label => [label.name, label.event_id])]);
}

function sharing_pulls(pulls, sha) {
  return pulls.filter(pull => pull.state === 'open' && [pull.head.sha, pull.merge_commit_sha].includes(sha));
}

function sync_statuses(client, current, now, pulls) {
  // Actions-owned check conclusions cannot be patched. A commit status with
  // the same required context must pass alongside that check (on either SHA).
  const shas = [...new Set([current.pull.head.sha, current.pull.merge_commit_sha].filter(Boolean))];
  const valid = current.labels.some(label => !label.expired);
  for (const sha of shas) {
    // Commit statuses are shared by every PR using the SHA. A fresh sibling PR
    // must not restore a status while another sibling still has expired CI.
    const peers = sharing_pulls(pulls, sha).filter(pull => pull.number !== current.pull.number);
    const snapshots = valid ? peers.map(pull => snapshot(client, pull.number, now)) : [];
    const state = valid && snapshots.every(value => value.labels.some(label => !label.expired)) ? 'success' : 'failure';
    const previous = client.get_statuses(sha).find(status => status.context === STATUS_CONTEXT);
    const status_time = now();
    if (state === 'success' && [current, ...snapshots].some(value =>
      !value.labels.some(label => status_time - Date.parse(label.added_at) < TTL_MS))) {
      throw new Error('CI labels expired while reading status; retry on the next sweep');
    }
    // GitHub limits each SHA/context to 1000 statuses; only publish transitions.
    if (previous?.state === state) continue;
    if (state === 'success') {
      const expected = [current.pull.number, ...peers.map(pull => pull.number)].sort((a, b) => a - b);
      const actual = sharing_pulls(client.list_pulls(), sha).map(pull => pull.number).sort((a, b) => a - b);
      if (JSON.stringify(expected) !== JSON.stringify(actual) || snapshots.some(value => {
        const latest = snapshot(client, value.pull.number, now);
        return identity(latest) !== identity(value) || !latest.labels.some(label => !label.expired);
      })) {
        throw new Error('PRs sharing this commit changed before publishing success; retry on the next sweep');
      }
    }
    const fresh = snapshot(client, current.pull.number, now);
    if (identity(fresh) !== identity(current) || fresh.labels.some(label => !label.expired) !== valid) {
      throw new Error('PR or CI labels changed before updating the status; retry on the next sweep');
    }
    const write_time = now();
    if (state === 'success' && [fresh, ...snapshots].some(value =>
      !value.labels.some(label => write_time - Date.parse(label.added_at) < TTL_MS))) {
      throw new Error('CI labels expired before publishing success; retry on the next sweep');
    }
    client.set_status(sha, state, current.pull.number);
  }
}

function block_after_error(client, pull, errors) {
  // A read error must not deliberately preserve an earlier success. If reads
  // fail too, try the last known SHAs; an API outage can still prevent revocation.
  let current = pull;
  try { current = client.get_pull(pull.number); } catch (error) {
    errors.push(`PR #${pull.number}: cannot refresh commits before blocking: ${error.message}`);
  }
  if (current.state !== 'open') return;
  for (const sha of new Set([current.head.sha, current.merge_commit_sha].filter(Boolean))) {
    let previous;
    try { previous = client.get_statuses(sha).find(status => status.context === STATUS_CONTEXT); } catch (error) {
      errors.push(`PR #${pull.number}: cannot read status on ${sha}: ${error.message}`);
    }
    if (previous?.state === 'failure') continue;
    try { client.set_status(sha, 'failure', pull.number); } catch (error) {
      errors.push(`PR #${pull.number}: cannot block ${sha}: ${error.message}`);
    }
  }
}

/** Expire labels and reconcile the required commit status; default to read-only. */
export function reconcile_labels(client, { write = false, now = Date.now, only_pr, completed_head } = {}) {
  const reports = [];
  const errors = [];
  const pulls = only_pr ? [client.get_pull(only_pr)] : client.list_pulls();
  for (const pull of pulls) {
    try {
      if (completed_head && pull.head.sha !== completed_head) continue;
      if (!completed_head && !pull.labels.some(label => PASS_LABELS.includes(label.name))) continue;
      let current = snapshot(client, pull.number, now, pull);
      if (current.pull.state !== 'open') continue;
      const expired = current.labels.filter(label => label.expired);
      reports.push({ number: pull.number, labels: current.labels });
      if (!write) continue;
      // GITHUB_TOKEN removals do not trigger the unlabeled workflow. A completed
      // label-check workflow also signals this reconciler to handle late successes.
      sync_statuses(client, current, now, pulls);
      for (const label of expired) {
        const fresh = snapshot(client, pull.number, now);
        if (identity(fresh) !== identity(current)) {
          throw new Error('PR or CI labels changed before removal; retry on the next sweep');
        }
        const candidate = fresh.labels.find(item => item.name === label.name);
        if (!candidate?.expired) continue;
        try {
          client.remove_label(pull.number, label.name);
        } catch (error) {
          if (client.get_pull(pull.number).labels.some(item => item.name === label.name)) throw error;
        }
        current = { pull: fresh.pull, labels: fresh.labels.filter(item => item.name !== label.name) };
      }
    } catch (error) {
      errors.push(`PR #${pull.number}: ${error.message}`);
      if (write) block_after_error(client, pull, errors);
    }
  }
  return { reports, errors };
}

function main() {
  const args = process.argv.slice(2);
  if (args.some((arg, index) => !['--check', '--write', '--dry-run', '--pr'].includes(arg) && args[index - 1] !== '--pr') ||
      args.filter(arg => ['--check', '--write', '--dry-run'].includes(arg)).length > 1) {
    throw new Error('Use --check --pr NUMBER, --dry-run [--pr NUMBER], or --write');
  }
  const only_pr = args.includes('--pr') ? Number(args[args.indexOf('--pr') + 1]) : undefined;
  if ((only_pr !== undefined && (!Number.isSafeInteger(only_pr) || only_pr <= 0)) ||
      (args.includes('--check') && !only_pr) || (args.includes('--write') && only_pr)) {
    throw new Error('--check needs a positive PR number; --write must sweep all open PRs');
  }
  const client = new LabelClient(process.env.GH_REPO ?? 'mindspore-ai/hyper-parallel');
  if (args.includes('--check')) {
    const current = snapshot(client, only_pr, Date.now);
    if (current.pull.state !== 'open' || !current.labels.some(label => !label.expired)) {
      throw new Error('Neither CI pass label is within its 24-hour validity period');
    }
    console.log('A CI pass label is present and less than 24 hours old.');
    return;
  }
  const completed_head = process.env.GITHUB_EVENT_NAME === 'workflow_run' ?
    JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8')).workflow_run.head_sha : undefined;
  if (completed_head !== undefined && !/^[a-f0-9]{40}$/.test(completed_head)) {
    throw new Error('Invalid completed workflow head SHA');
  }
  const result = reconcile_labels(client, { write: args.includes('--write'), only_pr, completed_head });
  const lines = result.reports.map(report => `PR #${report.number}: ${report.labels.map(label =>
    `${label.name}: ${label.expired ? 'expired' : 'valid'} (added ${label.added_at})`).join('; ') || 'no CI pass labels'}`);
  console.log(`${args.includes('--write') ? 'Expiry sweep' : 'Read-only evaluation'}\n${lines.join('\n')}`);
  for (const error of result.errors) console.error(error);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY,
      `## CI pass label expiry\n\n${lines.map(line => `- ${line}`).join('\n')}\n\nErrors: ${result.errors.length}\n`);
  }
  if (result.errors.length) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
