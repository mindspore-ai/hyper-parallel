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
const STATUS_CONTEXT = 'label-check';
const ACTIONS_APP_ID = 15368;

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
  get_checks(sha) {
    return this.api(`commits/${sha}/check-runs?check_name=label-check&filter=latest&per_page=100`,
      { paginate: true }).flatMap(page => page.check_runs);
  }
  get_statuses(sha) { return this.api(`commits/${sha}/statuses?per_page=100`); }
  set_status(sha, state) {
    this.api(`statuses/${sha}`, {
      method: 'POST', body: { state, context: STATUS_CONTEXT,
        description: state === 'success' ? 'CI pass label is valid' : 'Fresh CI pass and label check required' },
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
    value.labels.map(label => [label.name, label.event_id, label.expired])]);
}

function action_checks(client, sha) {
  return client.get_checks(sha).filter(check => check.name === STATUS_CONTEXT && check.app?.id === ACTIONS_APP_ID)
    .sort((left, right) => right.id - left.id);
}

function publish_status(client, sha, state, before_write = () => {}) {
  const latest = client.get_statuses(sha).find(status => status.context === STATUS_CONTEXT);
  if (latest?.state !== state || latest.creator?.login !== 'github-actions[bot]') {
    before_write();
    client.set_status(sha, state);
  }
}

function revalidate_group(client, current, now) {
  const head = current[0].pull.head.sha;
  const numbers = values => values.map(pull => pull.number).sort((left, right) => left - right).join(',');
  const sharing = client.list_pulls().filter(pull => pull.state === 'open' && pull.head.sha === head);
  if (numbers(sharing) !== numbers(current.map(entry => entry.pull))) {
    throw new Error('The PRs sharing this head changed during reconciliation');
  }
  for (const entry of current) {
    if (identity(snapshot(client, entry.pull.number, now)) !== identity(entry)) {
      throw new Error('PR or CI labels changed during reconciliation; retry on the next sweep');
    }
  }
}

/** Expire labels and publish a blocking status when CI is stale; default to read-only. */
export function reconcile_labels(client, { write = false, now = Date.now, only_pr, completed_head } = {}) {
  const reports = [];
  const errors = [];
  const pulls = only_pr ? [client.get_pull(only_pr)] : client.list_pulls();
  const open = pulls.filter(pull => pull.state === 'open');
  const heads = new Set(open.filter(pull => pull.head.sha === completed_head ||
    pull.labels.some(label => PASS_LABELS.includes(label.name))).map(pull => pull.head.sha));
  for (const head of heads) {
    const group = open.filter(pull => pull.head.sha === head);
    let current = [];
    const checks = new Map();
    try {
      current = group.map(pull => snapshot(client, pull.number, now, pull));
      reports.push(...current.map(entry => ({ number: entry.pull.number, labels: entry.labels })));
      if (!write) continue;
      const valid = current.every(entry => entry.labels.some(label => !label.expired));
      checks.set(head, action_checks(client, head));
      for (const entry of current) {
        const merge = entry.pull.merge_commit_sha;
        if (merge && !checks.has(merge)) {
          const merge_checks = action_checks(client, merge);
          // Creating a status on an otherwise unchecked test merge commit can
          // change which commit GitHub evaluates for all required checks.
          if (merge_checks.length) checks.set(merge, merge_checks);
        }
      }
      for (const [sha, runs] of checks) {
        const latest = runs[0];
        const state = valid && latest?.status === 'completed' && latest.conclusion === 'success' ?
          'success' : 'failure';
        publish_status(client, sha, state, () => revalidate_group(client, current, now));
      }
      for (const entry of current) {
        let previous = entry;
        for (const label of entry.labels.filter(item => item.expired)) {
          const fresh = snapshot(client, entry.pull.number, now);
          if (identity(fresh) !== identity(previous)) {
            throw new Error(`PR #${entry.pull.number} or its labels changed before removal`);
          }
          try {
            client.remove_label(entry.pull.number, label.name);
          } catch (error) {
            if (client.get_pull(entry.pull.number).labels.some(item => item.name === label.name)) throw error;
          }
          previous = { pull: fresh.pull, labels: fresh.labels.filter(item => item.name !== label.name) };
        }
      }
    } catch (error) {
      errors.push(`PRs ${group.map(pull => `#${pull.number}`).join(', ')}: ${error.message}`);
      if (write) {
        // When event history or check reads fail, a blocking status is safer
        // than leaving an older success usable. Never delete an uncertain label.
        try {
          const still_open = group.some(pull => {
            const fresh = client.get_pull(pull.number);
            return fresh.state === 'open' && fresh.head.sha === head;
          });
          if (still_open) {
            for (const sha of new Set([head, ...checks.keys()])) publish_status(client, sha, 'failure');
          }
        } catch (fallback_error) {
          errors.push(`Head ${head}: could not publish a blocking status: ${fallback_error.message}`);
        }
      }
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
