#!/bin/bash
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
# Rank the nodes of a shared pool by MEASURED idleness, so a launch picks the
# least-contended node instead of walking a hand-written priority list.
#
#   pool_rank.sh rank [COUNT]   print the COUNT (default 1) idlest admissible
#                               nodes, one per line, idlest first; exit 3 if
#                               fewer than COUNT are admissible
#   pool_rank.sh report         print the full table (admissible and not) with
#                               the reason each excluded node was excluded
#   pool_rank.sh sample         append one observation per node to the ledger
#                               and print nothing (for a periodic sampler)
#   pool_rank.sh self-test      parser positive controls, no network
#
# Why measured and not ordered: a fixed list sends every job to the same node,
# which is the one most likely to still be warm from the last job; and a list
# written yesterday encodes yesterday's occupancy. The ranking below uses, in
# order of authority:
#
#   1. admission - node_guard.sh's verdict, unchanged. A busy node is never
#      ranked, however idle it looks on average. Occupancy is a veto, not a
#      score, because sharing a die is the failure we are avoiding.
#   2. idle fraction over the recent window(s) - from this script's own
#      timestamped ledger, which every call appends to. Two windows are
#      reported: a short one (default 5 min) and a long one (default 30 min).
#      A node idle across both outranks one that just became free.
#   3. load average (5 and 15 minute) - the kernel's own decaying windows.
#      This is what makes the policy well defined on the FIRST call, before
#      any ledger history exists: loadavg already integrates the recent past,
#      for free, with no sampler to leave running. A node with no history is
#      ranked on loadavg alone and flagged "no-history" so the operator can
#      see that its window evidence is the kernel's and not ours.
#   4. resident device memory - sum of per-die used HBM. Non-zero on an
#      otherwise admissible node means a previous job released its processes
#      but not its memory; prefer a node that is actually clean.
#
# Keeping a sampler running just to fill the ledger would itself be residue on
# a shared machine, so this script does not start one: it samples on every
# call, and the natural call rate of a working session (every launch, every
# status check) fills the short window on its own. If a long window matters
# for a specific decision, call `sample` a few times a few minutes apart and
# then `rank` - deliberately, and with nothing left behind.
#
# The pool is NEVER hard-coded here. Pass it in:
#   POOL_NODES="host-a host-b"    space-separated, or
#   POOL_FILE=/path/to/pool       one host per line, '#' comments allowed
#   POOL_EXCLUDE="host-c"         hosts to skip (reserved, or another tenant's)
# Other knobs:
#   POOL_LEDGER    observation log (default ~/.cache/pool_rank/ledger.tsv)
#   SHORT_WINDOW_S long-tail of the short window, seconds (default 300)
#   LONG_WINDOW_S  long window, seconds (default 1800)
#   GUARD          path to node_guard.sh (default: next to this script)
#   SSH_OPTS       ssh options for the probe

set -u
MODE=${1:-report}
WANT=${2:-1}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GUARD=${GUARD:-$HERE/node_guard.sh}
LEDGER=${POOL_LEDGER:-$HOME/.cache/pool_rank/ledger.tsv}
SHORT_WINDOW_S=${SHORT_WINDOW_S:-300}
LONG_WINDOW_S=${LONG_WINDOW_S:-1800}
SSH_OPTS=${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new}

pool_list() {
    local raw=""
    if [ -n "${POOL_NODES:-}" ]; then
        raw=${POOL_NODES}
    elif [ -n "${POOL_FILE:-}" ] && [ -r "${POOL_FILE}" ]; then
        raw=$(sed -e 's/#.*//' "${POOL_FILE}")
    else
        echo "pool_rank: set POOL_NODES or POOL_FILE" >&2
        return 1
    fi
    local host keep
    for host in ${raw}; do
        keep=1
        for skip in ${POOL_EXCLUDE:-}; do
            [ "${host}" = "${skip}" ] && keep=0
        done
        if [ "${keep}" = 1 ]; then
            echo "${host}"
        fi
    done
    # Explicit: with `[ ... ] && echo` as the loop's last statement, excluding
    # the LAST pool entry made the function exit 1 and a caller read a
    # correctly resolved pool as a failure.
    return 0
}

# The remote half of a probe, as a template with placeholders. Kept as a
# quoted-literal heredoc and substituted at call time: building it inline in
# the ssh argument needs three levels of quoting around the awk programs, and
# a probe that misquotes itself reports a node as unreadable, which this
# script is required to treat as busy - a quoting slip would silently shrink
# the pool. Plain POSIX sh only: the remote login shell is not ours to choose,
# so no process substitution here.
IFS= read -r -d '' REMOTE_PROBE <<'EOS' || true
detail=$(bash "__GUARD__" pre 2>/dev/null | tail -1)
[ -z "$detail" ] && detail=UNKNOWN
# First word only. The guard's verdict line carries human-readable detail
# after the verdict ("BUSY  dies=8 maxHBM=52256MB"), and a multi-word state
# shifts every later column by one - which silently turned the ranking keys
# into text. Keep the verdict atomic and the detail in its own field.
state=${detail%% *}
detail=$(printf '%s' "$detail" | tr '\t' ' ')
l5=$(awk '{print $2}' /proc/loadavg 2>/dev/null)
l15=$(awk '{print $3}' /proc/loadavg 2>/dev/null)
hbm=$(npu-smi info 2>/dev/null \
      | grep -oE '[0-9]+ */ *__HBM__' | grep -oE '^[0-9]+' \
      | awk '{s+=$1} END {print s+0}')
printf '%s\t%s\t%s\t%s\t%s\n' "$state" "$detail" "${l5:-999}" \
    "${l15:-999}" "${hbm:-999999}"
EOS

# One remote probe per node. Prints: STATE<TAB>load5<TAB>load15<TAB>hbm_sum
# STATE is the guard's verdict word, or UNREACHABLE (ranked as busy).
probe() {
    local host=$1 out remote
    remote=${REMOTE_PROBE//__GUARD__/${GUARD_REMOTE:-$GUARD}}
    remote=${remote//__HBM__/${HBM_TOTAL_MB:-65536}}
    out=$(ssh -n ${SSH_OPTS} "${host}" "${remote}" 2>/dev/null | tail -1)
    case "${out}" in
    *"	"*) : ;;                                  # a tab means we parsed a row
    *) out=$(printf 'UNREACHABLE\tno answer from probe\t999\t999\t999999') ;;
    esac
    printf '%s\n' "${out}"
}

record() {  # host state load5 load15 hbm -> ledger
    mkdir -p "$(dirname "${LEDGER}")" 2>/dev/null
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$1" "$2" "$3" "$4" "$5" \
        >> "${LEDGER}"
}

# Idle fraction for a host over the last N seconds of ledger history.
# Prints "fraction<TAB>samples"; the fraction is 0 when there is no history,
# and the sample count is what tells the caller whether to believe it.
idle_fraction() {
    local host=$1 window=$2
    [ -r "${LEDGER}" ] || { printf '0\t0\n'; return; }
    awk -v host="${host}" -v cutoff="$(( $(date +%s) - window ))" '
        $2 == host && $1 >= cutoff { total++; if ($3 ~ /^IDLE/) idle++ }
        END { if (total) printf "%.2f\t%d\n", idle / total, total;
              else printf "0\t0\n" }
    ' "${LEDGER}"
}

# -> TAB-separated rows: host state detail load5 load15 hbm sfrac sn lfrac ln
# Everything downstream parses with -F'\t': `detail` is free text from the
# guard and `state` must stay a single token, so whitespace cannot be the
# field separator anywhere in this pipeline.
collect() {
    local host row state detail l5 l15 hbm
    while read -r host; do
        [ -z "${host}" ] && continue
        row=$(probe "${host}")
        state=$(printf '%s' "${row}" | cut -f1)
        detail=$(printf '%s' "${row}" | cut -f2)
        l5=$(printf '%s' "${row}" | cut -f3)
        l15=$(printf '%s' "${row}" | cut -f4)
        hbm=$(printf '%s' "${row}" | cut -f5)
        record "${host}" "${state}" "${l5}" "${l15}" "${hbm}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${host}" "${state}" \
            "${detail}" "${l5}" "${l15}" "${hbm}" \
            "$(idle_fraction "${host}" "${SHORT_WINDOW_S}")" \
            "$(idle_fraction "${host}" "${LONG_WINDOW_S}")"
    done < <(pool_list)
}

# Sort admissible rows idlest-first. Keys, in order: short-window idle
# fraction (desc, only when we have more than the sample we just took),
# long-window idle fraction (desc), load5 (asc), load15 (asc), hbm (asc).
rank_rows() {
    awk -F'\t' '
        $2 ~ /^IDLE/ {
            # fields: host state detail l5 l15 hbm sfrac sn lfrac ln
            sfrac = ($8 > 1) ? $7 : -1   # -1 = no history beyond this probe
            lfrac = ($10 > 1) ? $9 : -1
            printf "%.2f\t%.2f\t%.2f\t%.2f\t%d\t%s\t%s\n",
                   -sfrac, -lfrac, $4, $5, $6, $1, ($8 > 1 ? "" : "no-history")
        }
    ' | sort -k1,1g -k2,2g -k3,3g -k4,4g -k5,5n
}

case "${MODE}" in
pool)
    # Resolve and print the pool without probing anything. Worth having as
    # its own mode: the pool arrives from the environment, so "which hosts
    # did this launcher think it was allowed to use" is a question with a
    # wrong answer available, and the answer should be inspectable offline.
    pool_list
    ;;
sample)
    collect > /dev/null
    ;;
rank)
    rows=$(collect)
    picked=$(printf '%s\n' "${rows}" | rank_rows | cut -f6 | head -n "${WANT}")
    count=$(printf '%s' "${picked}" | grep -c . )
    printf '%s\n' "${picked}"
    [ "${count}" -ge "${WANT}" ] || {
        echo "pool_rank: wanted ${WANT} idle node(s), found ${count}" >&2
        exit 3
    }
    ;;
report)
    rows=$(collect)
    printf '%-18s %-12s %6s %6s %9s %8s %8s  %s\n' NODE STATE LOAD5 LOAD15 \
        HBM_MB "IDLE${SHORT_WINDOW_S}" "IDLE${LONG_WINDOW_S}" DETAIL
    printf '%s\n' "${rows}" | while IFS=$'\t' read -r h s d l5 l15 hbm sf sn lf ln; do
        [ -z "${h}" ] && continue
        printf '%-18s %-12s %6s %6s %9s %5s/%-2s %5s/%-2s  %s\n' \
            "${h}" "${s}" "${l5}" "${l15}" "${hbm}" "${sf}" "${sn}" \
            "${lf}" "${ln}" "${d}"
    done
    echo "-- idlest admissible first --"
    printf '%s\n' "${rows}" | rank_rows \
        | awk -F'\t' '{printf "%s%s\n", $6, ($7 == "" ? "" : "  (" $7 ")")}'
    ;;
self-test)
    # The ranking must be decidable with no ledger at all, must never rank a
    # busy node, and must prefer history over loadavg when history exists.
    fail=0
    # Fixtures carry a MULTI-WORD detail field on purpose: the guard's verdict
    # line reads "BUSY  dies=8 maxHBM=52256MB", and parsing these rows on
    # whitespace shifted every ranking key one field right, so the sort
    # compared text and the ranking was decided by nothing.
    no_history=$(printf '%s\n' \
        $'h-busy\tBUSY\tdies=8 maxHBM=52256MB\t0.00\t0.00\t0\t0\t0\t0\t0' \
        $'h-loaded\tIDLE\tmaxHBM=3154MB\t9.00\t9.00\t0\t0\t1\t0\t1' \
        $'h-quiet\tIDLE\tmaxHBM=3154MB\t0.10\t0.10\t0\t0\t1\t0\t1' \
        | rank_rows | cut -f6 | head -1)
    [ "${no_history}" = "h-quiet" ] || { echo "FAIL loadavg fallback: got '${no_history}'"; fail=1; }
    busy_only=$(printf '%s\n' \
        $'h-busy\tBUSY\tdies=8 maxHBM=52256MB\t0.00\t0.00\t0\t0\t9\t0\t9' \
        | rank_rows | wc -l)
    [ "${busy_only}" -eq 0 ] || { echo "FAIL busy node ranked"; fail=1; }
    # h-flaky has been idle 30% of the short window, h-steady 90%: history
    # decides even though h-flaky is momentarily quieter by loadavg.
    with_history=$(printf '%s\n' \
        $'h-flaky\tIDLE\td\t0.10\t0.10\t0\t0.30\t9\t0.30\t9' \
        $'h-steady\tIDLE\td\t5.00\t5.00\t0\t0.90\t9\t0.90\t9' \
        | rank_rows | cut -f6 | head -1)
    [ "${with_history}" = "h-steady" ] || { echo "FAIL history precedence: got '${with_history}'"; fail=1; }
    # Residual device memory breaks a tie that every other key leaves equal.
    dirty=$(printf '%s\n' \
        $'h-dirty\tIDLE\td\t0.10\t0.10\t40000\t0\t1\t0\t1' \
        $'h-clean\tIDLE\td\t0.10\t0.10\t0\t0\t1\t0\t1' \
        | rank_rows | cut -f6 | head -1)
    [ "${dirty}" = "h-clean" ] || { echo "FAIL hbm tiebreak: got '${dirty}'"; fail=1; }
    # A loadavg key must be read as a number, not as text: with the fields
    # shifted the sort saw "maxHBM=..." in the load column and every node
    # compared equal. Pin it by making loadavg the ONLY distinguishing key.
    numeric=$(printf '%s\n' \
        $'h-a\tIDLE\tmaxHBM=3154MB\t12.00\t12.00\t0\t0\t1\t0\t1' \
        $'h-b\tIDLE\tmaxHBM=3154MB\t2.00\t2.00\t0\t0\t1\t0\t1' \
        | rank_rows | cut -f6 | tr '\n' ' ')
    [ "${numeric}" = "h-b h-a " ] || { echo "FAIL numeric load keys: got '${numeric}'"; fail=1; }
    [ "${fail}" = 0 ] && echo "SELF-TEST PASS: ranking is decidable without history, vetoes busy nodes, prefers history over loadavg"
    exit "${fail}"
    ;;
*)
    echo "usage: pool_rank.sh [rank [COUNT] | report | sample | pool | self-test]" >&2
    exit 1
    ;;
esac
