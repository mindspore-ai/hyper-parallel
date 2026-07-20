// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

function extractFunction(source, functionName) {
  const marker = `function ${functionName}(`;
  const start = source.indexOf(marker);
  assert(start >= 0, `Missing function in memory_analysis.html: ${functionName}`);
  const bodyStart = source.indexOf("{", start + marker.length);
  assert(bodyStart >= 0, `Missing function body: ${functionName}`);

  let depth = 0;
  let quote = null;
  let escaped = false;
  let lineComment = false;
  let blockComment = false;
  for (let index = bodyStart; index < source.length; index += 1) {
    const character = source[index];
    const nextCharacter = source[index + 1];
    if (lineComment) {
      if (character === "\n") lineComment = false;
      continue;
    }
    if (blockComment) {
      if (character === "*" && nextCharacter === "/") {
        blockComment = false;
        index += 1;
      }
      continue;
    }
    if (quote !== null) {
      if (escaped) {
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === quote) {
        quote = null;
      }
      continue;
    }
    if (character === "/" && nextCharacter === "/") {
      lineComment = true;
      index += 1;
      continue;
    }
    if (character === "/" && nextCharacter === "*") {
      blockComment = true;
      index += 1;
      continue;
    }
    if (character === "\"" || character === "'" || character === "`") {
      quote = character;
      continue;
    }
    if (character === "{") depth += 1;
    if (character === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(start, index + 1);
    }
  }
  throw new Error(`Unterminated function body: ${functionName}`);
}

function encodeCsvRow(values) {
  return values.map((value) => {
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }).join(",");
}

const repositoryRoot = path.resolve(__dirname, "../../../..");
const viewerSource = fs.readFileSync(
  path.join(repositoryRoot, "memory_analysis.html"),
  "utf8"
);
const constantsStart = viewerSource.indexOf("const PERSISTENT_END_TIMESTAMP");
const constantsEnd = viewerSource.indexOf("const DEFAULT_VISIBLE_HEADERS", constantsStart);
assert(constantsStart >= 0 && constantsEnd > constantsStart, "Missing memory constants");
const constantsSource = viewerSource.slice(constantsStart, constantsEnd);
const functionNames = [
  "isTrackedMemoryPool",
  "parseCsvLine",
  "parseCsv",
  "extractUsers",
  "arrayMax",
  "parsePythonStack",
  "buildSnapshot",
  "buildSeries",
  "computeBounds",
  "analyzeRows",
  "rowsAliveAtTimePoint",
];
const functionSource = functionNames
  .map((functionName) => extractFunction(viewerSource, functionName))
  .join("\n");
const buildApi = new Function(
  `${constantsSource}\n`
  + "const state = { rows: [], fullRangeX: [0, 0] };\n"
  + `${functionSource}\n`
  + "return { parseCsv, analyzeRows, buildSnapshot, rowsAliveAtTimePoint, state };"
);
const api = buildApi();

const headers = [
  "start_time_stamp", "end_time_stamp", "device_addr", "stream_id",
  "pool_type", "size", "actual_used_memory", "actual_peak_memory",
  "file_name", "line_num", "type", "producer_task", "task_name",
  "node_name", "graph_name", "user_tasks", "last_user_task",
  "python_stack", "is_persistent", "is_small",
];
const persistentNodeName = 'aten."quoted",op';
const csvText = [
  encodeCsvRow(headers),
  encodeCsvRow([
    0, "9223372036854775807", "0xf00000000000", 7,
    "FakeTensorLogicalMemoryPool", 10, 10, 10, "/tmp/a,b.py", 7,
    "Activation", 0, "forward", persistentNodeName, "Model", "{1}", 1,
    "File:/tmp/a,b.py;Line:7;Function:f", 1, 1,
  ]),
  encodeCsvRow([
    1, 3, "0xf00000000200", 2, "FakeTensorLogicalMemoryPool", 20,
    30, 30, "/tmp/model.py", 8, "Activation", 1, "forward",
    "aten.add.Tensor", "Model", "{2}", 2,
    "File:/tmp/model.py;Line:8;Function:g", 0, 1,
  ]),
].join("\n");

const parsed = api.parseCsv(csvText);
assert.strictEqual(parsed.rows.length, 2);
assert.strictEqual(parsed.rows[0].node_name, persistentNodeName);
assert.strictEqual(parsed.rows[0].file_name, "/tmp/a,b.py");

const analysis = api.analyzeRows(parsed.rows);
assert(analysis.real.points.length > 0, "Fake pool must produce a real series");
assert.deepStrictEqual(analysis.real.peak, { time: 1, value: 30 });

const snapshot = api.buildSnapshot(parsed.rows);
const events = snapshot.device_traces[0];
assert.strictEqual(events.length, 4);
assert(events.some((event) => event.action === "alloc" && event.stream === 7));
assert(events.some((event) => event.action === "alloc" && event.stream === 2));
assert.strictEqual(events.filter((event) => event.action === "free_requested").length, 1);
assert.strictEqual(events.filter((event) => event.action === "free_completed").length, 1);
assert.strictEqual(snapshot.segments.length, 1);
assert.strictEqual(snapshot.segments[0].blocks.length, 1);
assert.strictEqual(snapshot.segments[0].blocks[0].frames[0].filename, "/tmp/a,b.py");

api.state.rows = parsed.rows;
api.state.fullRangeX = analysis.xRange;
const aliveBeforeEnd = api.rowsAliveAtTimePoint(2);
const aliveAtEnd = api.rowsAliveAtTimePoint(3);
assert.strictEqual(aliveBeforeEnd.length, 2);
assert.strictEqual(aliveAtEnd.length, 1);
assert.strictEqual(aliveAtEnd[0].node_name, persistentNodeName);

console.log("memory_analysis behavior contract passed");
