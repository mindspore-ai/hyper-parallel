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
"""Apply the accepted sampler lifecycle fix only to verified dependency content."""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "f1b6155f9ae25678705cba9fe639283c48e15882c1e4aa722ae06dce68991d54"
PATCHED_SHA256 = "d5f93c0701d5671712da47940edd26769f577cb2df09a3e9606701d5a1153a40"
VERL_ORIGINAL_SHA256 = "3cdbf8bf456d9a269ba1cf3382d4848a03b0f62ef7a506d336f37914da77a03e"
VERL_PATCHED_SHA256 = "be7a4bea9fbf9b763878572d54117e902a7682b2876b93b7641ac15448254830"


def apply_patch(target: Path) -> dict[str, str]:
    """Reject unknown versions rather than silently applying a different sampler change."""
    content = target.read_bytes()
    before = hashlib.sha256(content).hexdigest()
    accepted = {ORIGINAL_SHA256: PATCHED_SHA256, VERL_ORIGINAL_SHA256: VERL_PATCHED_SHA256}
    if before in accepted.values():
        return {"before": before, "after": before, "status": "already_applied"}
    if before not in accepted:
        raise ValueError("Unknown sampler source; verify the dependency before applying this patch")
    anchor = b"    torch.npu.current_stream().wait_stream(global_stream())\n"
    if content.count(anchor) != 1:
        raise ValueError("Expected one cross-stream sampling site")
    patched = content.replace(anchor, anchor + b"    q.record_stream(torch.npu.current_stream())\n")
    if hashlib.sha256(patched).hexdigest() != accepted[before]:
        raise ValueError("Patch does not match the accepted source digest")
    target.write_bytes(patched)
    return {"before": before, "after": hashlib.sha256(target.read_bytes()).hexdigest(), "status": "applied"}


def main() -> None:
    """Record the dependency identity alongside the experiment outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path("/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py")
    )
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    record = apply_patch(args.target)
    args.record.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
