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
"""Real two-rank checkpoint save/resume regression before expensive MoE training."""
import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from rl.trainer.checkpoint import CheckpointResources, RLCheckpointManager
from rl.trainer.distributed import run_synchronized

class Loader:
    def __init__(self, rank):
        self.position = rank + 7
    def state_dict(self):
        return {"position": self.position}
    def load_state_dict(self, value):
        self.position = value["position"]

dist.init_process_group("gloo", timeout=timedelta(seconds=90))
rank = dist.get_rank()
torch.manual_seed(7)
model = torch.nn.Linear(4, 4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
model(torch.ones(2, 4)).sum().mul(rank + 1).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
model_state = deepcopy(model.state_dict())
optimizer_state = deepcopy(optimizer.state_dict())
loader = Loader(rank)
device_handle = SimpleNamespace(
    get_rng_state=lambda _device: torch.get_rng_state(),
    set_rng_state=lambda value, _device: torch.set_rng_state(value),
)
resources = CheckpointResources(
    state=SimpleNamespace(max_steps=2),
    model_registration=SimpleNamespace(hf_architecture="Qwen3ForCausalLM"),
    train_dataloader=loader, device=torch.device("cpu"), device_handle=device_handle,
    actor=SimpleNamespace(actor_model=model, optimizer=optimizer, lr_scheduler=None),
    critic=None, tokenizer=None,
)
root = Path("/tmp/rl_only_checkpoint_smoke")
manager = RLCheckpointManager(resources, {"output_dir": str(root), "load_path": str(root / "step_2")},
                              {}, run_synchronized)
state = SimpleNamespace(global_step=2, epoch=1, consumed_samples=32, consumed_tokens=64)
rng = torch.get_rng_state().clone()
manager.save(state)
manager.validate_resume()
with torch.no_grad():
    for parameter in model.parameters():
        parameter.zero_()
for item in optimizer.state.values():
    for tensor in item.values():
        if isinstance(tensor, torch.Tensor):
            tensor.zero_()
loader.position = -1
manager.begin(state)
for key, expected in model_state.items():
    torch.testing.assert_close(model.state_dict()[key], expected, rtol=1e-6, atol=1e-7)
for key, expected in optimizer_state["state"].items():
    for name, value in expected.items():
        actual = optimizer.state_dict()["state"][key][name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        else:
            assert actual == value
assert loader.position == rank + 7
assert torch.equal(torch.get_rng_state(), rng)
receipts = [None] * dist.get_world_size()
dist.all_gather_object(receipts, {"rank": rank, "passed": True, "optimizer_moment": float(
    optimizer.state_dict()["state"][0]["exp_avg"].abs().max())})
if rank == 0:
    assert receipts[0]["optimizer_moment"] != receipts[1]["optimizer_moment"]
    (root / "passed.json").write_text(json.dumps({"world_size": 2, "passed": True, "ranks": receipts}, indent=2))
    print("Two-rank sequential save and full model/optimizer/RNG/dataloader resume passed.", flush=True)
dist.destroy_process_group()
