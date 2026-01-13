# Copyright 2025 Huawei Technologies Co., Ltd
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
"""hsdp common utils"""
from typing import List, Tuple
import re
import time
import mindspore as ms
from mindspore import nn
from hyper_parallel import hsdp_wait_grad_handle

hsdp_network_ckpt_path: str = "hsdp_network.ckpt"

def extract_metrics_from_log(log_path: str) -> List[Tuple]:
    """extract loss from log file"""
    # expected log format
    pattern = r"step: (\d+), loss: ([\d.]+)"
    # [(loss_0, ), (loss_1, )...(loss_n, )]
    indicator_list: List[Tuple] = []
    with open(log_path, mode="r", encoding="utf-8") as log:
        for line in log:
            effect_log_start_idx: int = line.find("step:")
            if effect_log_start_idx != -1:
                line = line[effect_log_start_idx:]
            else:
                continue
            match = re.match(pattern, line.strip("."))
            if match:
                loss = float(match.group(2))
                indicator_list.append((loss,))
    return indicator_list

class ErrorComparator:
    """compare baseline loss with hsdp loss"""
    def __init__(self, baseline_log_path: str, hsdp_case_log_path: str):
        self.baseline_path = baseline_log_path
        self.hsdp_case_path = hsdp_case_log_path
        self.baseline_metrics = extract_metrics_from_log(self.baseline_path)
        self.hsdp_case_metrics = extract_metrics_from_log(self.hsdp_case_path)
        assert self.baseline_metrics,\
            f"For HSDP precision case, {self.baseline_path} hasn't capture any metrics, Please check the log."
        assert len(self.baseline_metrics) == len(self.hsdp_case_metrics),\
            f"For HSDP precision case, get {len(self.baseline_metrics)} steps from {self.baseline_path},\
                but get {len(self.hsdp_case_metrics)} steps from {self.hsdp_case_path}"

    def get_relative_absolute_error(self) -> float:
        baseline_loss_list = self._extract_loss_from_metrics(self.baseline_metrics)
        hsdp_case_loss_list = self._extract_loss_from_metrics(self.hsdp_case_metrics)
        total_step: int = len(baseline_loss_list)
        total_relative_abs_loss: float = 0.0
        for baseline_loss, hsdp_case_loss in zip(baseline_loss_list, hsdp_case_loss_list):
            total_relative_abs_loss += abs(hsdp_case_loss - baseline_loss) / baseline_loss
        return total_relative_abs_loss / total_step

    def get_first_step_rel_abs_error(self) -> bool:
        baseline_first_step_metric: Tuple = self.baseline_metrics[0]
        case_first_step_metric: Tuple = self.hsdp_case_metrics[0]
        rel_abs_err = abs(baseline_first_step_metric[0] - case_first_step_metric[0]) / baseline_first_step_metric[0]
        return rel_abs_err

    def get_rel_abs_error_of_steps(self) -> List[float]:
        baseline_loss_list = self._extract_loss_from_metrics(self.baseline_metrics)
        hsdp_case_loss_list = self._extract_loss_from_metrics(self.hsdp_case_metrics)
        errors: List[float] = []
        for baseline_loss, hsdp_case_loss in zip(baseline_loss_list, hsdp_case_loss_list):
            errors.append(abs(hsdp_case_loss - baseline_loss) / baseline_loss)
        return errors

    def _extract_loss_from_metrics(self, metrics_list: List[Tuple]):
        if metrics_list is None:
            return []
        loss_list: List[float] = [per_step_metric[0] for per_step_metric in metrics_list]
        return loss_list

loss_fn = nn.MSELoss()
def get_forward_fn(net):
    """return forward func with data and label input"""
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    return forward_fn

def train_with_data_label(net, data, label, comm_async=True, train_steps=10):
    """train net with data and label"""
    train_steps = max(train_steps, 1)
    train_steps = train_steps + 1
    optimizer = nn.Adam(net.trainable_params(), 0.01)
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)
    cost_time = 0
    for i in range(train_steps):
        if i != 0:
            start_time = time.time()
        _, grads = grad_fn(data, label)
        if comm_async:
            hsdp_wait_grad_handle()
        if i != 0:
            end_time = time.time()
            cost_time = end_time - start_time + cost_time
        optimizer(grads)
    return cost_time
