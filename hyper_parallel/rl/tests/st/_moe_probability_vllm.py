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
"""Score a fixed sampled sequence using a freshly loaded native vLLM service."""
import argparse
import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

import torch.distributed as dist
import yaml

from rl.trainer.build import build_trainer


def main() -> None:
    """Use prompt log probabilities to score exact IDs without regenerating the answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(Path(args.config).read_text())
    trace = json.loads(Path(args.trace).read_text())
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(config)
    try:
        trainer.rollout_engine.start_service()

        def score() -> None:
            if dist.get_rank() != 0:
                return
            payload = {
                "model": trainer.rollout_engine.inference_model_name,
                "prompt": trace["prompt_ids"] + trace["response_ids"],
                "max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "logprobs": 1, "echo": True,
            }
            endpoint = trainer.rollout_engine.inference_base_url.rstrip("/") + "/v1/completions"
            request = Request(endpoint, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
            (destination / "response.json").write_text(json.dumps(result, indent=2))
            values = result["choices"][0]["logprobs"]["token_logprobs"]
            start = len(trace["prompt_ids"])
            selected = values[start:start + len(trace["response_ids"])]
            if len(selected) != len(trace["response_ids"]) or any(value is None for value in selected):
                raise ValueError("vLLM echo did not return the requested response-token probabilities")
            (destination / "result.json").write_text(json.dumps({"log_probs": selected}, indent=2))
            print("Fresh vLLM exact-token scoring completed", flush=True)

        trainer._run_rank_synchronized("fixed-token vLLM scoring", score)
        trainer.lifecycle.close()
    except BaseException as error:
        trainer.lifecycle.fail(error, "fixed-token-vllm-scoring")
        raise


if __name__ == "__main__":
    main()
