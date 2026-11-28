#!/usr/bin/env python3
"""
Example: Using MoE-Specific Rules in parallel-strategy-analyzer

Demonstrates how PSA applies MoE rules for Qwen3-30B-A3B
"""

# Model Configuration (Qwen3-30B-A3B)
model_config = {
    "name": "Qwen3-30B-A3B",
    "num_layers": 48,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "intermediate_size": 6144,

    # MoE-specific
    "is_moe": True,
    "num_experts": 128,
    "expert_hidden_size": 768,
    "num_experts_per_tok": 8,  # top_k
}

# Hardware Configuration
hardware_config = {
    "device_type": "Ascend910B",
    "device_count": 64,
    "memory_per_device_gb": 64,
    "interconnect": "HCCS",
    "bandwidth_gbps": 12.5,  # Effective
}

# Baseline Profiling (from moe-profiler)
baseline_profiling = {
    "load_imbalance_factor": 1.68,
    "alltoall_ratio": 0.32,
    "baseline_throughput": 94000,  # tokens/s
}

print("=" * 60)
print("MoE Strategy Analysis for Qwen3-30B-A3B")
print("=" * 60)

# Step 1: Check if MoE model
if model_config["is_moe"]:
    print("\n✓ MoE model detected, loading moe-strategy-rules.md")
    print(f"  - Experts: {model_config['num_experts']}")
    print(f"  - Top-K: {model_config['num_experts_per_tok']}")
    print(f"  - Expert hidden: {model_config['expert_hidden_size']}")

# Step 2: Apply Rule 1 - EP Size Selection
print("\n[Rule 1] EP Size Selection")
print("-" * 60)

num_experts = model_config["num_experts"]
device_count = hardware_config["device_count"]

# Case: Many experts (128 > 64 * 2? No)
# Case: Moderate experts (128 ≈ 64 * 2)
if num_experts > device_count * 2:
    ep_size = min(device_count, num_experts // 2)
    reason = "Many experts, prefer sharding"
else:
    ep_size = min(8, device_count // 2)  # Conservative for communication
    reason = "Moderate experts, balance shard+comm"

# Adjust for divisibility
while num_experts % ep_size != 0:
    ep_size -= 1

print(f"  Initial EP size: {ep_size}")
print(f"  Reason: {reason}")
print(f"  Experts per rank: {num_experts // ep_size}")

# Step 3: Apply Rule 2 - EP + TP Hybrid
print("\n[Rule 2] EP + TP Hybrid")
print("-" * 60)

hidden_size = model_config["hidden_size"]
remaining_devices = device_count // ep_size

if hidden_size >= 4096 and remaining_devices >= 2:
    tp_size = min(4, remaining_devices)
    use_hybrid = True
    print(f"  ✓ Use 2D mesh: hidden={hidden_size} ≥ 4096")
else:
    tp_size = 2 if hidden_size >= 2048 else 1
    use_hybrid = False
    print(f"  - Small hidden ({hidden_size}), minimal TP")

print(f"  TP size: {tp_size}")
print(f"  Remaining for DP: {device_count // (ep_size * tp_size)}")

# Step 4: Apply Rule 3 - Capacity Factor
print("\n[Rule 3] Capacity Factor Selection")
print("-" * 60)

load_imbalance = baseline_profiling["load_imbalance_factor"]

if load_imbalance < 1.3:
    capacity_factor = 1.0
    reason = "Mild imbalance"
elif load_imbalance < 1.6:
    capacity_factor = 1.25
    reason = "Moderate imbalance"
elif load_imbalance < 2.0:
    capacity_factor = 1.5
    reason = "High imbalance"
else:
    capacity_factor = 2.0
    reason = "Severe imbalance"

print(f"  Load imbalance: {load_imbalance:.2f}")
print(f"  Capacity factor: {capacity_factor}")
print(f"  Reason: {reason}")
print(f"  Memory impact: +{int((capacity_factor - 1.0) * 100)}% activation")

# Step 5: Generate Final Config
print("\n[Final Configuration]")
print("=" * 60)

dp_size = device_count // (ep_size * tp_size)

final_config = {
    "ep_size": ep_size,
    "tp_size": tp_size,
    "dp_size": dp_size,
    "capacity_factor": capacity_factor,
    "expert_bias": load_imbalance > 1.5,
}

print(f"  EP: {final_config['ep_size']}")
print(f"  TP: {final_config['tp_size']}")
print(f"  DP: {final_config['dp_size']}")
print(f"  Capacity factor: {final_config['capacity_factor']}")
print(f"  Expert bias: {final_config['expert_bias']}")

# Step 6: Estimate Impact
print("\n[Estimated Impact]")
print("=" * 60)

# Memory estimation
experts_per_rank = num_experts // ep_size
expert_params_gb = experts_per_rank * model_config["expert_hidden_size"] * 4096 * 2 / (1024**3)
expert_activation_gb = (
    experts_per_rank *
    model_config["expert_hidden_size"] *
    capacity_factor *
    1024 / (1024**3)  # Assuming 1024 avg tokens per rank
)
alltoall_buffer_gb = 2 * 4096 * hidden_size * 2 / (1024**3)

total_moe_memory = expert_params_gb + expert_activation_gb + alltoall_buffer_gb

print(f"  Expert params: {expert_params_gb:.1f} GB")
print(f"  Expert activation: {expert_activation_gb:.1f} GB")
print(f"  All-to-All buffer: {alltoall_buffer_gb:.1f} GB")
print(f"  Total MoE memory: {total_moe_memory:.1f} GB")

# Communication estimation
single_alltoall_ms = (
    4096 * hidden_size * 2 *
    (ep_size - 1) / ep_size *
    2 / (hardware_config["bandwidth_gbps"] * 125)  # GB/s to ms
)
total_alltoall_ms = 4 * single_alltoall_ms  # 4 per layer
alltoall_ratio = total_alltoall_ms / 175  # Assuming 175ms step time

print(f"\n  Single All-to-All: {single_alltoall_ms:.2f} ms")
print(f"  Total per layer: {total_alltoall_ms:.2f} ms")
print(f"  All-to-All ratio: {alltoall_ratio:.2%}")

# Expected improvement
comm_improvement = (baseline_profiling["alltoall_ratio"] - alltoall_ratio) / baseline_profiling["alltoall_ratio"]
expected_throughput = baseline_profiling["baseline_throughput"] * (1 + comm_improvement * 0.5)

print(f"\n  Baseline throughput: {baseline_profiling['baseline_throughput']:,} tok/s")
print(f"  Communication improvement: {comm_improvement:.1%}")
print(f"  Expected throughput: {int(expected_throughput):,} tok/s")
print(f"  Expected improvement: +{int((expected_throughput / baseline_profiling['baseline_throughput'] - 1) * 100)}%")

print("\n" + "=" * 60)
print("✅ MoE strategy analysis complete")
print("=" * 60)
