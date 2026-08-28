# Simple Model Training Example

FSDP training with HyperParallel Graph Mode.

## Quick Start

```bash
# Single card
python train.py --config config.yaml

# 8 cards
torchrun --nproc_per_node=8 train.py --config config.yaml
```

## Files

- `train.py` - Training entry script
- `config.yaml` - Training and sharding configuration

## Configuration

```yaml
parallel:
  enable_overlap: true

sharding:
  fsdp:
    enabled: true
    patterns:
      - pattern: "*"
```

## Code

```python
from hyper_parallel.compile import GraphTrainer, ParallelConfig, ShardingPlan

# Model
model = DummyModel(vocab_size, dim)

# Sharding
sharding_plan = ShardingPlan()
sharding_plan.fsdp_wrap_pattern("*")

# Trainer
trainer = GraphTrainer(
    model=model,
    train_fn=train_fn,
    parallel_config=ParallelConfig(enable_overlap=True),
    sharding_plan=sharding_plan,
)

# Train
trainer.train(dataloader, max_steps=1000)
```

## Limitations

- FSDP only (TP/EP/PP planned)
- Parameters must have dim 0 divisible by world_size