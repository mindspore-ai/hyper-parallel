# Workflow 2: Base Class API Design

## Goal

Define or modify abstract methods in `platform/platform.py` to establish the cross-platform contract.

## When to Skip

Skip this step if:
- Only modifying internal implementation (no API change)
- Only fixing a bug in existing backend code
- Only adding tests

## Steps

### 2.1 Design the API Signature

Follow existing conventions in `platform/platform.py`:

```python
def new_method(self, param1: Type1, param2: Type2 = default) -> ReturnType:
    """Brief description.

    Args:
        param1: Description.
        param2: Description. Defaults to default.

    Returns:
        Description of return value.

    Raises:
        NotImplementedError: If backend does not support this operation.
    """
    raise NotImplementedError
```

**Conventions:**
- Use type hints on all parameters and return values
- Use Google-style docstrings
- Default implementation raises `NotImplementedError`
- For properties, use `@property` decorator
- For class-level attributes, define as class variables with type annotation

### 2.2 Choose API Category

Place the new method near related methods. Categories in `platform.py`:

| Category | Existing Methods (examples) |
|----------|---------------------------|
| Tensor/Module types | `Tensor`, `Parameter`, `Module`, `DTensorBase` |
| Collective ops | `all_gather_into_tensor()`, `all_reduce()`, `reduce_scatter_tensor()` |
| Differentiable ops | `differentiable_all_gather_concat()`, `differentiable_all_reduce()` |
| Process groups | `create_group()`, `split_group()`, `init_process_group()` |
| Streams/Events | `new_stream()`, `get_stream_context()`, `new_event()` |
| Memory/Tensor creation | `new_tensor()`, `new_zero_parameter()`, `empty_like()` |
| Module manipulation | `get_cells_and_names()`, `search_parameter_by_name()` |
| Device/RNG | `device()`, `manual_seed()`, `get_rng_state()` |
| Gradient | `set_grad_reduce_handle()`, `wait_grad_handle()`, `clip_grad_norm_()` |
| Checkpoint | `checkpoint`, `ckpt_wrapper()`, `async_save_on_cpu()` |
| Hooks | `register_forward_pre_hook()`, `register_full_backward_hook()` |

### 2.3 Consider Backward Compatibility

- Adding new methods with `raise NotImplementedError` is safe
- Modifying existing method signatures requires updating both backends
- Removing methods requires checking all callers first

### 2.4 Implement in platform.py

Add the method to the `Platform` class with proper docstring and `raise NotImplementedError`.

## Output

- Updated `platform/platform.py` with new/modified abstract method(s)
- Clear contract for backend implementations

## Next Step

Proceed to **[Workflow 3: Backend Implementation](./03-backend-implementation.md)**
