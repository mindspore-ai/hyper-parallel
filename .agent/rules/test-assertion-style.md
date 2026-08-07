---
name: test-assertion-style
description: Assertion style rules for all test files. Covers f-string usage, line length, and formatting conventions.
paths:
  - tests/**/*.py
---

# Test Assertion Style Rules

Common assertion style rules that apply to all test files (UT, ST, MindSpore, PyTorch).

## F-String Rules

**IMPORTANT: All assertion messages must follow these rules:**

1. **Use f-string format**: Always use f-strings for assertion messages
2. **Print both values**: When comparing two values, print both in the error message
3. **Line length limit**: Each line must not exceed 120 characters
4. **Use parentheses for line continuation**: Split long f-strings using parentheses
5. **No trailing whitespace**: Lines must not end with spaces

```python
# Correct - f-string with both values, properly formatted
assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3), \
    (f"Softmax data parallel test failed: "
     f"standalone={standalone_output.asnumpy()}, "
     f"parallel={parallel_output.asnumpy()}")

# Correct - layout comparison
assert dist_output.layout == expected_layout, \
    (f"Softmax data parallel layout mismatch: "
     f"expected={expected_layout}, got={dist_output.layout}")

# Wrong - f-string without interpolation (no variables)
assert np.allclose(a, b, 1e-3, 1e-3), \
    f"Test failed"

# Wrong - line exceeds 120 characters
assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3), \
    f"Softmax data parallel test failed: standalone={standalone_output.asnumpy()}, parallel={parallel_output.asnumpy()}"

# Wrong - using format() instead of f-string
assert dist_output.layout == expected_layout, \
    "Softmax layout mismatch: expected {}, got {}".format(expected_layout, dist_output.layout)

# Wrong - trailing whitespace (line ends with space)
assert np.allclose(a, b, 1e-3, 1e-3), \
    f"Test failed: a={a}, b={b}"  # <-- No space at end of line!
```

## Common Mistakes to Avoid

| Mistake | Correct Approach |
|---------|------------------|
| f-string without variable interpolation | Always print both compared values in f-string |
| Line exceeds 120 characters | Use parentheses to split long f-strings |
| Using `.format()` instead of f-string | Always use f-string format for assertions |
| Trailing whitespace on lines | Remove all trailing spaces from line endings |
| No error message in assert | Always include descriptive error message |

## Line Formatting

```python
# Correct - no trailing whitespace, proper indentation
def test_example():
    x = 1
    y = 2
    assert x == y, \
        (f"Values not equal: "
         f"x={x}, y={y}")

# Wrong - trailing whitespace after closing parenthesis
def test_example():
    x = 1
    y = 2
    assert x == y, \
        (f"Values not equal: "
         f"x={x}, y={y}")  # <-- Trailing space here!
```

## Function Return Value Assertion

**IMPORTANT: When asserting function returns None, do NOT assign to variable first.**

This avoids linter warning: "Assigning result of a function call, where the function returns None"

```python
# Correct - direct assertion without variable assignment
assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
    f"get_expand_impl should return None, "
    f"got{op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
)

# Wrong - assigning None result to variable triggers linter warning
impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
assert impl is None, (
    f"get_expand_impl should return None, got {impl}"
)

# Correct - for callable return value, variable assignment is fine
impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
assert callable(impl), (
    f"get_expand_impl should return callable, got {type(impl)}"
)
```

## get_expand_impl Verification Rules

**IMPORTANT: Only verify `get_expand_impl` once per test class if the operator does NOT override it.**

### When Operator Does NOT Override get_expand_impl

If the operator uses the default implementation (returns `None`), verify it in **only one test case** with a comment explaining why other tests don't need this verification:

```python
@patch("hyper_parallel.core.dtensor.device_mesh.platform")
def test_softmax_data_parallel_success(self, mock_platform):
    """Test softmax with data parallel."""
    mesh = self._make_2x2x2_mesh(mock_platform)
    placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, placements, 3)
    
    output_layout = self.op.infer_layout((x_layout,), (-1,))
    
    assert output_layout.tensor_map == (2, -1, -1), (
        f"Expected (2, -1, -1), got {output_layout.tensor_map}"
    )
    
    # Since `get_expand_impl` is not overridden, it returns None by default.
    # The same applies to other test classes, so it is unnecessary to test its return value.
    assert self.op.get_expand_impl(None, output_layout, (x_layout,), (-1,)) is None

@patch("hyper_parallel.core.dtensor.device_mesh.platform")
def test_softmax_model_parallel_success(self, mock_platform):
    """Test softmax with model parallel."""
    mesh = self._make_2x2x2_mesh(mock_platform)
    placements = (Replicate(), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, placements, 3)
    
    output_layout = self.op.infer_layout((x_layout,), (-1,))
    
    assert output_layout.tensor_map == (-1, -1, 0), (
        f"Expected (-1, -1, 0), got {output_layout.tensor_map}"
    )
    # No need to verify get_expand_impl here - already verified in test_softmax_data_parallel_success
```

### When Operator DOES Override get_expand_impl

If the operator has a custom `get_expand_impl` that returns a callable, verify it in **every test case**:

```python
@patch("hyper_parallel.core.dtensor.device_mesh.platform")
def test_matmul_data_parallel_success(self, mock_platform):
    """Test matmul with data parallel."""
    mesh = self._make_2x2x2_mesh(mock_platform)
    # ... setup code ...
    
    output_layout = self.op.infer_layout((x_layout, w_layout), extra_args)
    
    assert output_layout.tensor_map == expected_map, f"..."
    
    # Must verify in every test since get_expand_impl is overridden
    impl = self.op.get_expand_impl(None, output_layout, (x_layout, w_layout), extra_args)
    assert callable(impl), f"Expected callable, got {type(impl)}"
```

### Summary Table

| Scenario | Verification Frequency | Comment Required |
|----------|------------------------|------------------|
| Default `get_expand_impl` (returns None) | **Once per test class** | Yes - explain why only once |
| Overridden `get_expand_impl` (returns callable) | **Every test case** | No - must verify each time |
