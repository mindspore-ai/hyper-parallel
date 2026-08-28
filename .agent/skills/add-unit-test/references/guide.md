# HyperParallel Unit Tests Guide

Procedural reference for the **add-unit-test** skill. Hard constraints:
`.agent/rules/unit-test.md` and `.agent/rules/testing.md`.

## When to Use

- Add UT for new or existing functionality / raise coverage
- Write tests that do not need real distributed communication or GPU/NPU

## Step-by-Step Guide

### Step 1: Understand Test Structure

**Runner:** pytest. **Authoring:** write cases as `unittest.TestCase` (pytest
discovers and runs them). Do not treat "unittest" and "pytest" as competing
frameworks — one is the case style, the other is the runner.

Layout:

| Component | Description                                                                                                         | Location Pattern                        |
| --------- |---------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| **Unit Tests** | Conducting independent testing of functions and modules using only the CPU                                          | `tests/ut/<module>/<feature>/test_*.py` |
| **Integration Tests** | Test interactions between components and modules that require actual distributed communication and GPU/NPU hardware | `tests/st/<module>/test_*.py`           |

**Key Principles:**
- Unit test should be **hardware-agnostic** (no GPU/NPU dependency)
- Unit test should **mock distributed communication** (no actual torch.distributed calls)
- Unit test should follow **Arrange-Act-Assert** pattern

### Step 2: Create Test File

Create test files with naming convention: `test_<src_file_name>.py`

```python
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
"""Unit tests for <Component> functionality."""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# Import the module to test
from hyper_parallel.platform import get_platform
...
```

### Step 3: Write Test Class

Create a test class inheriting from `unittest.TestCase`:

```python
class Test<Component>(unittest.TestCase):
    """Unit tests for <Component> functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure environment, choose torch or mindspore platform
        import os
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        
        # Initialize common test objects
        self.platform = get_platform()
        self.device = torch.device("cpu")  # Always use CPU for unit tests
    
    def tearDown(self):
        """Clean up after each test method (if needed)."""
        pass
```

### Step 4: Mock Distributed Environment

HyperParallel tests should never rely on actual distributed communication. Always mock `torch.distributed` calls:

```python
@patch('torch.distributed.all_gather')
def test_all_gather_function(self, mock_all_gather):
    """Test all_gather functionality with mocked distributed communication."""
    # Arrange
    mock_all_gather.return_value = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    tensor = torch.tensor([1.0, 2.0])
    
    # Act
    result = self.platform.some_function_that_uses_all_gather(tensor)
    
    # Assert
    mock_all_gather.assert_called_once()
    self.assertTrue(torch.allclose(result, torch.tensor([1.0, 2.0, 3.0, 4.0])))
```

### Step 5: Avoid Hardware Dependencies

Never assume GPU/NPU availability. Always use CPU device:

```python
def test_tensor_operation(self):
    """Test tensor operation without GPU dependency."""
    # Always use CPU device explicitly
    tensor = torch.randn(2, 2, device=self.device)
    
    # Perform operation
    result = self.platform.some_tensor_operation(tensor)
    
    # Assert
    self.assertEqual(result.device.type, "cpu")
    # Add more assertions...
```

### Step 6: Mock Complex Dependencies

For complex dependencies like DTensor or DeviceMesh, use mocking:

```python
@patch.object(DTensor, "from_local")
def test_dtensor_operation(self, mock_dtensor_from_local):
    """Test operation that uses DTensor with mocking."""
    # Arrange
    mock_dtensor = MagicMock(spec=DTensor)
    mock_dtensor._local_tensor = torch.tensor([1.0, 2.0])
    mock_dtensor_from_local.return_value = mock_dtensor
    
    # Act
    result = self.platform.some_dtensor_operation(torch.tensor([1.0, 2.0]))
    
    # Assert
    mock_dtensor_from_local.assert_called_once()
    # Add more assertions...
```

### Step 7: Test Error Conditions

Don't forget to test error cases:

```python
def test_invalid_input(self):
    """Test error handling for invalid input."""
    # Arrange
    invalid_input = "not a tensor"
    
    # Act & Assert
    with self.assertRaises(TypeError):
        self.platform.some_function(invalid_input)
```

### Step 8: Use Helper Methods

For common test setup, create helper methods:

```python
def _create_mock_dtensor(self, mock_dtensor_from_local, data):
    """Create a mock DTensor instance with common settings."""
    mock_dtensor_instance = MagicMock(spec=DTensor)
    mock_dtensor_instance._local_tensor = torch.tensor(data)
    mock_dtensor_instance.detach.return_value = mock_dtensor_instance
    mock_dtensor_from_local.return_value = mock_dtensor_instance
    return mock_dtensor_instance

@patch.object(DTensor, "from_local")
def test_with_mock_dtensor(self, mock_dtensor_from_local):
    """Test with mock DTensor using helper method."""
    # Arrange
    mock_dtensor = self._create_mock_dtensor(mock_dtensor_from_local, [1.0, 2.0, 3.0])
    
    # Act
    result = self.platform.some_function(mock_dtensor)
    
    # Assert
    # ... assertions ...
```

## Key Testing Patterns

### Mocking Distributed Communication

| Distributed Operation | Mock Pattern |
| --------------------- | ------------ |
| `torch.distributed.all_gather` | `@patch('torch.distributed.all_gather')` |
| `torch.distributed.all_reduce` | `@patch('torch.distributed.all_reduce')` |
| `torch.distributed.reduce_scatter` | `@patch('torch.distributed.reduce_scatter')` |
| `torch.distributed.new_group` | `@patch('torch.distributed.new_group')` |

### Testing Parameterized Functions

Use `subTest` for parameterized testing:

```python
def test_parameterized_function(self):
    """Test function with different parameters."""
    test_cases = [
        (1, 2, 3),   # (input1, input2, expected)
        (4, 5, 9),
        (10, -2, 8)
    ]
    
    for input1, input2, expected in test_cases:
        with self.subTest(input1=input1, input2=input2):
            result = self.platform.add(input1, input2)
            self.assertEqual(result, expected)
```

### Testing Nested Structures

Test functions that handle nested data structures:

```python
def test_nested_structure_processing(self):
    """Test processing of nested data structures."""
    # Arrange
    test_data = {
        "tensor1": torch.tensor([1.0, 2.0]),
        "list": [torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])],
        "nested": {
            "tensor2": torch.tensor([7.0, 8.0]),
            "value": 10
        }
    }
    
    # Act
    result = self.platform.process_nested_structure(test_data)
    
    # Assert
    self.assertTrue(torch.allclose(result["tensor1"], torch.tensor([2.0, 4.0])))  # Assuming doubling operation
    # Add more assertions...
```

## Common Mistakes to Avoid

- ❌ **Using actual GPU/NPU**: Always use CPU device in unit tests
- ❌ **Not mocking distributed communication**: Never rely on actual torch.distributed calls
- ❌ **Testing too many things at once**: Each test should focus on one functionality
- ❌ **Missing error cases**: Always test invalid inputs and error conditions
- ❌ **Hardcoding values**: Use random values or constants for test data
- ❌ **Not cleaning up**: Use tearDown if resources need cleanup
- ❌ **Poor test naming**: Follow test_<what>_<condition>_<expected> pattern
- ❌ **No docstrings**: Add detailed descriptive Google docstrings to test classes and methods
- ❌ **Do not avoid bugs in source code**: Fix the bugs in the source code rather than modifying tests to skip them.

## Running Tests

```bash
# Run all unit tests
pytest tests/ut

# Run specific test file
pytest tests/ut/platform/torch/test_platform.py

# Run specific test class
pytest tests/ut/platform/torch/test_platform::TestTorchPlatformCore

# Run specific test method
pytest tests/ut/platform/torch/test_platform::TestTorchPlatformCore::test_device_type

# Run with verbose output
pytest -v tests/ut/platform/torch/test_platform.py

# Run with coverage report. Besides HTML, also supports json and markdown formats.
pytest --cov=hyper_parallel --cov-report=html tests/ut

# Run specific test file with coverage report. Besides HTML, also supports json and markdown formats.
pytest --cov=hyper_parallel --cov-report=html tests/ut/platform/torch/test_platform.py

# Run specific test class with coverage report. Besides HTML, also supports json and markdown formats.
pytest --cov=hyper_parallel --cov-report=html tests/ut/platform/torch/test_platform::TestTorchPlatformCore

# Run specific test method with coverage report. Besides HTML, also supports json and markdown formats.
pytest --cov=hyper_parallel --cov-report=html tests/ut/platform/torch/test_platform::TestTorchPlatformCore::test_device_type
```

## Reference Implementations

| Test File | Description | Key Patterns |
| --------- | ----------- | ------------ |
| `tests/ut/platform/torch/test_platform.py` | Core platform functionality tests | Mocking distributed communication, device management |
| `tests/ut/platform/torch/fully_shard/test_fully_shard.py` | Fully sharded parameter tests | Mocking DTensor, state transitions |

## Integration with Development Workflow

- **Add tests first**: Follow test-driven development (TDD) when possible
- **Run tests before committing**: Ensure all tests pass before pushing code
- **Update tests when modifying code**: Always update tests to reflect code changes
- **Add tests for bug fixes**: Always write tests to prevent regression of fixed bugs

By following these guidelines, you can create robust, maintainable, and hardware-agnostic unit tests for HyperParallel that don't rely on actual distributed communication or specific hardware.