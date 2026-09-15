# Test Case Submission Guide

## 1. Basic Principles for Test Cases

- Currently, only UT test cases that run on the CPU backend are supported.
- Submit all test cases to the `tests/uni_agent` directory. Test cases outside this directory will not be run in the gate.
- Under the `tests/uni_agent` directory, submit test cases by functional module.

## 2. Test Case Submission Steps

### 1. Place runtime dependencies uniformly in `requirements-test.txt` in the root directory

### 2. Submit test cases to the designated location according to the principles described above

### 3. Add tags to test cases to specify the runtime backend and test case level

#### Supported tags:

- cpu: Specifies that the test case runs on the CPU backend
- gpu: Specifies that the test case runs on the GPU backend (not actually run for now)
- npu: Specifies that the test case runs on the NPU backend (not actually run for now)
- level0: Specifies the test case as a gate-level test case, which will run each time the gate is triggered
- level1: Specifies the test case as a version-level test case, which is only run manually during version verification.

#### Notes:

- Configure tags at the function granularity. Do not configure tags directly on a class.
- Both a backend tag and a test case level tag must be specified for the test case to run in the gate.
- Only one of `level0` and `level1` can be specified for a single test case.

#### Test case example

Script name format: `test_xxx.py`

```

import pytest


@pytest.mark.cpu
@pytest.mark.level0
def test_xxxx():
    ...

```

### 4. Submit the datasets and weights required by the test cases to the gate machine

Not supported for now; to be added.

## 3. Gate Test Case Verification Method

### 1. Logic verification

In the "checks" tab of the PR, find the Python Backend CI workflow. There are two sub-workflows under this workflow; click to view the corresponding gate logs.

The logs contain `test case runtime information` and `coverage statistics results`.

Ensure that the test cases run without errors before merging.

### 2. Coverage verification

**Currently, this verification is only a recommendation and is not set as a mandatory check**

After the Python Backend CI workflow finishes running, a coverage comment will be added to the PR, showing the coverage of the code corresponding to the changes in this PR.

This includes: `coverage of the modified code` and `coverage details displayed by code line`.

It is recommended to improve the test case coverage based on the results.
