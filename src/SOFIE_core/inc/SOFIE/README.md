# GSoC Doc 2026: ML Inference on Heterogeneous Architectures using SOFIE

This branch contains GPU operator implementations for the SOFIE alpaka backend.


## Exercise 3
Some things and issues I found while building standalone SOFIE.

1. When building SOFIE from source, I ran into a issue. I opened a PR for that. [PR #9](https://github.com/ML4EP/SOFIE/pull/9)
2. Understood the architecture of SOFIE and how files are linked.
3. Understood how operators are tested in the `SOFIE_core/test/` dir.
4. Tried writing standalone alpaka kernels in `src/test_alapaka_kernels/` folder.
5. Some Operators like selu and erf didn't have onnx files and .ref.hxx files so I wrote them myself.

Proceeded to write the operations.

## Exercise 4

Each operator has three GPU methods added: `Generate_GPU_Kernel_ALPAKA`, `Generate_GPU_Kernel_Definitions_ALPAKA`, and `Generate_GPU_ALPAKA`.

### Implemented Operators - 6 in total 

Below are the all the operators I wrote alpaka methods for and files changed.

### ELU
- `ROperator_Elu.hxx` — added three GPU methods
- `ROperator.hxx` — added `ELU` to operator enum
- Unit test and expected outputs added

### Tanh
- `ROperator_Tanh.hxx` — added three GPU methods
- `ROperator.hxx` — added `TANH` to operator enum
- Unit test and expected outputs added; inputs derived via `np.arctanh` on reference outputs

### Softplus (new operator)
- `ROperator_Softplus.hxx` — created from scratch; CPU `Generate()` with `log(1 + exp(x))` and all three GPU methods
- `ROperator.hxx` — added `SOFTPLUS` to operator enum
- `ParseSoftplus.cxx` — created; registered in `RModelParser_ONNX.cxx` and `CMakeLists.txt`
- `input_models/Softplus.onnx` — generated via `generate_softplus_onnx.py`
- Unit test and expected outputs added

### Selu
- `ROperator_Selu.hxx` — added three GPU methods; constants `lambda` and `alpha` hardcoded in kernel
- `input_models/Selu.onnx` — generated via `generate_selu_onnx.py`
- Unit test and expected outputs added

### Erf
- `ROperator_Erf.hxx` — added three GPU methods
- Unit test added; uses existing `Erf.onnx` from the repository
- Expected outputs computed via `scipy.special.erf`

### Identity
- `ROperator_Identity.hxx` — added three GPU methods
- `input_models/Identity.onnx` — generated via `generate_identity_onnx.py`
- Unit test added; expected outputs equal inputs

### Sigmoid
- Already implemented in the repository; used as reference pattern for all other operators

## Building and Running Tests
```bash
cd /path/to/SOFIE/build
cmake ..
cmake --build . --target emitFromONNXAlpaka
cd src/SOFIE_core/test && ./emitFromONNXAlpaka
cmake --build /path/to/SOFIE/build --target TestCustomModelsFromONNXForAlpakaCuda
ctest -R TestCustomModelsFromONNXForAlpakaCuda -V
```


# GSOC 2026 : Exercise 5 (BONUS)

## Overview

This documents the implementation of the ONNX `Where` operator for GPU inference
using the alpaka heterogeneous computing library in SOFIE.

The Where operator performs an element-wise ternary selection:
```cpp
output[i] = condition[i] ? A[i] : B[i]
```

## Files Changed

### `SOFIE/src/SOFIE_core/inc/SOFIE/ROperator_Where.hxx`
Added three GPU methods to the existing `ROperator_Where` class:

- `Generate_GPU_Kernel_ALPAKA()` — defines the `WhereKernel` struct. Each thread
  handles one element. Takes four parameters: `cond` (`uint8_t const*`), `a` (`T
  const*`), `b` (`T const*`), `out` (`T*`).

- `Generate_GPU_Kernel_Definitions_ALPAKA()` — instantiates `WhereKernel
  whereKernel` in the session scope.

- `Generate_GPU_ALPAKA()` — handles broadcasting then dispatches the kernel.
  Broadcasting is done on the host: device buffer is copied to a host vector,
  `UnidirectionalBroadcast` runs on CPU, result is copied back to device. Then
  the kernel is dispatched with all four device buffer pointers.

### `SOFIE/src/SOFIE_core/src/RModel_ALPAKA.cxx`

**`GenerateSessionCode_GPU_ALPAKA()`**
Added `BufUI81D` type alias:
```cpp
fGC += "using BufUI81D = alpaka::Buf<Acc, uint8_t, Dim, Idx>;\n\n";
```

**`GenerateInferSignature_GPU_ALPAKA()`**
Added `BOOL` case in `GetBufType` lambda:
```cpp
if (type == ETensorType::BOOL) return "BufUI81D";
```
This allows `cond` (a graph input of type BOOL) to be passed into `infer()` as
a `BufUI81D` device buffer.

**`GenerateGPU_ALPAKA_Buffers()`**
Two fixes for intermediate BOOL tensors:

1. Changed `std::vector<bool>` to `std::vector<uint8_t>` — `std::vector<bool>`
   is bit-packed in C++ and incompatible with `UnidirectionalBroadcast` which
   expects a contiguous byte array.

2. Added `BufUI81D deviceBuf_` allocation for BOOL intermediate tensors — the
   original code had a comment "No pointer allocation needed for BOOL" and
   skipped device allocation. Where requires the broadcasted condition tensor on
   device for the kernel.

# Testing

Input shapes from `Where.onnx`:
- `cond`: `[3, 1]` bool
- `inputA`: `[1, 2]` float32
- `inputB`: `[3, 2]` float32
- `output`: `[3, 2]` float32

Reference values verified with ONNX Runtime:
```
cond  = [[True], [False], [True]]
inputA = [[1.0, 2.0]]
inputB = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
output = [[1.0, 2.0], [30.0, 40.0], [1.0, 2.0]]
```

Test added to `TestCustomModelsFromONNXForAlpakaCuda.cxx` as
`SofieAlpakaTest.Where`. All 13 tests pass.

## AI use 
I used Claude Sonnet 4.6 extensively to quickly understand the SOFIE standalone project structure and how code is written. 
I take full responsibility for the code.
I know the shallowness of code can be dangerous and so I vividly and intensely focused on the testing of operator methods.

Best
Aayushya Tiwari
