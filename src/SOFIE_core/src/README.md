# GSoC 2026 Exercise 4
## Project: ML Inference on Heterogeneous Architectures using SOFIE

This branch contains GPU operator implementations for the SOFIE alpaka backend.
Each operator has three GPU methods added: `Generate_GPU_Kernel_ALPAKA`, `Generate_GPU_Kernel_Definitions_ALPAKA`, and `Generate_GPU_ALPAKA`.

## Implemented Operators

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
