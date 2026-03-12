# GSOC Exercise 4 

## (Project name: ML Inference on heterogeneous architectures using SOFIE)

# Alpaka GPU Operator Implementation - Session Notes EXERCISE 4

## ELU 
- `ROperator_Elu.hxx` — added three GPU methods
- `ROperator.hxx` — added `ELU = 8` to enum
- `TestCustomModelsFromONNXForAlpakaCuda.cxx` — added `TEST_F(SofieAlpakaTest, Elu)`
- `input_models/references/Elu.ref.hxx` — expected outputs
- generated inputs from the expected outputs

## Tanh
- `ROperator_Tanh.hxx` — added `Generate_GPU_Kernel_ALPAKA`, `Generate_GPU_Kernel_Definitions_ALPAKA`, `Generate_GPU_ALPAKA`; set `fKind` in constructor
- `ROperator.hxx` — added `TANH` to enum
- `TestCustomModelsFromONNXForAlpakaCuda.cxx` — added `TEST_F(SofieAlpakaTest, Tanh)`
- `input_models/references/Tanh.ref.hxx` — expected outputs; inputs derived via `np.arctanh` on reference outputs

## Softplus (new operator, nothing existed)
- `ROperator_Softplus.hxx` — created from scratch; CPU `Generate()` with `log(1 + exp(x))`; all three GPU methods;
- `ROperator.hxx` — added `SOFTPLUS` to enum
- `src/SOFIE_parsers/src/ParseSoftplus.cxx` — created; copied from `ParseTanh.cxx`, swapped Tanh for Softplus
- `src/SOFIE_parsers/src/RModelParser_ONNX.cxx` — added `extern ParserFuncSignature ParseSoftplus` and `RegisterOperator("Softplus", ParseSoftplus)`
- `src/SOFIE_parsers/CMakeLists.txt` — added `src/ParseSoftplus.cxx` to sources
- `input_models/Softplus.onnx` — generated via Python/onnx, shape `[1, 6]`; file name: generate_softplus_onnx.py
- `input_models/references/Softplus.ref.hxx` — expected outputs computed via `np.log(1 + np.exp(inputs))`
- `TestCustomModelsFromONNXForAlpakaCuda.cxx` — added `TEST_F(SofieAlpakaTest, Softplus)`

## Sigmoid
- Already implemented in src/SOFIE_core/src/ROperator_Sigmoid.hxx file.

