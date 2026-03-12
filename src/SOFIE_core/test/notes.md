## Test Setup Notes

### input_models/
- `Softplus.onnx` — generated using Python onnx library, shape `[1, 6]`, single Softplus node

### input_models/references/
- `Softplus.ref.hxx` — expected outputs computed via numpy: `np.log(1 + np.exp(inputs))` on inputs `[1.0, -2.0, 3.0, 0.5, -1.0, 2.0]`
- `Tanh.ref.hxx` — expected outputs are tanh values; inputs in the test were derived by running `np.arctanh` on these reference outputs

### TestCustomModelsFromONNXForAlpakaCuda.cxx
Three new TEST_F blocks added — Elu, Tanh, Softplus.


## Build and Test Pipeline

### For Tanh and Elu (no new parser file, just ROperator changes) 
```bash
cmake --build . --target SOFIE_core
cmake --build . --target emitFromONNXAlpaka
cd src/SOFIE_core/test
./emitFromONNXAlpaka
```
Sanity check before building test (example for Tanh):
```bash
grep -n "TanhKernel\|tanhKernel" ~/code/sofie_gpu/SOFIE/build/src/SOFIE_core/test/Tanh_FromONNX_GPU_ALPAKA.hxx
```
Then for testing:
```bash
cmake --build ~/code/sofie_gpu/SOFIE/build --target TestCustomModelsFromONNXForAlpakaCuda
ctest -R TestCustomModelsFromONNXForAlpakaCuda -V
```

### For Softplus (new ONNX file + new parser file)
Since `Softplus.onnx` was newly added and `ParseSoftplus.cxx` was a new source file, cmake reconfigure was needed:
```bash
cd ~/code/sofie_gpu/SOFIE/build
cmake ..
cmake --build . --target SOFIE_core
cmake --build . --target SOFIE_parsers
cmake --build . --target emitFromONNXAlpaka
cd src/SOFIE_core/test
./emitFromONNXAlpaka
```
Sanity check:
```bash
grep -n "SoftplusKernel\|softPlusKernel" ~/code/sofie_gpu/SOFIE/build/src/SOFIE_core/test/Softplus_FromONNX_GPU_ALPAKA.hxx
```
Then:
```bash
cmake --build ~/code/sofie_gpu/SOFIE/build --target TestCustomModelsFromONNXForAlpakaCuda
ctest -R TestCustomModelsFromONNXForAlpakaCuda -V
```

