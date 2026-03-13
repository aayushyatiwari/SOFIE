# GSoC - ParseSoftplus new file added
## ParseSoftplus Implementation Notes

Softplus was not registered in the ONNX parser at all, so the emitter was failing with "Operator type Softplus is not yet supported".

Three things needed to happen:

1. `ParseSoftplus.cxx` — the actual parsing function. Copied structure from `ParseTanh.cxx` since both are parameterless elementwise operators. Reads input tensor name, checks its type is registered, creates an `ROperator_Softplus<float>` instance, registers the output tensor type.

2. `RModelParser_ONNX.cxx` — two lines added:
   - `extern ParserFuncSignature ParseSoftplus;` near the other extern declarations
   - `RegisterOperator("Softplus", ParseSoftplus);` near the other RegisterOperator calls

3. `CMakeLists.txt` for SOFIE_parsers — added `src/ParseSoftplus.cxx` to the source list, otherwise the file just gets ignored at build time.

After rebuilding `SOFIE_parsers` and `emitFromONNXAlpaka`, the emitter picked up `Softplus.onnx` and generated `Softplus_FromONNX_GPU_ALPAKA.hxx` correctly.

