import onnx 

model = onnx.load("~/code/sofie_gpu/SOFIE/src/SOFIE_core/test/input_models/Where.onnx")

print(model.graph)
