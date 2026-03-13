import onnx 
model = onnx.load("input_models/Where.onnx")
print(model.graph)
