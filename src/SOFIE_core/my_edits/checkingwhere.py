import onnxruntime as ort
import numpy as np
import onnx

model = onnx.load("/home/imaayush/code/sofie_gpu/SOFIE/src/SOFIE_core/test/input_models/Where.onnx")
#print(model.graph)

session = ort.InferenceSession("/home/imaayush/code/sofie_gpu/SOFIE/src/SOFIE_core/test/input_models/Where.onnx")

cond = np.array([[True], [False], [True]], dtype=bool)
a = np.array([[1.0, 2.0]], dtype=np.float32)
b = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)

out = session.run(None, {"cond": cond, "inputA": a, "inputB": b})
print(out)
