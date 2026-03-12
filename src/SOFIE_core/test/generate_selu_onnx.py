import numpy as np
import onnx
from onnx import helper, TensorProto

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 6])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 6])

node = helper.make_node('Selu', inputs=['X'], outputs=['Y'])

graph = helper.make_graph([node], 'Selu', [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

onnx.save(model, 'input_models/Selu.onnx')
