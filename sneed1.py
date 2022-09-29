import onnx

# onnx_model is an in-memory ModelProto
onnx_model = onnx.load('kek.onnx')
print(onnx_model)