# # encoder_export.py
# import torch

# model = torch.jit.load('net_encode.pt')
# dummy_input = torch.randn(1, 3, 384, 384)

# torch.onnx.export(
#     model,
#     dummy_input,
#     'net_encode.onnx',
#     input_names=['input_image'],
#     output_names=['feature_vector'],
#     dynamic_axes={'input_image': {0: 'batch'}},
#     opset_version=13
# )
# generator_export.py
import torch

model = torch.jit.load('net_decode.pt')
dummy_feature = torch.randn(1, 512).to('cuda')  # 需与encoder输出维度一致
dummy_param = torch.randn(1, 32).to('cuda')

torch.onnx.export(
    model,
    (dummy_feature, dummy_param),
    'net_decode.onnx',
    input_names=['feature_input', 'param_input'],
    output_names=['mouth_image'],
    dynamic_axes={
        'feature_input': {0: 'batch'},
        'param_input': {0: 'batch'}
    },
    opset_version=13
)