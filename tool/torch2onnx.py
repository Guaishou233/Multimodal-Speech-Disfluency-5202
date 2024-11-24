import torch
import onnx

import onnxruntime as ort
import sys
sys.path.append('E:/HKUST_ASS/5202/Multimodal-Speech-Disfluency-main')

from main_audio_video_unified_fusion import AVStutterNet

# 加载 PyTorch 模型
model = AVStutterNet()
checkpoint = torch.load("G:/HKUST_DATA/5202/27216024/Multimodal_dataset/multimodal_speech/main_codebase/saved_models/audio_video_only/baseline_0053.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 定义一个示例输入
# torch.Size([64, 1, 149, 768])
# torch.Size([64, 1, 90, 768])
# 模拟第一个输入，形状为torch.Size([64, 1, 149, 768])
input1 = torch.randn(64, 1, 149, 768)

# 模拟第二个输入，形状为torch.Size([64, 1, 90, 768])
input2 = torch.randn(64, 1, 90, 768)
dummy_input = (input1,input2)  # 根据模型输入形状调整

# 导出为 ONNX 格式
onnx_file_path = "G:/HKUST_DATA/5202/27216024/Multimodal_dataset/multimodal_speech/main_codebase/saved_models/audio_video_only/baseline_0053.onnx"
torch.onnx.export(model, dummy_input, onnx_file_path, opset_version=14,
                  input_names=["input"], output_names=["output"])
print(f"ONNX 模型已保存到: {onnx_file_path}")

model = onnx.load(onnx_file_path)
ort_session = ort.InferenceSession(onnx_file_path)
onnx_outputs = ort_session.run(None, {'input': dummy_input})
print('Export ONNX!')

