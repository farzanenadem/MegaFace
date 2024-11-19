import torch
from megaface_model_loader import network_builder


class Megaface2Onnx:
    def __init__(self, arch, batch_size, cpu_mode, embedding_size):
        self.arch = arch
        self.batch_size = batch_size
        self.cpu_mode = cpu_mode
        self.embedding_size = embedding_size

    def convert(self, pth_path, onnx_path, opset_version=10):
        model = network_builder(self.arch, self.embedding_size, self.cpu_mode, pth_path)
        model.eval()
        x = torch.randn(self.batch_size, 3, 112, 112)

        torch.onnx.export(
            model,
            x,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

