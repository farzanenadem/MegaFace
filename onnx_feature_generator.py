import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import onnx
import onnxruntime as ort
import numpy as np

class OnnxEmbeddingGenerator:
    def __init__(self, onnx_model_path, batch_size=256, cpu_mode=True):
        self.onnx_model_path = onnx_model_path
        self.batch_size = batch_size
        self.cpu_mode = cpu_mode

        # Load the ONNX model
        self.sess = ort.InferenceSession(onnx_model_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
        ])

    def _load_image(self, img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f'{img_path} does not exist')

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'{img_path} is empty or cannot be read')

        img = cv2.resize(img, (112, 112))
        return self.transform(img)

    def extract_features(self, img_paths):
        embeddings = []

        # Process images in batches
        for i in range(0, len(img_paths), self.batch_size):
            batch_paths = img_paths[i:i + self.batch_size]
            images = [self._load_image(path) for path in batch_paths]
            images = torch.stack(images)
            images = images.numpy()

            # Run the ONNX model to extract embeddings
            input_name = self.sess.get_inputs()[0].name
            output_name = self.sess.get_outputs()[0].name
            embedding_batch = self.sess.run([output_name], {input_name: images})[0]

            # Normalize embeddings
            embedding_batch = F.normalize(torch.tensor(embedding_batch), p=2, dim=1).cpu().numpy()

            embeddings.extend(embedding_batch)

        return embeddings
