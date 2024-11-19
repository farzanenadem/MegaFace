import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from megaface_model_loader import network_builder

class MegafaceEmbeddingGenerator:
    def __init__(self,
                 embedding_size=512,
                 batch_size=256,
                 workers=4,
                 cpu_mode=True,
                 arch="iresnet50",
                 model_path="weight/magface_iresnet50_MS1MV2_ddp_fp32.pth"):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.workers = workers
        self.cpu_mode = cpu_mode

        # Initialize the model
        self.model = network_builder(arch, embedding_size, cpu_mode, model_path)
        if not cpu_mode:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.model = torch.nn.DataParallel(self.model)

        self.model.eval()

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
        with torch.no_grad():
            for i in range(0, len(img_paths), self.batch_size):
                batch_paths = img_paths[i:i + self.batch_size]
                images = [self._load_image(path) for path in batch_paths]
                images = torch.stack(images)

                if not self.cpu_mode:
                    images = images.cuda()

                # Forward pass to extract embeddings
                embedding_batch = self.model(images)
                embedding_batch = F.normalize(embedding_batch, p=2, dim=1)  # Normalize embeddings

                embeddings.extend(embedding_batch.cpu().numpy())

        return embeddings


