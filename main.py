import argparse
import os
from megaface_feature_generator import MegafaceEmbeddingGenerator
from onnx_feature_generator import OnnxEmbeddingGenerator
from similarity import VectorSimilarity
from convert_megaface_to_onnx import Megaface2Onnx

# Define the argument parser
parser = argparse.ArgumentParser(description="Face embedding extraction with Megaface and Onnx")
parser.add_argument('--image1_path', type=str, default='test_imgs/test1.jpg', help='Path to the input image 1')
parser.add_argument('--image2_path', type=str, default='test_imgs/test2.jpg', help='Path to the input image 2')
parser.add_argument('--arch', type=str, default='iresnet50', help='Model architecture (default: iresnet50)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing images (default: 64)')
parser.add_argument('--cpu_mode', type=bool, default=True, help='Use CPU mode (default: True)')
parser.add_argument('--embedding_size', type=int, default=512, help='Size of the embedding (default: 512)')
parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
parser.add_argument('--model_path', type=str, default='weight/magface_iresnet50_MS1MV2_ddp_fp32.pth',
                    help='Path to the model weights (default: weight/magface_iresnet50_MS1MV2_ddp_fp32.pth)')
parser.add_argument('--onnx_output_path', type=str, default='weight/magface_iresnet50.onnx',
                    help='Path to save the converted ONNX model (default: weight/magface_iresnet50.onnx)')

# Parse the arguments
args = parser.parse_args()

# Validate the model path
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f'{args.model_path} does not exist')

# Initialize
similarity_calculator = VectorSimilarity()
image_paths = [args.image_path1, args.image_path2]

# Base Approach usingimage_paths MegafaceEmbeddingGenerator
print("************************** Base Approach **************************")
feature_generator = MegafaceEmbeddingGenerator(
    arch=args.arch,
    batch_size=args.batch_size,
    cpu_mode=args.cpu_mode,
    embedding_size=args.embedding_size,
    workers=args.workers,
    model_path=args.model_path,
)

embeddings = feature_generator.extract_features(image_paths)

cosine_similarity = similarity_calculator.cosine_similarity(embeddings[0], embeddings[1])
print(f"Cosine similarity between [{args.image_path1}] and [{args.image_path2}]: {cosine_similarity:.5f}")

# Onnx Approach
print("************************** Onnx Approach **************************")
converter = Megaface2Onnx(args.arch, args.batch_size, args.cpu_mode, args.embedding_size)
converter.convert(args.model_path, args.onnx_output_path)

feature_generator = OnnxEmbeddingGenerator(
    onnx_model_path=args.onnx_output_path,
    batch_size=args.batch_size,
    cpu_mode=args.cpu_mode)

embeddings = feature_generator.extract_features(image_paths)

cosine_similarity = similarity_calculator.cosine_similarity(embeddings[0], embeddings[1])
print(f"Cosine similarity between [{args.image_path1}] and [{args.image_path2}]: {cosine_similarity:.5f}")
