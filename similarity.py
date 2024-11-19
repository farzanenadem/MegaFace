import numpy as np

class VectorSimilarity:
    @staticmethod
    def cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return dot_product / (norm1 * norm2)

    @staticmethod
    def euclidean_distance(vector1, vector2):
        return np.linalg.norm(np.array(vector1) - np.array(vector2))

    @staticmethod
    def manhattan_distance(vector1, vector2):
        return np.sum(np.abs(np.array(vector1) - np.array(vector2)))

