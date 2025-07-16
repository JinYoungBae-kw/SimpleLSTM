import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, word_num, embedding_dim):
        super().__init__()
        # num_embeddings = 전체 데이터셋에 존재하는 고유한 단어의 개수
        # embedding_dim = 임베딩 차원수
        self.embedding = nn.Embedding(num_embeddings=word_num, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)