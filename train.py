import torch
import torch.nn as nn
import torch.optim as optim
from model.embedding import Embedding
from model.LSTM import SimpleLSTM
from tqdm import tqdm
import wandb

word_num = 100
embedding_dim = 32
hidden_size = 8
output_size = 1
seq_len = 7
batch_size = 10
num_samples = 100
num_epochs = 100

wandb.init(project="simple_LSTM", name="LSTM_baseline", config={
    "epochs": num_epochs,
    "batch_size": batch_size,
    "hidden_size": hidden_size,
    "embedding_dim": embedding_dim,
    "lr": 0.01
})

embedding = Embedding(word_num, embedding_dim)
lstm = SimpleLSTM(embedding_dim, hidden_size, output_size)

criterion = nn.MSELoss()
# 임베딩과 모델 파라미터 모두 학습 가능하게 설정
optimizer = optim.Adam(list(embedding.parameters()) + list(lstm.parameters()), lr=0.01)

# 더미 데이터
# 총 1000개의 문장, 시퀀스 길이 7
X_tokens = torch.randint(0, word_num, (num_samples, seq_len))
y_true = torch.rand(num_samples, 1)

lstm.train()
for epoch in range(num_epochs):
    total_loss = 0
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1:03d}:"):
        x_batch = X_tokens[i:i+batch_size]
        y_batch = y_true[i:i+batch_size]

        # 임베딩 진행
        # 임베딩 함수가 데이터셋의 각 단어(요소)를 모두 돌며 임베딩 차원수로 임베딩 진행
        # 결과: (배치사이즈, 시퀀스 길이, 임베딩 차원수) = (10, 7, 32)
        # 이때 배치사이즈 = 한 배치 안의 문장수
        x_embed = embedding(x_batch)

        output = lstm(x_embed)

        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / (num_samples/batch_size)
    wandb.log({"loss": avg_loss, "epoch": epoch + 1})
    print(f"[{epoch + 1:03d}] Loss: {avg_loss:.4f}")

# 학습된 모델 저장
torch.save({
    'embedding': embedding.state_dict(),
    'lstm': lstm.state_dict()
}, 'model_weights.pth')