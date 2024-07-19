import torch
import torch.nn as nn
import torch.optim as optim
import os

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.embedding = nn.Embedding(input_dim, d_model)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src, tgt):
        # Embedding
        src = self.embedding(src) * torch.sqrt(torch.tensor([self.d_model], dtype=torch.float32))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor([self.d_model], dtype=torch.float32))
        
        # Transformer
        output = self.transformer(src, tgt)
        
        # Final linear layer
        output = self.fc_out(output)
        return output

# 超参数
input_dim = 1000  # 输入词汇表大小，可以理解成有个词汇表，有1000个词，每个词用（0～1000）的数字表示
output_dim = 1000  # 输出词汇表大小
d_model = 512  # 嵌入维度
nhead = 8  # 多头注意力机制中的头数
num_encoder_layers = 6  # 编码器层数
num_decoder_layers = 6  # 解码器层数
dim_feedforward = 2048  # 前馈网络维度
dropout = 0.1  # dropout概率
num_epochs =100  # 训练的epoch数

# 初始化模型
model = TransformerModel(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 生成一些随机的输入数据和目标数据 (batch_size, seq_length)
src = torch.randint(0, input_dim, (10, 32))  # 假设batch_size=10, seq_length=32
tgt = torch.randint(0, output_dim, (10, 32))

# 训练循环
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(src, tgt)
    loss = criterion(output.view(-1, output_dim), tgt.view(-1)) #view作用类似reshape
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 保存模型参数
model_path = "/home2/wzc/python project/Uniad_related/Transformer_example/transformer_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

