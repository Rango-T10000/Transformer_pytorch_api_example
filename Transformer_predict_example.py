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
input_dim = 1000  # 输入词汇表大小
output_dim = 1000  # 输出词汇表大小
d_model = 512  # 嵌入维度
nhead = 8  # 多头注意力机制中的头数
num_encoder_layers = 6  # 编码器层数
num_decoder_layers = 6  # 解码器层数
dim_feedforward = 2048  # 前馈网络维度
dropout = 0.1  # dropout概率



# 预测示例
def predict(model, src, max_length=32):
    model.eval()
    tgt = torch.zeros((src.size(0), max_length), dtype=torch.long)  # 初始化目标序列
    with torch.no_grad():
        for i in range(max_length):
            output = model(src, tgt)
            next_word = output[:, i, :].argmax(dim=1)
            tgt[:, i] = next_word
            if (next_word == 0).all():  # 假设0是结束符
                break
    return tgt

# 加载模型参数并进行预测
model_path = "/home2/wzc/python project/Uniad_related/Transformer_example/transformer_model.pth"
model = TransformerModel(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
model.load_state_dict(torch.load(model_path))

# 假设我们有一个新的输入序列
new_src = torch.randint(0, input_dim, (1, 32))  # batch_size=1, seq_length=32
predicted_tgt = predict(model, new_src)
print(f"Predicted target sequence: {predicted_tgt}")