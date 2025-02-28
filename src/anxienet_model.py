# src/anxienet_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# 1. 定义Transformer模型架构
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=6):
        super(TransformerModel, self).__init__()
        
        # 定义Transformer编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # 定义分类器
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # 将输入传入Transformer编码器
        x = self.transformer_encoder(x)
        
        # 对Transformer输出的最后一层进行池化
        x = x.mean(dim=1)
        
        # 最后一层的输出通过全连接层得到预测结果
        x = self.fc(x)
        return x

# 2. 强化学习优化部分：用于训练过程中的奖励优化
class ReinforcementLearningOptimizer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
    
    def train(self, X_train, y_train, epochs=10, reward_function=None):
        """
        训练模型并使用强化学习优化
        :param X_train: 训练数据
        :param y_train: 训练标签
        :param epochs: 训练轮数
        :param reward_function: 用于评估奖励的自定义函数
        :return: 训练过程中的损失和奖励
        """
        self.model.train()
        
        total_loss = []
        total_rewards = []
        
        for epoch in range(epochs):
            # 将数据转换为tensor
            inputs = Variable(X_train).to(self.device)
            labels = Variable(y_train).to(self.device)
            
            # 清空梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 计算奖励
            reward = reward_function(outputs, labels) if reward_function else 0
            
            # 反向传播并优化
            loss.backward()
            self.optimizer.step()
            
            # 保存损失和奖励
            total_loss.append(loss.item())
            total_rewards.append(reward)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Reward: {reward:.4f}")
        
        return total_loss, total_rewards

# 3. 自定义奖励函数：基于准确度的奖励函数
def accuracy_reward(outputs, labels):
    """
    基于分类准确度的奖励函数
    :param outputs: 模型的输出
    :param labels: 真实标签
    :return: 奖励分数
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

# 4. 模型训练入口
def train_anxienet(X_train, y_train, input_dim=256, hidden_dim=128, output_dim=2, num_heads=8, num_layers=6, epochs=10, learning_rate=0.001):
    """
    训练AnxieNet模型
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param input_dim: 输入特征维度
    :param hidden_dim: Transformer隐层维度
    :param output_dim: 输出类别数
    :param num_heads: Transformer注意力头数
    :param num_layers: Transformer编码器层数
    :param epochs: 训练轮数
    :param learning_rate: 学习率
    :return: 训练过程中的损失和奖励
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = TransformerModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 初始化强化学习优化器
    rl_optimizer = ReinforcementLearningOptimizer(model, criterion, optimizer, device)
    
    # 训练模型并返回损失和奖励
    total_loss, total_rewards = rl_optimizer.train(X_train, y_train, epochs=epochs, reward_function=accuracy_reward)
    
    return total_loss, total_rewards
