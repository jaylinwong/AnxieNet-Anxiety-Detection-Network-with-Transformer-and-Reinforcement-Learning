# src/reinforcement_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class ReinforcementLearningOptimizer:
    def __init__(self, model, criterion, optimizer, device, gamma=0.99):
        """
        初始化强化学习优化器
        :param model: 需要优化的模型
        :param criterion: 损失函数
        :param optimizer: 优化器（如Adam）
        :param device: 使用的设备（CPU或GPU）
        :param gamma: 折扣因子，决定奖励的衰减程度
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma

    def compute_reward(self, outputs, labels):
        """
        基于准确率或其他指标计算奖励
        :param outputs: 模型的输出
        :param labels: 真实标签
        :return: 奖励值
        """
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        return accuracy  # 这里使用准确度作为奖励

    def optimize(self, X_train, y_train, epochs=10, batch_size=32):
        """
        训练并使用强化学习优化器进行优化
        :param X_train: 训练数据
        :param y_train: 训练标签
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :return: 训练过程中的损失和奖励
        """
        self.model.train()

        total_loss = []
        total_rewards = []

        for epoch in range(epochs):
            # 将数据转换为tensor并移动到设备
            inputs = Variable(X_train).to(self.device)
            labels = Variable(y_train).to(self.device)

            # 数据分批
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                # 清空梯度
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_labels)

                # 计算奖励
                reward = self.compute_reward(outputs, batch_labels)

                # 强化学习：通过奖励进行反向传播
                loss.backward()
                self.optimizer.step()

                # 保存损失和奖励
                total_loss.append(loss.item())
                total_rewards.append(reward)

            # 打印当前训练的损失和奖励
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {np.mean(total_loss[-len(inputs) // batch_size:]):.4f}, Reward: {np.mean(total_rewards[-len(inputs) // batch_size:]):.4f}")

        return total_loss, total_rewards


def train_model_with_rl(model, X_train, y_train, criterion, optimizer, device, epochs=10, batch_size=32):
    """
    用强化学习训练模型
    :param model: 模型
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 使用的设备（CPU或GPU）
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :return: 训练过程中每一轮的损失和奖励
    """
    # 初始化强化学习优化器
    rl_optimizer = ReinforcementLearningOptimizer(model, criterion, optimizer, device)

    # 使用强化学习优化器进行训练
    loss, rewards = rl_optimizer.optimize(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return loss, rewards
