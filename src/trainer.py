# src/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# 1. 定义训练过程
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练模型的一轮
    :param model: 训练的模型
    :param train_loader: 训练数据的DataLoader
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 使用的设备（CPU或GPU）
    :return: 当前训练轮次的平均损失
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        total_loss += loss.item()
        
        # 反向传播并更新权重
        loss.backward()
        optimizer.step()
        
        # 计算准确度
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 2. 定义验证过程
def evaluate(model, val_loader, criterion, device):
    """
    验证模型在验证集上的表现
    :param model: 训练的模型
    :param val_loader: 验证数据的DataLoader
    :param criterion: 损失函数
    :param device: 使用的设备（CPU或GPU）
    :return: 验证集上的平均损失和准确度
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 计算准确度
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            # 收集预测结果用于后续的指标计算
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1

# 3. 定义测试过程
def test(model, test_loader, device):
    """
    测试模型在测试集上的最终表现
    :param model: 训练的模型
    :param test_loader: 测试数据的DataLoader
    :param device: 使用的设备（CPU或GPU）
    :return: 测试集上的准确度和F1分数
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算准确度
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            # 收集预测结果用于后续的指标计算
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

# 4. 训练过程的主循环
def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=10, batch_size=32, learning_rate=0.001, device='cuda'):
    """
    模型训练、验证和测试的主过程
    :param model: 模型
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param X_val: 验证数据
    :param y_val: 验证标签
    :param X_test: 测试数据
    :param y_test: 测试标签
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param learning_rate: 学习率
    :param device: 使用的设备（'cpu' 或 'cuda'）
    :return: 最终测试结果
    """
    # 将数据转换为TensorDataset并加载到DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 将模型移到指定设备
    model.to(device)
    
    # 训练循环
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # 训练过程
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        
        # 验证过程
        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}')
    
    # 测试过程
    test_accuracy, test_f1 = test(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}')
    
    return test_accuracy, test_f1

