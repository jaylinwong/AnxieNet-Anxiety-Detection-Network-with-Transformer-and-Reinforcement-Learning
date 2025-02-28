# train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from model import TransformerModel  # 假设TransformerModel在 model.py 中定义
from trainer import train_model  # 从trainer.py导入训练函数
from utils import save_model, plot_training_history  # 从utils.py导入保存和可视化函数

def load_data(config):
    """
    加载训练、验证和测试数据
    :param config: 配置文件内容
    :return: 训练集、验证集和测试集的DataLoader
    """
    # 假设数据已经是numpy数组格式，并根据config中的路径加载
    X_train = np.random.randn(100, config['model']['input_dim'])  # 模拟的训练数据
    y_train = np.random.randint(0, config['model']['output_dim'], 100)  # 模拟的训练标签

    X_val = np.random.randn(20, config['model']['input_dim'])  # 模拟的验证数据
    y_val = np.random.randint(0, config['model']['output_dim'], 20)  # 模拟的验证标签

    X_test = np.random.randn(20, config['model']['input_dim'])  # 模拟的测试数据
    y_test = np.random.randint(0, config['model']['output_dim'], 20)  # 模拟的测试标签

    # 将数据转为PyTorch tensor并放入DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    # 读取配置文件
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # 设置设备（使用GPU或CPU）
    device = torch.device(config['device']['device_type'] if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader, val_loader, test_loader = load_data(config)

    # 初始化模型
    model = TransformerModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers']
    ).to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 根据需要选择损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # 开始训练过程
    print("Starting training...")
    loss, rewards = train_model(
        model=model,
        X_train=torch.tensor(np.random.randn(100, config['model']['input_dim']), dtype=torch.float32),
        y_train=torch.tensor(np.random.randint(0, config['model']['output_dim'], 100), dtype=torch.long),
        X_val=torch.tensor(np.random.randn(20, config['model']['input_dim']), dtype=torch.float32),
        y_val=torch.tensor(np.random.randint(0, config['model']['output_dim'], 20), dtype=torch.long),
        X_test=torch.tensor(np.random.randn(20, config['model']['input_dim']), dtype=torch.float32),
        y_test=torch.tensor(np.random.randint(0, config['model']['output_dim'], 20), dtype=torch.long),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        device=device
    )

    # 保存训练好的模型
    if config['logging']['save_model']:
        save_model(model, config['logging']['model_save_path'])

    # 可视化训练历史
    if config['logging']['save_training_history']:
        plot_training_history(losses=loss, rewards=rewards, save_path=config['logging']['training_history_path'])

    # 测试模型性能
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
