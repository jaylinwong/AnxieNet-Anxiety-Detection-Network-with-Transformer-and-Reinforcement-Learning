# src/utils.py

import torch
import matplotlib.pyplot as plt
import os
import shutil

def plot_training_history(losses, rewards, save_path=None):
    """
    可视化训练过程中的损失和奖励
    :param losses: 每一轮的训练损失
    :param rewards: 每一轮的奖励
    :param save_path: 如果提供，则保存图像到该路径
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制损失图
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制奖励图
    plt.subplot(1, 2, 2)
    plt.plot(rewards, label='Reward', color='blue')
    plt.title('Training Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model(model, save_path):
    """
    保存训练好的模型
    :param model: 训练好的模型
    :param save_path: 保存模型的路径
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path):
    """
    加载模型的权重
    :param model: 模型实例
    :param load_path: 模型文件路径
    :return: 加载权重后的模型
    """
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")
    else:
        print(f"Error: The file at {load_path} does not exist.")
    
    return model

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    """
    绘制混淆矩阵
    :param cm: 混淆矩阵
    :param classes: 类别标签
    :param title: 图表标题
    :param cmap: 颜色图
    :param save_path: 保存路径
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 在矩阵中添加数值
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()

def create_directory(path):
    """
    如果指定路径不存在，创建该路径
    :param path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)

def move_directory(src, dst):
    """
    移动目录从源路径到目标路径
    :param src: 源路径
    :param dst: 目标路径
    """
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Directory moved from {src} to {dst}")
    else:
        print(f"Error: The source directory {src} does not exist.")
