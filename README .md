

```
# AnxieNet: A Transformer-based Multi-modal Anxiety Detection Model

## 项目介绍

AnxieNet 是一个基于 Transformer 架构的多模态焦虑检测模型，旨在通过脑电信号（EEG）和其他生理信号（如皮肤电反应、眼电等）来检测焦虑情绪。该项目结合了深度学习中的 Transformer 模型和强化学习优化方法，能够自动从EEG信号中提取特征，并通过训练优化焦虑检测的准确性。

模型使用 PyTorch 框架进行实现，并结合多种数据预处理和特征提取方法（如 ICA 去噪、Hjorth 特征、功率谱密度、时频域特征等）。训练过程中，采用强化学习对模型进行优化，提升预测性能。

## 主要功能

- 基于 Transformer 的多模态情感分析模型
- 数据预处理与特征提取（包括时域、频域、时频域特征）
- 强化学习优化机制，提升模型的训练效果
- 支持模型训练、验证和测试的完整流程
- 训练历史和结果可视化，包括损失、奖励和混淆矩阵
- 支持模型的保存与加载

### **创新性设计亮点**：
- **多模态输入融合**：不仅利用EEG信号，还结合参与者的情绪评分（SAM、HAM-A量表评分）以及可能的生理信号（如皮肤电反应、眼电等），通过多模态输入的深度融合来提高焦虑检测的准确性。
  
- **自适应注意力机制**：在Transformer的基础上，使用自适应注意力机制对EEG信号和情绪评分等信息进行加权处理，自动识别每个输入特征的重要性。

- **时序特征学习与焦虑动态建模**：使用基于Transformer的自注意力机制，特别是在时间维度上进行深入学习，捕捉焦虑的动态特征。焦虑的状态变化通过时序模型得以建模，允许模型通过时间推移识别出焦虑的变化趋势。

- **基于强化学习的焦虑检测优化**：使用强化学习（Reinforcement Learning）进行焦虑检测的决策优化，不仅考虑当前的焦虑状态，还能够适应个体的变化，使得模型可以在多次反馈下不断优化其焦虑等级的预测。

### **模型名称**：
**AnxieNet**：Anxiety Detection Network with Transformer and Reinforcement Learning.



### **创新性设计亮点详细解释**：
1. **多模态输入融合**：
   - 将EEG信号与其他生理信号（如皮肤电、眼电等）以及心理评分（SAM、HAM-A）结合，利用深度神经网络进行端到端的训练，从而提升焦虑检测的准确性。

   - 通过设计**多模态融合层**，模型能够在多个输入源之间进行信息交互，从而捕捉到更多的焦虑相关信息。

2. **自适应注意力机制**：
   - 基于Transformer架构的**自适应注意力机制**，能够根据数据的不同模态特征自动调整每个输入信号的权重。
   - 该机制将帮助模型更好地识别和处理EEG信号中与焦虑相关的关键信号成分。

3. **时序特征学习**：
   - 通过**Transformer的自注意力机制**，模型能够捕捉EEG信号的时序依赖关系，学习焦虑的动态变化。
   - 这种建模方式有助于焦虑状态的变化趋势的预测，使得模型能够预测短期内焦虑水平的变化。

4. **强化学习优化**：
   - 使用**强化学习**进行焦虑检测任务的优化，模型可以通过不断的反馈进行学习，提高对焦虑状态的精确分类能力。
   - 强化学习通过环境与模型的交互，不仅关注当前焦虑状态，还能适应个体的变化，为个性化的焦虑检测提供可能。


## 项目结构

```plaintext
AnxieNet/
├── data/
│   ├── raw_data/                    # 存放原始数据（.edf、.mat）
│   ├── processed_data/              # 存放预处理后的数据（.mat）
│   ├── multimodal_data/             # 存放情绪评分、皮肤电等生理信号
│   ├── labels/                      # 存放焦虑等级标签（SAM、HAM-A评分）
├── notebooks/                       # Jupyter笔记本文件
│   ├── data_preprocessing.ipynb      # 数据预处理流程
│   ├── feature_extraction.ipynb      # 特征提取与分析
│   ├── anxienet_model.ipynb         # AnxieNet模型训练与评估
├── src/
│   ├── data_preprocessing.py         # 数据预处理脚本（去噪、标准化等）
│   ├── feature_extraction.py         # 特征提取脚本（时间域、频域、时频域特征）
│   ├── anxienet_model.py            # AnxieNet模型的实现（Transformer + Reinforcement Learning）
│   ├── trainer.py                   # 训练与评估脚本
│   ├── reinforcement_learning.py     # 强化学习部分，用于优化焦虑检测
│   ├── utils.py                     # 辅助工具函数（如模型保存、可视化等）
├── configs/                         # 配置文件
│   ├── config.yaml                  # 存放训练超参数、数据路径等配置
├── requirements.txt                 # Python依赖库
├── README.md                        # 项目说明文档
└── train.py                          # 模型训练入口
```

### **目录结构说明**：
- **data/**：存放所有数据文件，包括原始EEG数据、预处理后的数据、情绪评分以及标签。
  - **raw_data/**：包含EEG信号的原始数据文件（如.edf、.mat）。
  - **processed_data/**：存放经过预处理的数据（去噪、标准化等）。
  - **multimodal_data/**：存放多模态数据（如皮肤电、眼电等），以及情绪评分（SAM、HAM-A等）。
  - **labels/**：存放焦虑等级的标签，供训练时使用。

- **notebooks/**：包含Jupyter笔记本文件，用于数据分析、特征提取、模型训练等过程。
  - **data_preprocessing.ipynb**：数据预处理的详细步骤和可视化。
  - **feature_extraction.ipynb**：如何从EEG信号中提取时域、频域、时频域特征，以及如何处理情绪评分等数据。
  - **anxienet_model.ipynb**：AnxieNet模型的训练、调优和评估过程。

- **src/**：存放源代码文件。
  - **data_preprocessing.py**：进行数据预处理，包括去噪、标准化和特征提取。
  - **feature_extraction.py**：提取时域、频域、时频域的EEG信号特征。
  - **anxienet_model.py**：实现AnxieNet模型，包括Transformer结构、强化学习优化等。
  - **trainer.py**：训练过程的脚本，负责模型训练流程。


## 安装依赖

### 1. 克隆项目

```bash
git clone https://github.com/jaylinwong/AnxieNet.git
cd AnxieNet
```

### 2. 创建虚拟环境并安装依赖

```bash
# 使用 Python 3.7+ 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # 对于Windows系统，使用 `venv\Scripts\activate`

# 安装项目依赖
pip install -r requirements.txt
```

## 配置文件说明

`config.yaml` 是该项目的配置文件，包含了训练的超参数、数据路径和设备设置等。

### 示例配置：

```yaml
# config.yaml

# 数据集配置
data:
  raw_data_path: "/path/to/raw/data"
  processed_data_path: "/path/to/processed/data"
  multimodal_data_path: "/path/to/multimodal/data"
  labels_path: "/path/to/labels"
  output_dir: "./output"

# 模型配置
model:
  input_dim: 256
  hidden_dim: 128
  output_dim: 2
  num_layers: 6
  num_heads: 8

# 训练配置
training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  loss_function: "cross_entropy"
  
# 设备配置
device:
  device_type: "cuda"

# 日志和保存配置
logging:
  save_model: true
  model_save_path: "./output/model.pth"
  save_training_history: true
  training_history_path: "./output/training_history.png"
```

## 训练流程

1. **准备数据**：
   - 确保原始数据已经准备好，并将数据路径配置在 `config.yaml` 中。
   - 数据应包括 EEG 信号和对应的标签，最好提供多模态数据（如皮肤电、眼电等）。

2. **训练模型**：
   - 使用 `train.py` 脚本进行训练：
   
   ```bash
   python train.py
   ```

   该脚本将根据 `config.yaml` 中的配置自动加载数据、初始化模型、训练模型并保存训练好的模型。

   **训练过程**：
   - 训练将自动进行指定轮数（默认为 20 轮），每一轮训练后会显示训练损失和奖励。
   - 训练完成后，模型将保存在 `output/` 目录中，并保存训练历史图像。

3. **查看训练历史和结果**：
   - 训练过程中，会自动生成训练损失与奖励的可视化图表，保存在 `output/training_history.png` 中。
   - 混淆矩阵和其他评估结果可以通过修改 `train.py` 进一步扩展和保存。

4. **测试模型**：
   - 测试模型在测试集上的表现，输出准确度和 F1 分数。

## 示例输出

```
Starting training...
Epoch 1/20, Loss: 0.42, Reward: 0.85
Epoch 2/20, Loss: 0.35, Reward: 0.90
...
Test Accuracy: 85.30%
```

## 模型保存与加载

- **保存模型**：在训练完成后，模型会被保存在 `output/model.pth` 中。你可以使用以下方法加载模型：
  
  ```python
  model = TransformerModel(input_dim=256, hidden_dim=128, output_dim=2)
  model = load_model(model, './output/model.pth')
  ```

## 结果可视化

使用 `utils.py` 中的 `plot_training_history()` 和 `plot_confusion_matrix()` 函数可以方便地可视化训练过程中的损失、奖励以及模型的分类性能（混淆矩阵）。

### 混淆矩阵示例：

```python
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'], save_path='output/confusion_matrix.png')
```

## 项目贡献

如果你希望为本项目贡献代码，请遵循以下步骤：

1. Fork 本项目。
2. 创建自己的分支：`git checkout -b feature/your-feature`.
3. 提交你的更改：`git commit -am 'Add new feature'`.
4. 推送到远程分支：`git push origin feature/your-feature`.
5. 创建拉取请求（PR）。

## 许可证

本项目使用 [MIT 许可证](https://opensource.org/licenses/MIT)。

---

感谢你使用 AnxieNet！如果你有任何问题或建议，欢迎提 issue 或直接联系我。
```

### **文档说明**：

1. **项目介绍**：简要介绍了 `AnxieNet` 模型的背景和功能。
2. **安装依赖**：指导用户如何安装项目所需的依赖，并说明如何配置虚拟环境。
3. **配置文件说明**：详细介绍了 `config.yaml` 配置文件的各个字段，确保用户能够根据需求调整配置。
4. **训练流程**：提供了清晰的步骤，指导用户如何准备数据、训练模型、查看训练结果和测试模型。
5. **示例输出**：提供了训练输出的示例，帮助用户理解模型训练的过程。
6. **模型保存与加载**：说明如何保存和加载训练好的模型。
7. **贡献与许可证**：提供了项目贡献的指导，并声明了使用的许可证类型。







