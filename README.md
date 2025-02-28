Here’s the English version of the README document for your AnxieNet project:

---

# AnxieNet: A Transformer-based Multi-modal Anxiety Detection Model

## Project Introduction

AnxieNet is a Transformer-based multi-modal anxiety detection model designed to detect anxiety through EEG signals and other physiological signals (such as skin conductance, eye movement, etc.). The project combines the Transformer architecture from deep learning and reinforcement learning optimization techniques, enabling the automatic extraction of features from EEG signals and training for improved anxiety detection accuracy.

The model is implemented using the PyTorch framework and includes various data preprocessing and feature extraction techniques (such as ICA denoising, Hjorth features, power spectral density, time-frequency domain features, etc.). Reinforcement learning is employed during training to optimize the model and enhance prediction performance.

## Key Features

- Transformer-based multi-modal emotion analysis model
- Data preprocessing and feature extraction (including time-domain, frequency-domain, and time-frequency-domain features)
- Reinforcement learning optimization mechanism to improve model training effectiveness
- Complete workflow supporting model training, validation, and testing
- Training history and result visualization, including loss, reward, and confusion matrix
- Model saving and loading support

### **Innovative Design Highlights**:
- **Multi-modal Input Fusion**: The model uses not only EEG signals but also combines participants' emotional ratings (SAM, HAM-A scales) and other physiological signals (such as skin conductance, eye movement) to improve anxiety detection accuracy by deep fusion of multi-modal inputs.
  
- **Adaptive Attention Mechanism**: Based on Transformer, the adaptive attention mechanism assigns weights to EEG signals and emotional ratings, automatically identifying the importance of each input feature.

- **Temporal Feature Learning and Dynamic Anxiety Modeling**: Using the self-attention mechanism of Transformer, the model learns temporal dependencies and captures dynamic anxiety characteristics over time, allowing it to model changes in anxiety states and identify trends in anxiety levels.

- **Reinforcement Learning-based Anxiety Detection Optimization**: Reinforcement learning (RL) is applied to optimize anxiety detection decisions, allowing the model to adapt to individual variations and continuously improve its anxiety level predictions over multiple feedback cycles.

### **Model Name**:
**AnxieNet**: Anxiety Detection Network with Transformer and Reinforcement Learning.

### **Detailed Explanation of Innovative Design Highlights**:
1. **Multi-modal Input Fusion**:
   - Combines EEG signals with other physiological signals (e.g., skin conductance, eye movement) and psychological ratings (e.g., SAM, HAM-A), training end-to-end with deep neural networks to improve anxiety detection accuracy.
   - By designing a **multi-modal fusion layer**, the model enables information exchange across multiple input sources, capturing more anxiety-related information.

2. **Adaptive Attention Mechanism**:
   - The **adaptive attention mechanism** based on Transformer can automatically adjust the weight of each input signal based on the characteristics of the data.
   - This mechanism helps the model better identify and process key components of EEG signals that are related to anxiety.

3. **Temporal Feature Learning**:
   - Through the **self-attention mechanism of Transformer**, the model captures temporal dependencies in EEG signals and learns the dynamic changes in anxiety.
   - This modeling approach helps predict the trends in anxiety levels, allowing the model to predict short-term anxiety changes.

4. **Reinforcement Learning Optimization**:
   - Using **reinforcement learning**, the model continuously learns from feedback, improving its ability to accurately classify anxiety states.
   - RL considers not only the current anxiety state but also adapts to individual variations, providing personalized anxiety detection capabilities.

## Project Structure

```plaintext
AnxieNet/
├── data/
│   ├── raw_data/                    # Raw data files (.edf, .mat)
│   ├── processed_data/              # Preprocessed data files (.mat)
│   ├── multimodal_data/             # Emotion ratings, skin conductance, and other physiological signals
│   ├── labels/                      # Anxiety level labels (SAM, HAM-A scores)
├── notebooks/                       # Jupyter notebooks
│   ├── data_preprocessing.ipynb      # Data preprocessing workflow
│   ├── feature_extraction.ipynb      # Feature extraction and analysis
│   ├── anxienet_model.ipynb         # AnxieNet model training and evaluation
├── src/
│   ├── data_preprocessing.py         # Data preprocessing scripts (denoising, normalization, etc.)
│   ├── feature_extraction.py         # Feature extraction scripts (time-domain, frequency-domain, time-frequency-domain features)
│   ├── anxienet_model.py            # AnxieNet model implementation (Transformer + Reinforcement Learning)
│   ├── trainer.py                   # Training and evaluation scripts
│   ├── reinforcement_learning.py     # Reinforcement learning section for anxiety detection optimization
│   ├── utils.py                     # Utility functions (e.g., model saving, visualization)
├── configs/                         # Configuration files
│   ├── config.yaml                  # Configuration for training hyperparameters, data paths, etc.
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── train.py                          # Model training entry point
```

### **Directory Structure Explanation**:
- **data/**: Contains all data files, including raw EEG data, preprocessed data, emotion ratings, and labels.
  - **raw_data/**: Contains raw EEG signal data files (e.g., .edf, .mat).
  - **processed_data/**: Contains preprocessed data (denoised, normalized, etc.).
  - **multimodal_data/**: Contains multi-modal data (e.g., skin conductance, eye movement) and emotion ratings (SAM, HAM-A, etc.).
  - **labels/**: Contains the anxiety level labels for training.

- **notebooks/**: Contains Jupyter notebooks for data analysis, feature extraction, and model training.
  - **data_preprocessing.ipynb**: Steps for data preprocessing and visualization.
  - **feature_extraction.ipynb**: How to extract time-domain, frequency-domain, and time-frequency-domain features from EEG signals and process emotion ratings.
  - **anxienet_model.ipynb**: Training, tuning, and evaluating the AnxieNet model.

- **src/**: Contains source code files.
  - **data_preprocessing.py**: Handles data preprocessing, including denoising, normalization, and feature extraction.
  - **feature_extraction.py**: Extracts EEG signal features in time-domain, frequency-domain, and time-frequency-domain.
  - **anxienet_model.py**: Implements the AnxieNet model, including the Transformer architecture and reinforcement learning optimization.
  - **trainer.py**: Handles the training process of the model.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jaylinwong/AnxieNet.git
cd AnxieNet
```

### 2. Create Virtual Environment and Install Dependencies

```bash
# Create a virtual environment using Python 3.7+
python3 -m venv venv
source venv/bin/activate  # For Windows, use `venv\Scripts\activate`

# Install the project dependencies
pip install -r requirements.txt
```

## Configuration File Explanation

The `config.yaml` file contains training hyperparameters, data paths, and device settings.

### Sample Configuration:

```yaml
# config.yaml

# Dataset configuration
data:
  raw_data_path: "/path/to/raw/data"
  processed_data_path: "/path/to/processed/data"
  multimodal_data_path: "/path/to/multimodal/data"
  labels_path: "/path/to/labels"
  output_dir: "./output"

# Model configuration
model:
  input_dim: 256
  hidden_dim: 128
  output_dim: 2
  num_layers: 6
  num_heads: 8

# Training configuration
training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  loss_function: "cross_entropy"
  
# Device configuration
device:
  device_type: "cuda"

# Logging and saving configuration
logging:
  save_model: true
  model_save_path: "./output/model.pth"
  save_training_history: true
  training_history_path: "./output/training_history.png"
```

## Training Process

1. **Prepare the Data**:
   - Ensure that raw data is ready and paths are configured in `config.yaml`.
   - The data should include EEG signals and corresponding labels, and ideally, multi-modal data (e.g., skin conductance, eye movement).

2. **Train the Model**:
   - Use the `train.py` script to train the model:
   
   ```bash
   python train.py
   ```

   This script will automatically load data, initialize the model, train it, and save the trained model based on the configuration in `config.yaml`.

   **Training Process**:
   - The training will run for the specified number of epochs (default is 20), showing training loss and reward after each epoch.
   - After training, the model will be saved in the `output/` directory, along with the training history.

3. **View Training History and Results**:
   - Training loss and reward will be visualized in a graph saved to `output/training_history.png`.
   - Confusion matrix and other evaluation results can be further extended and saved by modifying `train.py`.

4. **Test the Model**:
   - Test the model’s performance on the test set and output accuracy and F1 score.

## Model Saving and Loading

- **Save Model**: After training, the model will be saved in `output/model.pth`. You can load the model with the following code:
  
  ```python
  model = TransformerModel(input_dim=256, hidden_dim=128, output_dim=2)
  model = load_model(model, './output/model.pth')
  ```

## Result Visualization

Use `utils.py`’s `plot_training_history()` and `plot_confusion_matrix()` functions to easily visualize training losses, rewards, and model classification performance (confusion matrix).

### Confusion Matrix Example:

```python
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'], save_path='output/confusion_matrix.png')
```

Sure! Based on the information you've provided, here’s how you can write the **Contributing** section for your README:

---

## Contributing

Thank you for your interest in contributing to this project! If you would like to help improve the multi-modal anxiety detection model, please follow these steps:

### 1. Fork the Repository
Start by forking the repository to your GitHub account. This will allow you to make changes without affecting the main codebase directly.

### 2. Create a Feature Branch
Create a new branch for your feature or bugfix. You can do this with the following Git command:

```bash
git checkout -b feature/your-feature
```

Replace `your-feature` with a descriptive name for your branch.

### 3. Commit Your Changes
Make your changes, and when you're ready, commit them with a clear message:

```bash
git commit -am 'Add new feature: multi-modal anxiety detection model'
```

Make sure to include a description of what your changes address. If you're adding a model based on the DASPS dataset for multi-modal anxiety detection, mention it here.

### 4. Push Your Changes
Push your changes to your forked repository:

```bash
git push origin feature/your-feature
```

### 5. Open a Pull Request
Once you've pushed your changes, go to your forked repository on GitHub and open a pull request (PR) to merge your feature branch into the main project.

--- 

This section allows anyone interested in contributing to understand your current work, the direction you’re heading, and how they can help. Let me know if you'd like to modify any part!
## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

This citation references the DASPS dataset as presented in the paper "DASPS: A Database for Anxious States based on a Psychological Stimulation". You can adjust the format depending on the citation style you are using, but this should cover the necessary details.

---

Thank you for using AnxieNet! If you have any questions or suggestions, feel free to open an issue or contact me directly.

--- 
