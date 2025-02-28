# src/feature_extraction.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import pywt

def hjorth_parameters(data):
    """
    提取Hjorth特征：活动性、迁移性和复杂性
    
    :param data: EEG数据
    :return: Hjorth特征（活动性，迁移性，复杂性）
    """
    # 计算活动性（Activity）
    activity = np.var(data)  # 活动性：信号的方差
    
    # 计算迁移性（Mobility）
    mobility = np.std(np.diff(data)) / np.std(data)  # 迁移性：信号的频率变化
    
    # 计算复杂性（Complexity）
    complexity = np.std(np.diff(np.diff(data))) / np.std(np.diff(data))  # 复杂性：信号的变化率
    
    return activity, mobility, complexity

def power_spectral_density(data, fs=128):
    """
    计算功率谱密度（Power Spectral Density, PSD）
    
    :param data: EEG信号数据
    :param fs: 采样频率（Hz）
    :return: f - 频率，Pxx - 功率谱密度
    """
    f, Pxx = welch(data, fs=fs, nperseg=256)
    return f, Pxx

def wavelet_transform(data, wavelet='db4', level=5):
    """
    小波变换（Wavelet Transform），用于时频域特征提取
    
    :param data: EEG信号数据
    :param wavelet: 小波类型（默认为db4）
    :param level: 分解的层数
    :return: 小波系数
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def extract_features(raw_data_clean):
    """
    从预处理后的EEG信号中提取特征：Hjorth特征，功率谱密度，和小波变换
    
    :param raw_data_clean: 预处理后的EEG信号
    :return: 提取的特征集合
    """
    features = []
    
    # 提取Hjorth特征
    activity, mobility, complexity = hjorth_parameters(raw_data_clean)
    features.append(activity)
    features.append(mobility)
    features.append(complexity)
    
    # 提取功率谱密度特征
    f, Pxx = power_spectral_density(raw_data_clean)
    features.append(np.mean(Pxx))  # 功率谱的均值
    
    # 提取小波变换特征
    coeffs = wavelet_transform(raw_data_clean)
    features.append(np.mean(coeffs[0]))  # 小波变换的近似系数均值
    
    return features

def process_data(raw_data_path):
    """
    处理原始EEG信号，去噪、标准化并提取特征
    
    :param raw_data_path: 原始EEG信号数据路径
    :return: 提取的特征
    """
    # 加载数据
    raw_data = load_raw_data(raw_data_path)
    
    processed_features = []
    for data in raw_data:
        # 进行预处理
        preprocessed_data = preprocess_data(data)
        
        # 提取特征
        features = extract_features(preprocessed_data)
        
        processed_features.append(features)
    
    return np.array(processed_features)

def load_raw_data(raw_data_path):
    """
    加载原始EEG数据（假设是.mat或.edf文件）
    
    :param raw_data_path: 数据路径
    :return: 加载的EEG信号数据
    """
    # 加载数据的方式，可以是.edf或者.mat文件
    data = []
    for filename in os.listdir(raw_data_path):
        filepath = os.path.join(raw_data_path, filename)
        if filepath.endswith('.edf'):
            raw = mne.io.read_raw_edf(filepath, preload=True)
            data.append(raw)
        elif filepath.endswith('.mat'):
            mat = sio.loadmat(filepath)
            data.append(mat)
    
    return data

def preprocess_data(raw_data):
    """
    对EEG数据进行预处理（去噪、标准化等）
    
    :param raw_data: 原始EEG数据
    :return: 处理后的EEG数据
    """
    # 去伪影：应用ICA去除眼动等伪影
    ica = ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw_data)
    raw_data_clean = ica.apply(raw_data)
    
    # 标准化：进行z-score标准化
    raw_data_clean = raw_data_clean.get_data()
    scaler = StandardScaler()
    raw_data_clean = scaler.fit_transform(raw_data_clean.reshape(-1, raw_data_clean.shape[-1]))
    
    return raw_data_clean
