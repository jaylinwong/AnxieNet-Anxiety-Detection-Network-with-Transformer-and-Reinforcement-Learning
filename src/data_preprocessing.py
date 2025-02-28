# src/data_preprocessing.py

import os
import mne
import numpy as np
import scipy.io as sio
from mne.preprocessing import ICA
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
import pywt

def load_raw_data(raw_data_path):
    """
    加载原始EEG数据
    支持两种格式：.edf (MNE库处理) 和 .mat (SciPy加载)
    
    :param raw_data_path: 原始数据所在路径
    :return: EEG信号数据
    """
    data = []
    for filename in os.listdir(raw_data_path):
        filepath = os.path.join(raw_data_path, filename)
        if filepath.endswith('.edf'):
            # 使用MNE库加载edf文件
            raw = mne.io.read_raw_edf(filepath, preload=True)
            data.append(raw)
        elif filepath.endswith('.mat'):
            # 使用SciPy加载mat文件
            mat = sio.loadmat(filepath)
            data.append(mat)
    
    return data

def preprocess_data(raw_data):
    """
    对EEG数据进行去噪和标准化处理
    
    :param raw_data: 原始EEG数据
    :return: 预处理后的EEG数据
    """
    # 去伪影：应用ICA去眼动伪影
    ica = ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw_data)
    raw_data_clean = ica.apply(raw_data)
    
    # 标准化：将EEG数据进行z-score标准化
    raw_data_clean = raw_data_clean.get_data()
    scaler = StandardScaler()
    raw_data_clean = scaler.fit_transform(raw_data_clean.reshape(-1, raw_data_clean.shape[-1]))
    
    return raw_data_clean

def hjorth_parameters(data):
    """
    提取Hjorth特征：活动性、迁移性和复杂性
    
    :param data: EEG数据
    :return: Hjorth特征
    """
    # 计算活动性、迁移性和复杂性
    activity = np.var(data)  # 活动性：信号的方差
    mobility = np.std(np.diff(data)) / np.std(data)  # 迁移性：信号的频率变化
    complexity = np.std(np.diff(np.diff(data))) / np.std(np.diff(data))  # 复杂性：信号的变化率
    return activity, mobility, complexity

def power_spectral_density(data, fs=128):
    """
    计算功率谱密度（PSD）
    
    :param data: EEG数据
    :param fs: 采样频率
    :return: 频谱与功率
    """
    f, Pxx = welch(data, fs=fs, nperseg=256)
    return f, Pxx

def wavelet_transform(data, wavelet='db4', level=5):
    """
    小波变换提取特征
    
    :param data: EEG数据
    :param wavelet: 小波类型
    :param level: 分解层数
    :return: 小波系数
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def extract_features(raw_data_clean):
    """
    从预处理后的EEG数据中提取特征：Hjorth特征、功率谱密度、以及小波变换
    
    :param raw_data_clean: 预处理后的EEG数据
    :return: 提取的特征集合
    """
    features = []
    
    # Hjorth特征
    activity, mobility, complexity = hjorth_parameters(raw_data_clean)
    features.append(activity)
    features.append(mobility)
    features.append(complexity)
    
    # 功率谱密度
    f, Pxx = power_spectral_density(raw_data_clean)
    features.append(np.mean(Pxx))  # 功率谱均值
    
    # 小波变换
    coeffs = wavelet_transform(raw_data_clean)
    features.append(np.mean(coeffs[0]))  # 小波变换的近似系数均值
    
    return features

def process_data(raw_data_path):
    """
    处理原始EEG数据，去噪、标准化并提取特征
    
    :param raw_data_path: 原始EEG数据路径
    :return: 处理后的特征数据
    """
    raw_data = load_raw_data(raw_data_path)
    
    # 预处理并提取特征
    processed_features = []
    for data in raw_data:
        preprocessed_data = preprocess_data(data)
        features = extract_features(preprocessed_data)
        processed_features.append(features)
    
    return np.array(processed_features)
