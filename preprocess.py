import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

def apply_bandpass_filter(signal, lowcut=0.5, highcut=48.0, fs=250, order=4):
    """Klinik EKG bandı + yüksek frekanslı QRS detayı korunur (0.5 - 48 Hz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal, axis=-1)
    return y

def apply_notch_filter(signal, freq=50.0, q=30.0, fs=250):
    """Şebeke gürültüsü filtresi (50 Hz)."""
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, q)
    y = filtfilt(b, a, signal, axis=-1)
    return y

def z_score_normalize(signal):
    """Derivasyon başına z-score normalizasyon."""
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    # 10^-6 altı standart sapma (ölü kanal) koruması
    std[std < 1e-6] = 1.0 
    return (signal - mean) / std

def standardize_length(signal, target_length=2500):
    """Kısa kayıt -> sıfır pad, uzun kayıt -> merkez kırp"""
    _, seq_len = signal.shape
    if seq_len == target_length:
        return signal
    elif seq_len < target_length:
        pad_len = target_length - seq_len
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        return np.pad(signal, ((0,0), (pad_left, pad_right)), 'constant')
    else:
        start = (seq_len - target_length) // 2
        return signal[:, start:start+target_length]

def preprocess_ecg(signal, fs=500, target_fs=250):
    """Tüm ön işleme adımlarını uygular."""
    # Sinyalde inf veya nan varsa 0 ile doldur
    signal = np.nan_to_num(signal)
    
    # Eğer örnekleme hızı farklıysa hedef hıza (250Hz) düşür/çıkar
    if fs != target_fs:
        num_samples = int(signal.shape[-1] * (target_fs / fs))
        signal = resample(signal, num_samples, axis=-1)
    
    # 1. Bandpass Filtre
    sig_filtered = apply_bandpass_filter(signal, fs=target_fs)
    # 2. Notch Filtre
    sig_filtered = apply_notch_filter(sig_filtered, fs=target_fs)
    # 3. Z-Score Normalizasyonu
    sig_norm = z_score_normalize(sig_filtered)
    # 4. Uzunluk standardizasyonu (2500 nokta)
    sig_final = standardize_length(sig_norm, target_length=2500)
    
    # RAM'de az yer kaplaması için HDF5'e yazmadan float16'ya çevir
    return sig_final.astype(np.float16)
