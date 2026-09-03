import numpy as np

def add_gaussian_noise(signal, noise_level=0.01):
    """
    Sensör gürültüsünü simüle etmek için Gaussian (normal dağılımlı) gürültü ekler.
    noise_level, standart sapmayı (std) belirler.
    """
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def add_baseline_wander(signal, max_amplitude=0.5, fs=250):
    """
    Solunum kaynaklı düşük frekanslı taban çizgisi kaymasını (baseline wander) simüle eder.
    0.1 ile 0.5 Hz arasında rastgele bir sinüs dalgası ekler.
    """
    num_leads, seq_len = signal.shape
    t = np.arange(seq_len) / fs
    
    # 0.1 - 0.5 Hz arası rastgele frekans
    freq = np.random.uniform(0.1, 0.5)
    
    # Her derivasyona aynı veya farklı fazda eklenebilir, genelde solunum tüm sensörleri etkiler
    phase = np.random.uniform(0, 2 * np.pi)
    
    wander = max_amplitude * np.sin(2 * np.pi * freq * t + phase)
    # wander boyutu (seq_len,), bunu (12, seq_len) sinyaline ekle
    return signal + wander

def apply_time_shift(signal, max_shift_ms=500, fs=250):
    """
    Sinyali zaman ekseninde sağa veya sola kaydırır.
    Kaydırılan (boşalan) alanlar sıfır ile doldurulur (veya baştan/sondan tekrar edilebilir).
    max_shift_ms: maksimum kaydırma miktarı (milisaniye cinsinden).
    """
    num_leads, seq_len = signal.shape
    max_shift_samples = int((max_shift_ms / 1000) * fs)
    
    shift = np.random.randint(-max_shift_samples, max_shift_samples)
    
    if shift == 0:
        return signal
        
    shifted_signal = np.zeros_like(signal)
    
    if shift > 0:
        # Sağa kaydır (başı sıfır kalır)
        shifted_signal[:, shift:] = signal[:, :-shift]
    else:
        # Sola kaydır (sonu sıfır kalır)
        shift = abs(shift)
        shifted_signal[:, :-shift] = signal[:, shift:]
        
    return shifted_signal

def augment_ecg(signal, fs=250):
    """
    Train setindeki azınlık sınıflarına (AFIB, RBBB, LBBB) uygulanacak ana augmentasyon fonksiyonu.
    Gereksinimler:
    - Sinyalin morfolojisi çok bozulmamalı
    - Genlik (amplitude) değiştirilmemeli
    """
    # Her augmentasyonun belirli bir olasılıkla uygulanması daha iyi genelleme sağlar
    if np.random.rand() < 0.5:
        signal = add_gaussian_noise(signal, noise_level=np.random.uniform(0.005, 0.02))
        
    if np.random.rand() < 0.5:
        signal = add_baseline_wander(signal, max_amplitude=np.random.uniform(0.1, 0.3), fs=fs)
        
    if np.random.rand() < 0.5:
        signal = apply_time_shift(signal, max_shift_ms=300, fs=fs)
        
    return signal.astype(np.float32)
