import os
import glob
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from preprocess import preprocess_ecg
# Not: wfdb kütüphanesinin (pip install wfdb) kurulu olması gerekmektedir.
try:
    import wfdb
except ImportError:
    print("Lütfen wfdb kütüphanesini kurun: pip install wfdb")

# Hedef Sınıflar (One-Hot Encoding Sırası: NORMAL, AFIB, AFL, RBBB, LBBB)
TARGET_CLASSES = ['NORMAL', 'AFIB', 'AFL', 'RBBB', 'LBBB']

def map_labels(diagnoses):
    """
    Kayıttaki tanıları (diagnoses) hedef sınıflara (NORMAL, AFIB, AFL, RBBB, LBBB) dönüştürür.
    Sinyalde hedef sınıflardan hiçbiri yoksa [0, 0, 0, 0, 0] döndürür (Bilinmeyen / Unknown sınıf mantığı).
    Çoklu etiket (Multi-label) desteklenir.
    """
    label_vector = np.zeros(len(TARGET_CLASSES), dtype=np.float32)
    # TODO: SNOMED-CT veya doğrudan etiket ismine göre özel eşleştirme (mapping) 
    # mantığı veri setlerine göre detaylandırılacaktır.
    # Örnek eşleştirme:
    for dx in diagnoses:
        dx_upper = str(dx).upper()
        if 'NORM' in dx_upper or 'SR' in dx_upper: # Sinus Rhythm
            label_vector[0] = 1.0
        elif 'AFIB' in dx_upper:
            label_vector[1] = 1.0
        elif 'AFL' in dx_upper:
            label_vector[2] = 1.0
        elif 'RBBB' in dx_upper or 'CRBBB' in dx_upper:
            label_vector[3] = 1.0
        elif 'LBBB' in dx_upper or 'CLBBB' in dx_upper:
            label_vector[4] = 1.0
            
    return label_vector

def build_hdf5(dataset_paths, output_h5_path):
    """
    Belirtilen veri setlerindeki EKG kayıtlarını okur, ön işleme sokar ve float16 HDF5 formatında kaydeder.
    """
    all_signals = []
    all_labels = []
    
    for ds_path in dataset_paths:
        print(f"İşleniyor: {ds_path}")
        # .hea uzantılı dosyaları bul (wfdb başlık dosyaları)
        hea_files = glob.glob(os.path.join(ds_path, '**', '*.hea'), recursive=True)
        
        for hea_file in hea_files:
            if "median_beats" in hea_file or "records100" in hea_file:
                # PTB-XL için median_beats ve 100Hz versiyonları atla (tekrarları ve geçersiz formatları önlemek için)
                continue
            record_name = hea_file.replace('.hea', '')
            try:
                # wfdb ile oku (.mat veya .dat destekler)
                record = wfdb.rdrecord(record_name)
                signal = record.p_signal.T # [12, seq_len]
                fs = record.fs
                
                # Etiketleri çek (comments kısmında genellikle "Dx: ..." formatında bulunur)
                diagnoses = []
                for comment in record.comments:
                    if comment.startswith('Dx:'):
                        diagnoses.extend(comment.split(':')[1].strip().split(','))
                        
            except Exception as e:
                # Georgia veri seti gibi wfdb'nin okuyamadığı bozuk başlıklı (.hea) dosyalar için manuel okuma (Fallback)
                try:
                    mat_data = sio.loadmat(record_name + '.mat')
                    signal = mat_data['val'] # PhysioNet standart mat key'i
                    diagnoses = []
                    fs = 500 # Standart varsayılan
                    with open(hea_file, 'r') as hf:
                        lines = hf.readlines()
                        if len(lines) > 0:
                            parts = lines[0].strip().split()
                            if len(parts) >= 3:
                                fs = float(parts[2])
                        for line in lines:
                            if line.startswith('#Dx:') or line.startswith('# Dx:'):
                                dx_str = line.split(':')[1].strip()
                                diagnoses.extend(dx_str.split(','))
                except Exception as fallback_e:
                    print(f"Hata - Okunamadı (Fallback de başarısız): {record_name} | {fallback_e}")
                    continue
                
            # Etiketleri hedef vektöre eşleştir
            label_vec = map_labels(diagnoses)
            
            # Sinyali ön işleme sok
            processed_sig = preprocess_ecg(signal, fs=fs, target_fs=250)
            
            # Boyut kontrolü (Sadece tam 12 derivasyonlu ve 2500 uzunluklu sinyalleri kabul et)
            if processed_sig.shape != (12, 2500):
                print(f"Atlanıyor (Hatalı Boyut {processed_sig.shape}): {record_name}")
                continue
            
            all_signals.append(processed_sig)
            all_labels.append(label_vec)
                
    # Listeleri numpy dizisine çevir
    print("Numpy dizisine dönüştürülüyor...")
    np_signals = np.stack(all_signals) # [N, 12, 2500] float16
    np_labels = np.stack(all_labels)   # [N, 5] float32
    
    # HDF5'e yaz
    print(f"HDF5 dosyası oluşturuluyor: {output_h5_path}")
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('signals', data=np_signals, compression="gzip", compression_opts=1)
        f.create_dataset('labels', data=np_labels, compression="gzip", compression_opts=1)
        
    print("Tamamlandı.")

if __name__ == "__main__":
    # Pretraining veri setleri (Şartnamedeki yollar)
    pretrain_paths = [
        "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
        "Georgia",
        "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
    ]
    
    # Finetuning veri seti
    finetune_paths = [
        "teknofest_ECG_2026_dataset"
    ]
    
    # Gerçek veri işleme süreci çok uzun süreceği için doğrudan çalıştırıldığında aktif olmaz,
    # Kullanımı şu şekildedir:
    build_hdf5(pretrain_paths, "pretrain_cache.h5")
    build_hdf5(finetune_paths, "finetune_cache.h5")
