import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        """
        HDF5 formatındaki veriyi okuyacak PyTorch Dataset sınıfı.
        Tüm veri seti float16 tensörleri halinde HDF5'te bulunur.
        Disk okuma darboğazını önlemek için __getitem__ içerisinde thread-local dosya işleyicisi (h5py) kullanılır.
        """
        self.h5_path = h5_path
        self.transform = transform
        
        # Etiketleri ve veri seti boyutunu doğrudan RAM'e al (küçük oldukları için)
        with h5py.File(self.h5_path, 'r') as f:
            self.labels = f['labels'][:]
            self.num_samples = f['signals'].shape[0]
            
        # Dataloader worker'ları için hdf5 objesini None başlat
        self.h5_file = None

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Multiprocessing DataLoader ile kullanıldığında h5py dosyasının worker özelinde açılması gerekir
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        signal = self.h5_file['signals'][idx] # [12, 2500] float16 array
        label = self.labels[idx]              # [5] array (NORMAL, AFIB, AFL, RBBB, LBBB)
        
        # Sinyal float32'ye model eğitimi öncesi çevrilir (mixed precision için model içi ayarlanabilir)
        signal = signal.astype(np.float32)
        
        if self.transform:
            signal = self.transform(signal)
            
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
