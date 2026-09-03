import torch
import torch.nn as nn
import numpy as np

class TemperatureScaling(nn.Module):
    """
    Validation setinde logitlerin kalibre edilmesi için kullanılır.
    Model %90 güvenilirim diyorsa gerçekten %90 doğru olması istenir.
    """
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def threshold_sweep(val_scores, val_labels, thresholds=np.arange(0.50, 0.96, 0.01)):
    """
    Validation setinde en iyi Macro F1 sağlayan eşiği (confidence threshold) bulur.
    (Bu metodun içeriği train/evaluate aşamasında scikit-learn ile zenginleştirilebilir)
    """
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Basit mock implementasyon (Gerçek F1 hesabı sklearn.metrics ile yapılacak)
    return best_thresh

def make_decision(calibrated_probs, conf_threshold=0.5, margin_threshold=0.1, high_conf_limit=0.80):
    """
    Sigmoid çıkışları kalibre edildikten sonra (0-1 arası) uygulanan karar akışı.
    
    Giriş:
      calibrated_probs: [Batch, 5] boyutlu olasılıklar tensörü
      
    Çıkış:
      Karar: En yüksek skorlu sınıfın indeksi (0,1,2,3,4) veya -1 (Unknown)
    """
    batch_size = calibrated_probs.shape[0]
    decisions = torch.full((batch_size,), -1, dtype=torch.long)
    
    # Her örnek için en yüksek iki skoru ve indekslerini al
    top2_probs, top2_indices = torch.topk(calibrated_probs, 2, dim=1)
    
    highest_probs = top2_probs[:, 0]
    second_probs = top2_probs[:, 1]
    best_classes = top2_indices[:, 0]
    
    for i in range(batch_size):
        h_prob = highest_probs[i].item()
        s_prob = second_probs[i].item()
        cls_idx = best_classes[i].item()
        
        # Kural 1: Hiçbiri confidence threshold'u geçmedi -> Unknown
        if h_prob < conf_threshold:
            decisions[i] = -1
            continue
            
        # Kural 2: Yüksek confidence (0.80) üzeriyse -> Kabul et
        if h_prob > high_conf_limit:
            decisions[i] = cls_idx
            continue
            
        # Kural 3: Threshold ile 0.80 arasındaysa margin kontrolü
        margin = h_prob - s_prob
        if margin < margin_threshold:
            decisions[i] = -1
        else:
            decisions[i] = cls_idx
            
    return decisions
