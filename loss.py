import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    BCE tabanlıdır ancak pozitif ve negatif sınıflara farklı odaklanma (gamma) cezaları uygular.
    Tıbbi problemlerde (örn: LBBB'yi kaçırmanın tehlikesi) çok etkilidir.
    """
    def __init__(self, gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        
        # Sınıf ağırlıkları (genelsema.txt)
        # NORMAL, AFIB, AFL, RBBB, LBBB
        self.class_weights = torch.tensor([0.35, 1.0, 0.40, 0.95, 2.2])

    def forward(self, x, y):
        """
        x: logits (sigmoid öncesi ham tahminler), shape: [batch, num_classes]
        y: hedefler (0 veya 1), shape: [batch, num_classes]
        """
        # Cihaz ayarı (CPU/GPU)
        self.class_weights = self.class_weights.to(x.device)
        
        # Olasılıklar
        x_sigmoid = torch.sigmoid(x)
        
        # Negatif olasılıklar (1 - p)
        x_inv = 1 - x_sigmoid
        
        # Hard Negative Mining (Asymmetric Clipping)
        x_inv = torch.clamp(x_inv - self.clip, min=0.0)
        
        # Focal Loss ağırlıkları
        pos_weight = (1 - x_sigmoid) ** self.gamma_pos
        neg_weight = (x_inv) ** self.gamma_neg

        # BCE Loss (manuel hesaplama)
        pos_loss = -y * torch.log(x_sigmoid + self.eps) * pos_weight
        neg_loss = -(1 - y) * torch.log(x_inv + self.eps) * neg_weight
        
        loss = pos_loss + neg_loss
        
        # Class weights (Ağırlıkları uygula)
        loss = loss * self.class_weights.unsqueeze(0)
        
        # Batch ve Sınıf üzerinden ortalama
        return loss.mean()
