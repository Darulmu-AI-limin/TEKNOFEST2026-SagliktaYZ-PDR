import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

def evaluate_model(y_true, y_pred_probs, conf_threshold=0.5):
    """
    Model performansını değerlendirir. 
    y_true: [N, 5] one-hot
    y_pred_probs: [N, 5] kalibre edilmiş sigmoid çıkışları
    """
    
    # Karar eşiği ile tahminleri binarize et (basit karar mekanizması)
    y_pred = (y_pred_probs >= conf_threshold).astype(int)
    
    # 1. Macro F1 Score
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 2. Per-class AUC
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            auc = 0.5 # Tek bir sınıf örneği (örn: hepsi negatif) kalmışsa
        aucs.append(auc)
        
    # 3. Matthews Correlation Coefficient (Dengesiz sınıflar için)
    # Çoklu etiket için MCC per-class hesaplanıp ortalaması alınabilir
    mccs = []
    for i in range(y_true.shape[1]):
        mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
        mccs.append(mcc)
        
    return {
        "macro_f1": macro_f1,
        "per_class_auc": aucs,
        "mean_auc": np.mean(aucs),
        "per_class_mcc": mccs,
        "mean_mcc": np.mean(mccs)
    }

def print_evaluation_report(metrics_dict):
    print(f"Macro F1 Score : {metrics_dict['macro_f1']:.4f}")
    print(f"Mean AUC       : {metrics_dict['mean_auc']:.4f}")
    print(f"Mean MCC       : {metrics_dict['mean_mcc']:.4f}")
