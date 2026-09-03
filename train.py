import os
import math
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import numpy as np

from dataset import ECGDataset
from model import HybridECGModel
from loss import AsymmetricLoss

# ─────────────────────────────────────────────────────────
# Yardımcı: Cosine + Warmup LR Scheduler
# ─────────────────────────────────────────────────────────
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────
# Tek Epoch Eğitimi
# ─────────────────────────────────────────────────────────
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ─────────────────────────────────────────────────────────
# Validation / Değerlendirme
# ─────────────────────────────────────────────────────────
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = torch.sigmoid(model(inputs)).cpu().numpy()
            all_preds.append((outputs >= threshold).astype(int))
            all_labels.append(targets.numpy().astype(int))

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    return macro_f1, micro_f1


# ─────────────────────────────────────────────────────────
# Eğitim Döngüsü (Genel)
# ─────────────────────────────────────────────────────────
def run_training(h5_path, checkpoint_path, epochs=80, batch_size=64,
                 lr=1e-3, val_split=0.1, resume_from=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Dataset + split
    full_dataset = ECGDataset(h5_path)
    n_val   = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Eğitim: {n_train} | Validasyon: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model     = HybridECGModel(num_classes=5).to(device)
    criterion = AsymmetricLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps  = epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP Scaler (sadece GPU'da)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    best_macro_f1 = 0.0

    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_macro_f1 = ckpt.get('best_macro_f1', 0.0)
        print(f"Checkpoint yüklendi: {resume_from} (epoch {start_epoch})")

    print("=" * 60)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, scheduler, device, scaler)
        macro_f1, micro_f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:>3}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Macro-F1: {macro_f1:.4f} | "
              f"Micro-F1: {micro_f1:.4f} | "
              f"{elapsed:.0f}s")

        # En iyi model checkpoint
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save({
                'epoch':         epoch + 1,
                'model':         model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'best_macro_f1': best_macro_f1
            }, checkpoint_path)
            print(f"  ✔ Yeni en iyi model kaydedildi: {checkpoint_path} (Macro-F1={best_macro_f1:.4f})")

    print("=" * 60)
    print(f"Eğitim tamamlandı. En iyi Macro-F1: {best_macro_f1:.4f}")
    return checkpoint_path


# ─────────────────────────────────────────────────────────
# Finetuning (önceki checkpoint'ten başlar)
# ─────────────────────────────────────────────────────────
def run_finetuning(h5_path, pretrained_checkpoint, finetune_checkpoint,
                   epochs=30, batch_size=32, lr=2e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FINETUNING] Cihaz: {device}")

    full_dataset = ECGDataset(h5_path)
    n_val   = max(1, int(len(full_dataset) * 0.1))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Finetuning Eğitim: {n_train} | Validasyon: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = HybridECGModel(num_classes=5).to(device)

    # Pretrained ağırlıkları yükle
    if os.path.exists(pretrained_checkpoint):
        ckpt = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Pretrained ağırlıklar yüklendi: {pretrained_checkpoint}")
    else:
        print(f"UYARI: Pretrained checkpoint bulunamadı ({pretrained_checkpoint}), sıfırdan başlanıyor.")

    criterion = AsymmetricLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps  = epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    best_macro_f1 = 0.0

    print("=" * 60)
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, scheduler, device, scaler)
        macro_f1, micro_f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"[FT] Epoch {epoch+1:>3}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Macro-F1: {macro_f1:.4f} | "
              f"Micro-F1: {micro_f1:.4f} | "
              f"{elapsed:.0f}s")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save({
                'epoch':         epoch + 1,
                'model':         model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'best_macro_f1': best_macro_f1
            }, finetune_checkpoint)
            print(f"  ✔ Yeni en iyi model (FT): {finetune_checkpoint} (Macro-F1={best_macro_f1:.4f})")

    print("=" * 60)
    print(f"Finetuning tamamlandı. En iyi Macro-F1: {best_macro_f1:.4f}")


# ─────────────────────────────────────────────────────────
# Ana Giriş Noktası
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── ADIM 1: Pretraining (Chapman + Georgia + PTB-XL) ──
    run_training(
        h5_path          = "pretrain_cache.h5",
        checkpoint_path  = "best_pretrain.pth",
        epochs           = 80,
        batch_size       = 64,
        lr               = 1e-3,
    )

    # ── ADIM 2: Finetuning (TEKNOFEST 2026 veri seti) ──
    run_finetuning(
        h5_path               = "finetune_cache.h5",
        pretrained_checkpoint = "best_pretrain.pth",
        finetune_checkpoint   = "best_finetune.pth",
        epochs                = 30,
        batch_size            = 32,
        lr                    = 2e-4,
    )
