import os
from dataclasses import dataclass

import torch
import torchmetrics
import timm
from tqdm import tqdm


@dataclass
class TrainConfig:
    model_name: str = "rexnet_150"
    lr: float = 3e-4
    epochs: int = 30
    patience: int = 5
    threshold: float = 0.0
    save_dir: str = "saved_models"
    save_prefix: str = "timm_baseline"
    dev_mode: bool = False


class TrainValidation:
    def __init__(self, classes, tr_dl, val_dl, device, cfg: TrainConfig):
        self.classes = classes
        self.tr_dl = tr_dl
        self.val_dl = val_dl
        self.device = device
        self.cfg = cfg

        self.model = timm.create_model(cfg.model_name, pretrained=True, num_classes=len(classes)).to(device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(classes)).to(device)

        os.makedirs(cfg.save_dir, exist_ok=True)

        self.best_f1 = -1.0
        self.not_improved = 0

        self.tr_losses, self.val_losses = [], []
        self.tr_accs, self.val_accs = [], []
        self.tr_f1s, self.val_f1s = [], []

    @staticmethod
    def to_device(batch, device):
        ims, gts = batch
        return ims.to(device), gts.to(device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        self.f1_metric.reset()

        for idx, batch in tqdm(enumerate(self.tr_dl), total=len(self.tr_dl), desc="Training"):
            if self.cfg.dev_mode and idx >= 2:
                break

            ims, gts = self.to_device(batch, self.device)
            preds = self.model(ims)
            loss = self.loss_fn(preds, gts)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * gts.size(0)
            correct += (preds.argmax(dim=1) == gts).sum().item()
            self.f1_metric.update(preds, gts)

        n = len(self.tr_dl.dataset)
        return total_loss / n, correct / n, float(self.f1_metric.compute().item())

    def val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        self.f1_metric.reset()

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc="Validation"):
                if self.cfg.dev_mode and idx >= 2:
                    break

                ims, gts = self.to_device(batch, self.device)
                preds = self.model(ims)
                loss = self.loss_fn(preds, gts)

                total_loss += loss.item() * gts.size(0)
                correct += (preds.argmax(dim=1) == gts).sum().item()
                self.f1_metric.update(preds, gts)

        n = len(self.val_dl.dataset)
        return total_loss / n, correct / n, float(self.f1_metric.compute().item())

    def maybe_save_best(self, val_f1):
        if val_f1 > self.best_f1 + self.cfg.threshold:
            self.best_f1 = val_f1
            path = os.path.join(self.cfg.save_dir, f"{self.cfg.save_prefix}_best.pth")
            torch.save(self.model.state_dict(), path)
            self.not_improved = 0
            print(f"✅ Saved best model: F1={val_f1:.3f} -> {path}")
        else:
            self.not_improved += 1
            print(f"No improvement ({self.not_improved}/{self.cfg.patience})")

    def run(self):
        for epoch in range(self.cfg.epochs):
            if self.cfg.dev_mode and epoch >= 2:
                break

            print(f"\nEpoch {epoch+1}/{self.cfg.epochs}")
            tr_loss, tr_acc, tr_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.val_epoch()

            self.tr_losses.append(tr_loss); self.tr_accs.append(tr_acc); self.tr_f1s.append(tr_f1)
            self.val_losses.append(val_loss); self.val_accs.append(val_acc); self.val_f1s.append(val_f1)

            print(f"Train: loss={tr_loss:.3f} acc={tr_acc:.3f} f1={tr_f1:.3f}")
            print(f"Val:   loss={val_loss:.3f} acc={val_acc:.3f} f1={val_f1:.3f}")

            self.maybe_save_best(val_f1)

            if self.not_improved >= self.cfg.patience:
                print("🛑 Early stopping")
                break

        return self