# models/Comparison_Models/Kaggle_Model/train_timm.py
# Takes training data and teaches a pretrained ResNet18 to classify tennis action, monitor validation performance and saves best version of the model and stops early if not improving.

import os
from dataclasses import dataclass

import torch
import timm #pretrained image models.
import torchmetrics
from tqdm import tqdm


@dataclass
class TrainConfig:
    model_name: str = "resnet18" # timm model
    lr: float = 3e-4 # learning rate for optimizer
    epochs: int = 8 # # of epochs
    patience: int = 2 # early stop of val F1 does not improve after 2 epochs
    threshold: float = 0.0 # how much F1 must improve to count as "real imrpovement"
    save_dir: str = "saved_models" # save best model checkpoint and the name for it
    save_prefix: str = "timm_baseline"
    dev_mode: bool = False # if true only run a couple batches/epochs, for debugging.

class TrainValidation:
    def __init__(self, classes, tr_dl, val_dl, device, cfg: TrainConfig):
        self.classes = classes
        self.tr_dl = tr_dl
        self.val_dl = val_dl
        self.device = device
        self.cfg = cfg

        # creates pre-trained model (imagenet weights), replaces the final layer to output len(classes) (4), moves it to GPU/CPU
        self.model = timm.create_model(cfg.model_name, pretrained=True, num_classes=len(classes)).to(device)
        # loss function for multi-class classification
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # optimize all params.
        params = self.model.parameters()
        # optimizer updates model weights using the gradients
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        # creat F1 metric object, track F1 across batches. 
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=len(classes)).to(device)

        # creates the save folder if it doesn't exist, best validation F1 so far and how many epoches since it imrpoved
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.best_f1 = -1
        self.no_improve = 0

    # helper: move batch to device.
    def _move(self, batch):
        x, y = batch
        return x.to(self.device), y.to(self.device)
    
    # train enables training behavior, reset the F1 metric at start of epoch
    def train_epoch(self):
        self.model.train()
        self.f1.reset()

        # run totals for loss and number correct.
        total_loss = 0.0
        correct = 0

        # loop through training batches with progress bar.
        for i, batch in tqdm(enumerate(self.tr_dl), total=len(self.tr_dl), desc="Training"):
            if self.cfg.dev_mode and i >= 2:
                break

            x, y = self._move(batch)

            # forward pass and compute cross-entropy loss.
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            #clear old gradients before computing new ones.
            self.opt.zero_grad(set_to_none=True)
            # backprop: compute graduents and update model weights based on gradients.
            loss.backward()
            self.opt.step()

            # calculate total loss scaled by batch size and batch accuracy, update F1 metric.
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            self.f1.update(logits, y)

        # compute average loss and accuracy over all samples and final epoch F1.
        n = len(self.tr_dl.dataset)
        return total_loss / n, correct / n, float(self.f1.compute().item())

    #no_grad() disables gradient tracking, eval() eval bahvior, reset F1 agian.
    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        self.f1.reset()

        # below is same loop structure but no backward/optimizers
        total_loss = 0.0
        correct = 0

        for i, batch in tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc="Validation"):
            if self.cfg.dev_mode and i >= 2:
                break

            x, y = self._move(batch)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            self.f1.update(logits, y)

        n = len(self.val_dl.dataset)
        return total_loss / n, correct / n, float(self.f1.compute().item())

    # if validation F1 improves enough update best and reset early stop counter, save model weights to disk.
    def maybe_save(self, val_f1):
        if val_f1 > self.best_f1 + self.cfg.threshold:
            self.best_f1 = val_f1
            self.no_improve = 0
            path = os.path.join(self.cfg.save_dir, f"{self.cfg.save_prefix}_best.pth")
            torch.save(self.model.state_dict(), path)
            print(f"Saved best: f1={val_f1:.3f} -> {path}")
        else:
            # if it doesn't improve, print message.
            self.no_improve += 1
            print(f"No improvement ({self.no_improve}/{self.cfg.patience})")

    #loop through epochs, if dev_mod, only do 2 epochs otherwise print epoch number and run one trianing epoch and one validation epoch
    def run(self):
        for ep in range(self.cfg.epochs):
            if self.cfg.dev_mode and ep >= 2:
                break

            print(f"\nEpoch {ep+1}/{self.cfg.epochs}")
            tr_loss, tr_acc, tr_f1 = self.train_epoch()
            va_loss, va_acc, va_f1 = self.val_epoch()

            print(f"Train: loss={tr_loss:.3f} acc={tr_acc:.3f} f1={tr_f1:.3f}")
            print(f"Val:   loss={va_loss:.3f} acc={va_acc:.3f} f1={va_f1:.3f}")


            # save best checkpoint if improved, and stop training if val F1 hasn't improved for patience epochs.
            self.maybe_save(va_f1)

            if self.no_improve >= self.cfg.patience:
                print("Early stopping")
                break
        # return trainer object.
        return self