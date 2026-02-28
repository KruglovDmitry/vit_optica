import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import os
from torch.utils.tensorboard import SummaryWriter  # опционально

# Вспомогательные функции для CutMix и MixUp
def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

class Trainer:
    def __init__(self, model, optimizer, loss_fn, exp_name, device,
                 scheduler=None, clip_grad_norm=1.0, use_cutmix=False, use_mixup=False,
                 mixup_alpha=1.0, cutmix_alpha=1.0, writer=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.writer = writer or SummaryWriter(f'wd/runs/{exp_name}')  # для TensorBoard

        # Нельзя одновременно использовать CutMix и MixUp
        if use_cutmix and use_mixup:
            raise ValueError("Choose either CutMix or MixUp, not both.")

    def train(self, trainloader, testloader, epochs, warmup_epochs=10,
              save_model_every_n_epochs=10, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)

        train_losses, test_losses, accuracies = [], [], []
        best_acc = 0.0

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(trainloader, epoch)

            # Evaluate
            accuracy, test_loss = self.evaluate(testloader)

            # Log
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/test', accuracy, epoch)
            if self.scheduler:
                self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % save_model_every_n_epochs == 0:
                self.save_checkpoint(os.path.join(save_dir, f'{self.exp_name}_epoch_{epoch+1}.pth'),
                                     epoch, accuracy, test_loss)
                if accuracy > best_acc:
                    best_acc = accuracy
                    self.save_checkpoint(os.path.join(save_dir, f'{self.exp_name}_best.pth'),
                                         epoch, accuracy, test_loss, is_best=True)

        # Save final model
        self.save_checkpoint(os.path.join(save_dir, f'{self.exp_name}_final.pth'),
                             epochs-1, accuracy, test_loss)
        print(f"Best accuracy: {max(accuracies):.4f}")
        self.writer.close()
        return train_losses, test_losses, accuracies

    def train_epoch(self, trainloader, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Apply CutMix or MixUp with 50% probability (common practice)
            apply_aug = np.random.rand() < 0.5
            if self.use_cutmix and apply_aug:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=self.cutmix_alpha)
                loss = lam * self.loss_fn(self.model(images)[0], labels_a) + \
                       (1 - lam) * self.loss_fn(self.model(images)[0], labels_b)
            elif self.use_mixup and apply_aug:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=self.mixup_alpha)
                loss = lam * self.loss_fn(self.model(images)[0], labels_a) + \
                       (1 - lam) * self.loss_fn(self.model(images)[0], labels_b)
            else:
                logits, _ = self.model(images)
                loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            total_loss += loss.item() * len(images)

            # Log per-batch loss to TensorBoard (optional, can be heavy)
            # if batch_idx % 100 == 0:
            #     self.writer.add_scalar('Loss/batch', loss.item(), epoch * len(trainloader) + batch_idx)

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        for images, labels in testloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * len(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss

    def save_checkpoint(self, path, epoch, accuracy, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'loss': loss,
            'config': self.model.config
        }
        torch.save(checkpoint, path)
        if is_best:
            print(f"Best model saved to {path}")
        else:
            print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}, epoch {checkpoint['epoch']}, acc {checkpoint['accuracy']:.4f}")
        return checkpoint['epoch']