import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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
                 mixup_alpha=1.0, cutmix_alpha=1.0, writer=None,
                 save_attention_every_n_epochs=10, num_attention_samples=8):
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
        self.writer = writer or SummaryWriter(f'./runs/{exp_name}')  # для TensorBoard
        self.save_attention_every_n_epochs = save_attention_every_n_epochs
        self.num_attention_samples = num_attention_samples
        self.fixed_images = None   # будут инициализированы позже
        self.fixed_labels = None

        # Нельзя одновременно использовать CutMix и MixUp
        if use_cutmix and use_mixup:
            raise ValueError("Choose either CutMix or MixUp, not both.")

    def set_fixed_attention_samples(self, dataloader):
        """Берём первые несколько изображений из даталоадера для визуализации внимания"""
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        self.fixed_images = images[:self.num_attention_samples].to(self.device)
        self.fixed_labels = labels[:self.num_attention_samples].to(self.device)
        print(f"Fixed {self.num_attention_samples} images for attention visualization.")

    def train(self, trainloader, testloader, epochs, warmup_epochs=10,
              save_model_every_n_epochs=10, save_dir='checkpoints'):
        
        os.makedirs(save_dir, exist_ok=True)
        self.set_fixed_attention_samples(testloader)

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
                    
            if epoch % self.save_attention_every_n_epochs == 0 or epoch == epochs:
                self.save_attention_maps(epoch)

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
    
    def save_attention_maps(self, epoch):
        """Сохранить attention maps для фиксированных изображений"""
        if self.fixed_images is None:
            return

        self.model.eval()
        with torch.no_grad():
            logits, attentions = self.model(self.fixed_images, output_attentions=True)
            # attentions - список тензоров [num_layers, batch, heads, seq_len, seq_len]

        # Создаём папку для сохранения
        save_dir = Path(self.exp_name) / f"attention_epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем сами карты внимания в тензорном формате
        torch.save(attentions, save_dir / "attentions.pt")

        # Также сохраняем изображения и предсказания для справки
        torch.save(self.fixed_images.cpu(), save_dir / "images.pt")
        torch.save(self.fixed_labels.cpu(), save_dir / "labels.pt")
        torch.save(logits.cpu(), save_dir / "logits.pt")

        print(f"Attention maps saved to {save_dir}")

        # По желанию можно сразу сгенерировать несколько картинок
        self.visualize_attention_maps(attentions, self.fixed_images.cpu(), epoch, save_dir)

    def visualize_attention_maps(self, attentions, images, epoch, save_dir, num_heads_to_show=2):
        """Создаёт и сохраняет PNG-визуализации для нескольких слоёв и голов"""
        num_layers = len(attentions)
        num_images = images.shape[0]
        patch_size = self.model.config["patch_size"]  # предположим, config доступен

        for img_idx in range(min(num_images, 4)):  # покажем не более 4 картинок
            img = images[img_idx].permute(1,2,0).numpy()
            # Нормализуем в [0,1] для отображения (зависит от preprocessing)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            for layer_idx in range(0, num_layers, max(1, num_layers//3)):  # каждый 3-й слой
                attn_layer = attentions[layer_idx][img_idx]  # [heads, seq_len, seq_len]
                num_heads = attn_layer.shape[0]

                for head_idx in range(0, num_heads, max(1, num_heads//num_heads_to_show)):
                    # Извлекаем внимание от CLS-токена к патчам (для классификации)
                    attn_cls = attn_layer[head_idx, 0, 1:].cpu().numpy()  # [num_patches]
                    # Преобразуем в пространственную карту
                    num_patches_side = int(np.sqrt(attn_cls.shape[0]))
                    attn_map = attn_cls.reshape(num_patches_side, num_patches_side)
                    # Увеличиваем до размера изображения
                    attn_map_resized = np.kron(attn_map, np.ones((patch_size, patch_size)))

                    # Рисуем
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
                    ax1.imshow(img)
                    ax1.set_title(f"Image {img_idx}")
                    ax1.axis('off')

                    ax2.imshow(img, alpha=0.3)
                    im = ax2.imshow(attn_map_resized, cmap='jet', alpha=0.7)
                    ax2.set_title(f"Layer {layer_idx}, Head {head_idx}")
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

                    save_path = save_dir / f"img{img_idx}_L{layer_idx}_H{head_idx}.png"
                    plt.savefig(save_path, bbox_inches='tight', dpi=100)
                    plt.close(fig)