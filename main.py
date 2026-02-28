import torch
from config import config
from model import ViT
from data import prepare_data
from trainer import Trainer
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Конфигурация
config.update({
    "num_classes": 10,          # для CIFAR-10
    "use_optical": False,       # пока False для быстрого обучения
    "stochastic_depth_rate": 0.1,
})

# Данные
trainloader, testloader, num_classes, _ = prepare_data(
    batch_size=128,
    dataset_name='cifar10',
    use_autoaugment=True,
)

# Модель
model = ViT(config)

# Оптимизатор
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

# Планировщик
total_epochs = 300
warmup_epochs = 20
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                         milestones=[warmup_epochs])

# Тренер
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=torch.nn.CrossEntropyLoss(),
    exp_name='vit_cifar10',
    device='cuda',
    scheduler=scheduler,
    clip_grad_norm=1.0,
    use_cutmix=True,      # или use_mixup=True
    mixup_alpha=1.0,
    cutmix_alpha=1.0,
)

# Запуск
trainer.train(
    trainloader, 
    testloader, 
    epochs=total_epochs, 
    warmup_epochs=warmup_epochs)