import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_VISIBLE_DEVICES'] = f"0,1,2,3,4,5,6,7"

import torch
from config import config
from model import ViT
from data import prepare_data
from trainer import Trainer
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Конфигурация
config.update(
    {
        "num_classes": 10,             # для CIFAR-10
        "use_optical": False,          # использовать ли оптическое умножение
        "optical_layers": 0,           # количество слоев с оптикой (0 = все, если use_optical=True)
        "stochastic_depth_rate": 0.1,
    }
)

# Данные
trainloader, testloader, num_classes, _ = prepare_data(
    batch_size=256,
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
    exp_name='vit_cifar10_256',
    device='cuda',
    scheduler=scheduler,
    clip_grad_norm=1.0,
    use_cutmix=True,      # или use_mixup=True
    mixup_alpha=1.0,
    cutmix_alpha=1.0,
    save_attention_every_n_epochs=50,    # сохранять каждые 50 эпох
    num_attention_samples=8              # использовать 8 изображений
)

# Запуск
trainer.train(
    trainloader, 
    testloader, 
    epochs=total_epochs, 
    warmup_epochs=warmup_epochs)