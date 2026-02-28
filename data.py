import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment

def prepare_data(
    batch_size=128,
    num_workers=4,
    dataset_name='cifar10',        # 'cifar10' или 'cifar100'
    data_dir='/wd/vit_optica/data',
    train_sample_size=None,
    test_sample_size=None,
    use_autoaugment=True,
):
    """
    Загружает и подготавливает датасет CIFAR-10 или CIFAR-100.
    
    Args:
        batch_size: размер батча для DataLoader
        num_workers: число воркеров для загрузки
        dataset_name: 'cifar10' или 'cifar100'
        data_dir: путь для сохранения данных
        train_sample_size: если указано, использовать только подвыборку train (для экспериментов)
        test_sample_size: аналогично для test
        use_autoaugment: использовать ли AutoAugment (политика CIFAR10)
    
    Returns:
        trainloader, testloader, num_classes, class_names (если есть)
    """
    
    # Выбор датасета
    if dataset_name.lower() == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
        num_classes = 10
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')
        autoaugment_policy = autoaugment.AutoAugmentPolicy.CIFAR10
    elif dataset_name.lower() == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
        num_classes = 100
        class_names = None  # у CIFAR100 нет имен классов по умолчанию
        autoaugment_policy = autoaugment.AutoAugmentPolicy.CIFAR10  # можно использовать ту же политику
    else:
        raise ValueError("dataset_name must be either 'cifar10' or 'cifar100'")
    
    # --- Трансформации для тренировки ---
    train_transforms_list = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(
            size=32,
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
    ]
    
    if use_autoaugment:
        train_transforms_list.append(
            autoaugment.AutoAugment(policy=autoaugment_policy)
        )
    
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_transform = transforms.Compose(train_transforms_list)
    
    # --- Трансформации для теста (без аугментаций) ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Загрузка тренировочного набора
    trainset = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # ускоряет передачу на GPU
        drop_last=True,   # избегаем проблем с BatchNorm на последнем неполном батче
    )
    
    # Загрузка тестового набора
    testset = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return trainloader, testloader, num_classes, class_names