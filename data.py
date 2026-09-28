"""Датасеты. Из обучающей части выделяется фиксированная валидационная выборка
(для выбора модели); официальная тестовая/val часть используется только для
итоговой оценки (в старом коде лучший чекпоинт выбирался по тесту)."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import autoaugment

IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CIFAR_STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

# размер изображения / патч по умолчанию: во всех случаях 64 патча + CLS = 65 токенов,
# как в экспериментах на CIFAR-10 (матрицы внимания 65x65 помещаются в поле 512)
DATASET_DEFAULTS = {
    'cifar10':      dict(img_size=32,  patch=4,  classes=10,  stats=CIFAR_STATS),
    'cifar100':     dict(img_size=32,  patch=4,  classes=100, stats=CIFAR_STATS),
    'tinyimagenet': dict(img_size=64,  patch=8,  classes=200, stats=IMAGENET_STATS),
    # val_fraction: доля train под валидацию; у Imagenette/Imagewoof train ~9 тыс., поэтому
    # 10 % (~900 изображений), иначе валидация из ~450 изображений слишком шумная
    # Imagenette/Imagewoof: 160x160 с патчем 20 (или 224 с патчем 28 из версии 320px) —
    # разрешение растёт, а число токенов (и стоимость оптики) остаётся 65
    'imagenette':   dict(img_size=160, patch=20, classes=10,  stats=IMAGENET_STATS, val_fraction=0.1),
    'imagewoof':    dict(img_size=160, patch=20, classes=10,  stats=IMAGENET_STATS, val_fraction=0.1),
    'fake':         dict(img_size=32,  patch=4,  classes=10,  stats=CIFAR_STATS),
}

TINY_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
WOOF_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-{}.tgz'


class TinyImageNet(Dataset):
    """Tiny ImageNet: 200 классов, 64x64; train — 500 изобр./класс,
    val — 50 изобр./класс (размечен, используется как тест)."""
    def __init__(self, root, split, transform=None, download=False):
        base = Path(root) / 'tiny-imagenet-200'
        if not base.exists():
            if not download:
                raise FileNotFoundError(f'{base} не найден; запусти с --download 1')
            download_and_extract_archive(TINY_URL, root)
        wnids = sorted(p.name for p in (base / 'train').iterdir() if p.is_dir())
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        self.samples = []
        if split == 'train':
            for w in wnids:
                for f in sorted((base / 'train' / w / 'images').glob('*.JPEG')):
                    self.samples.append((f, self.class_to_idx[w]))
        else:
            for line in (base / 'val' / 'val_annotations.txt').read_text().splitlines():
                f, w = line.split('\t')[:2]
                self.samples.append((base / 'val' / 'images' / f, self.class_to_idx[w]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f, y = self.samples[i]
        img = Image.open(f).convert('RGB')
        return (self.transform(img) if self.transform else img), y


def _transforms(name, img_size, stats, autoaug=True):
    small = img_size <= 64
    if small:   # CIFAR / Tiny ImageNet: изображения уже нужного размера
        tr = [transforms.RandomResizedCrop(img_size, scale=(0.6 if name == 'tinyimagenet' else 0.8, 1.0)),
              transforms.RandomHorizontalFlip()]
        te = [transforms.Resize(img_size)]
    else:
        tr = [transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
              transforms.RandomHorizontalFlip()]
        # версия 160px уже имеет короткую сторону 160: при img_size 160 не растягиваем её
        # до 182, а только обрезаем центр; для 224 из версии 320px — стандартные 256 -> 224
        rs = img_size if img_size == 160 else int(img_size * 1.14)
        te = [transforms.Resize(rs), transforms.CenterCrop(img_size)]
    if autoaug and name != 'fake':
        pol = (autoaugment.AutoAugmentPolicy.CIFAR10 if name.startswith('cifar')
               else autoaugment.AutoAugmentPolicy.IMAGENET)
        tr.append(autoaugment.AutoAugment(pol))
    tail = [transforms.ToTensor(), transforms.Normalize(*stats)]
    return transforms.Compose(tr + tail), transforms.Compose(te + tail)


def _pair(name, root, train_tf, test_tf, download, img_size=160):
    """(train с аугментацией, train без аугментации, тест)."""
    ver = '160' if img_size <= 160 else '320'     # исходная версия fast.ai
    if name in ('cifar10', 'cifar100'):
        C = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
        return (C(root, True, train_tf, download=download), C(root, True, test_tf),
                C(root, False, test_tf, download=download))
    if name == 'tinyimagenet':
        return (TinyImageNet(root, 'train', train_tf, download), TinyImageNet(root, 'train', test_tf),
                TinyImageNet(root, 'val', test_tf))
    if name == 'imagenette':
        mk = lambda s, tf, dl=False: datasets.Imagenette(root, s, f'{ver}px', download=dl, transform=tf)
        return mk('train', train_tf, download), mk('train', test_tf), mk('val', test_tf)
    if name == 'imagewoof':
        base = Path(root) / f'imagewoof2-{ver}'
        if not base.exists():
            if not download:
                raise FileNotFoundError(f'{base} не найден; запусти с --download 1')
            download_and_extract_archive(WOOF_URL.format(ver), root)
        mk = lambda s, tf: datasets.ImageFolder(base / s, tf)
        return mk('train', train_tf), mk('train', test_tf), mk('val', test_tf)
    if name == 'fake':  # для проверки кода без данных
        mk = lambda n, tf, seed: datasets.FakeData(n, (3, 32, 32), 10, tf, random_offset=seed)
        return mk(512, train_tf, 0), mk(512, test_tf, 0), mk(256, test_tf, 10_000)
    raise ValueError(f'неизвестный датасет {name}')


def build_datasets(name, root, val_fraction=None, split_seed=0, download=False,
                   autoaug=True, img_size=None):
    cfg = DATASET_DEFAULTS[name]
    if val_fraction is None or val_fraction < 0:
        val_fraction = cfg.get('val_fraction', 0.05)
    img_size = img_size or cfg['img_size']
    train_tf, test_tf = _transforms(name, img_size, cfg['stats'], autoaug)
    tr_aug, tr_plain, test = _pair(name, root, train_tf, test_tf, download, img_size)
    n = len(tr_aug)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed)).tolist()
    n_val = int(round(n * val_fraction))
    train = Subset(tr_aug, perm[n_val:])
    val = Subset(tr_plain, perm[:n_val])
    return train, val, test, cfg['classes'], img_size


def build_loaders(train, val, test, batch_size, eval_batch_size, workers, seed):
    g = torch.Generator().manual_seed(seed)
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=workers > 0)
    tl = DataLoader(train, batch_size, shuffle=True, drop_last=True, generator=g, **kw)
    # оценка: фиксированный порядок и фиксированный размер батча во всех запусках
    vl = DataLoader(val, eval_batch_size, shuffle=False, **kw)
    te = DataLoader(test, eval_batch_size, shuffle=False, **kw)
    return tl, vl, te
