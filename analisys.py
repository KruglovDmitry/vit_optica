import torch
from model import ViT
from data import prepare_data
from profiler import PROFILER

device = "cuda:0"
path = "/wd/vit_optica/checkpoints/vit_cifar10_simple_512_optica_new_final.pth"

ckpt = torch.load(path, map_location=device)

cfg = dict(ckpt["config"])
cfg["use_optical"] = False        # профилируем Q/K/V, оптика не нужна
cfg["optical_layers"] = 0
model = ViT(cfg).to(device)

missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"missing: {len(missing)} | unexpected: {len(unexpected)}")

# unexpected — буферы симулятора (sim/_slm_doe/_propagator/_kron), их нет в модели без оптики -> ОК.
# Падаем только если потерян настоящий вес сети.
SIM = ("sim", "simulator", "_slm_doe", "_propagator", "_kron")
real_missing = [k for k in missing if not any(t in k for t in SIM)]
assert not real_missing, real_missing[:10]

_, testloader, _, _ = prepare_data(batch_size=128, dataset_name="cifar10",
                                   use_autoaugment=False)
model.eval()
PROFILER.reset(); PROFILER.enabled = True
with torch.no_grad():
    for i, (imgs, _) in enumerate(testloader):
        model(imgs.to(device))
        if i >= 10:
            break
PROFILER.enabled = False
PROFILER.report()