import copy as _copy

import torch
import torch.nn as nn

_MODULE_INTERNALS = {
    '_parameters', '_buffers', '_modules', '_non_persistent_buffers_set',
    '_backward_hooks', '_backward_pre_hooks', '_forward_hooks',
    '_forward_hooks_with_kwargs', '_forward_pre_hooks',
    '_forward_pre_hooks_with_kwargs', '_state_dict_hooks',
    '_state_dict_pre_hooks', '_load_state_dict_pre_hooks',
    '_load_state_dict_post_hooks', 'training',
}


def _relocate(module, device):
    module.to(device)
    for m in module.modules():
        for name, val in list(m.__dict__.items()):
            if name in _MODULE_INTERNALS:
                continue
            if isinstance(val, torch.Tensor):
                m.__dict__[name] = val.to(device)
            elif isinstance(val, (list, tuple)) and any(
                    isinstance(v, torch.Tensor) for v in val):
                m.__dict__[name] = type(val)(
                    v.to(device) if isinstance(v, torch.Tensor) else v
                    for v in val)
            elif isinstance(val, dict) and any(
                    isinstance(v, torch.Tensor) for v in val.values()):
                m.__dict__[name] = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in val.items()}
    return module


class ParallelSim(nn.Module):
    def __init__(self, sim, devices, output_device=None):
        super().__init__()
        self.devices = list(devices)
        self.output_device = output_device or self.devices[0]
        self.replicas = nn.ModuleList(
            [_relocate(_copy.deepcopy(sim), d) for d in self.devices])

    def _apply(self, *args, **kwargs):
        return self

    def _single(self, P, Q):
        d = self.devices[0]
        return self.replicas[0](P.to(d), Q.to(d)).to(self.output_device)

    def forward(self, P, Q):
        n = len(self.devices)
        if n < 2 or P.shape[-2] < n:
            return self._single(P, Q)
        dim = P.dim() - 2
        P_chunks = nn.parallel.scatter(P, self.devices, dim)
        Q_bcast = nn.parallel.comm.broadcast(Q, self.devices)
        k = len(P_chunks)
        outs = [self.replicas[i](P_chunks[i], Q_bcast[i]) for i in range(k)]
        return nn.parallel.gather(outs, self.output_device, dim)


@torch.no_grad()
def check_parallel(psim, aperture, hd, seq_len, device):
    ref = psim.replicas[0]
    d0 = psim.devices[0]
    shapes = [
        ('linear', (2, 1, seq_len, aperture), (1, 1, aperture, aperture)),
        ('qk',     (2, 8, seq_len, hd),       (2, 8, hd, seq_len)),
        ('av',     (2, 8, seq_len, seq_len),  (2, 8, seq_len, hd)),
    ]
    worst = 0.0
    for name, sp, sq in shapes:
        P = torch.rand(*sp, device=device)
        Q = torch.rand(*sq, device=device)
        out_par = psim(P, Q)
        out_ref = ref(P.to(d0), Q.to(d0)).to(psim.output_device)
        rel = ((out_par - out_ref).norm() / out_ref.norm().clamp_min(1e-9)).item()
        worst = max(worst, rel)
        print(f'  [check_parallel] {name:6s}: rel diff {rel:.2e}')
    if worst > 1e-4:
        print(f'  ВНИМАНИЕ: max rel diff {worst:.2e} > 1e-4 — sim, похоже, '
              f'связывает строки по оси M. Шардинг по M НЕкорректен.')
    else:
        print(f'  [check_parallel] OK: max rel diff {worst:.2e} (< 1e-4).')
    return worst
