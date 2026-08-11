"""Check the fix's algebra: zero-embed + dispatcher sum == B@(A@x), forward AND gradients.

T is deliberately not divisible by ETP so the pad/unpad path is exercised; the pad's adjoint
is what the `no_grad` removal restores.
"""
import torch

torch.manual_seed(0)
T, IN, DIM, OUT, ETP = 7, 12, 4, 8, 4
in_sh, out_sh = IN // ETP, OUT // ETP
x = torch.randn(T, IN, dtype=torch.float64)
g = torch.randn(T, OUT, dtype=torch.float64)


def fresh():
    A = torch.randn(DIM, IN, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    B = torch.randn(OUT, DIM, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    return A.requires_grad_(True), B.requires_grad_(True)


def run(fn):
    A, B = fresh()
    out = fn(A, B)
    out.backward(g)
    return out.detach(), A.grad.clone(), B.grad.clone()


def merged(A, B):
    return x @ (B @ A).t()


def patched(A, B):
    z = sum(x[:, r * in_sh:(r + 1) * in_sh] @ A[:, r * in_sh:(r + 1) * in_sh].t() for r in range(ETP))
    parts = []
    for r in range(ETP):
        Br = B[r * out_sh:(r + 1) * out_sh, :]
        left, right = r * out_sh, (ETP - 1) * out_sh - r * out_sh
        parts.append(torch.nn.functional.pad(Br @ z.t(), (0, 0, left, right)).t())
    return sum(parts)


def unpatched(A, B):
    zs = [x[:, r * in_sh:(r + 1) * in_sh] @ A[:, r * in_sh:(r + 1) * in_sh].t() for r in range(ETP)]
    full = torch.cat([B[r * out_sh:(r + 1) * out_sh, :] @ zs[r].t() for r in range(ETP)], dim=0).t()
    return ETP * full


ref, ref_dA, ref_dB = run(merged)
for name, fn in (("PATCHED", patched), ("UNPATCHED (negative control)", unpatched)):
    out, dA, dB = run(fn)
    f, a, b = (out - ref).abs().max(), (dA - ref_dA).abs().max(), (dB - ref_dB).abs().max()
    print(f"{name:30s} forward {f:.3e}  dL/dA {a:.3e}  dL/dB {b:.3e}")
    verdict = "matches merged weight" if max(f, a, b) < 1e-10 else "DIFFERS from merged weight"
    print(f"{'':30s} -> {verdict}")
