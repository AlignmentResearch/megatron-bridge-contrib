"""fc1 (expert, column-parallel linear_in): the forward is exact, the BACKWARD is not.

A is sharded over the low-rank dim and gathered to a full z; B is sharded over the output and
consumes that full z, so the true dL/dz is a sum over every rank. explicit_expert_comm forces
allreduce_dgrad=False on the adapter's linear_out, so without a compensating backward
all-reduce rank r sees only its own contribution and dL/dA_r loses every cross-rank term.
`copy_to` (forward identity, backward all-reduce) restores it and leaves the forward alone.

The reference is produced by autograd on the merged-weight formulation, independently of how
the sharded arms are assembled -- otherwise the patched arm would be compared against its own
definition and could not fail.
"""
import torch

torch.manual_seed(0)
T, IN, DIM, OUT, ETP = 7, 12, 8, 16, 4
dim_sh, out_sh = DIM // ETP, OUT // ETP
x = torch.randn(T, IN, dtype=torch.float64)
A0 = torch.randn(DIM, IN, dtype=torch.float64)
B0 = torch.randn(OUT, DIM, dtype=torch.float64)
g = [torch.randn(T, out_sh, dtype=torch.float64) for _ in range(ETP)]

# ---- reference: autograd through W + B@A, sharded over the output dim ----
A = A0.clone().requires_grad_(True)
B = B0.clone().requires_grad_(True)
outs = [x @ (B[r * out_sh:(r + 1) * out_sh, :] @ A).t() for r in range(ETP)]
torch.autograd.backward(outs, g)
ref_dA, ref_dB = A.grad.clone(), B.grad.clone()

Bs = [B0[r * out_sh:(r + 1) * out_sh, :] for r in range(ETP)]
z = x @ A0.t()                                    # forward all-gather: full and exact

fwd_err = max((z @ Bs[r].t() - x @ (Bs[r] @ A0).t()).abs().max().item() for r in range(ETP))
print(f"forward (the fix must not move it)  max|err| = {fwd_err:.3e}")

# What rank r's linear_out backward produces locally, before any collective.
local_dz = [g[r] @ Bs[r] for r in range(ETP)]

def dA_from(per_rank_dz):
    """dL/dA_r is built from rank r's slice of whatever dL/dz that rank holds."""
    return torch.cat(
        [per_rank_dz[r][:, r * dim_sh:(r + 1) * dim_sh].t() @ x for r in range(ETP)], dim=0
    )

summed = sum(local_dz)                            # what an all-reduce leaves on every rank
arms = {
    "PATCHED (backward all-reduce)": dA_from([summed] * ETP),
    "UNPATCHED": dA_from(local_dz),
}
for name, dA in arms.items():
    rel = (dA - ref_dA).abs().max().item() / ref_dA.abs().max().item()
    print(f"{name:32s} dL/dA rel err = {rel:.3e} "
          f"-> {'matches merged weight' if rel < 1e-12 else 'DIFFERS from merged weight'}")

dB = torch.cat([g[r].t() @ z for r in range(ETP)], dim=0)
print(f"\ndL/dB rel err = {(dB - ref_dB).abs().max().item() / ref_dB.abs().max().item():.3e} "
      f"(already correct in both arms: B consumes the full z)")
