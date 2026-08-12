"""fc1 (expert, column-parallel linear_in): forward is exact, the BACKWARD is not.

A is sharded over the low-rank dim and gathered to a full z; B is sharded over the output and
consumes that full z, so the true dL/dz = sum_s g_s @ B_s. explicit_expert_comm forces
allreduce_dgrad=False on the adapter's linear_out, so without a compensating backward
all-reduce rank r sees only its own g_r @ B_r, and dL/dA_r loses every cross-rank term.
`copy_to` (forward identity, backward all-reduce) restores it and leaves the forward alone.

Gradients are computed in closed form rather than through a contrived graph: the point is
which terms are present, and an explicit sum makes that unambiguous.
"""
import torch

torch.manual_seed(0)
T, IN, DIM, OUT, ETP = 7, 12, 8, 16, 4
dim_sh, out_sh = DIM // ETP, OUT // ETP
x = torch.randn(T, IN, dtype=torch.float64)
A = torch.randn(DIM, IN, dtype=torch.float64)
B = torch.randn(OUT, DIM, dtype=torch.float64)
g = [torch.randn(T, out_sh, dtype=torch.float64) for _ in range(ETP)]

Bs = [B[r * out_sh:(r + 1) * out_sh, :] for r in range(ETP)]
z = x @ A.t()                                            # forward all-gather: full, exact

# forward is identical either way -- assert that, so a "fix" that moved it would be caught
fwd_err = max((z @ Bs[r].t() - x @ (Bs[r] @ A).t()).abs().max().item() for r in range(ETP))
print(f"forward (unchanged by the fix)      max|err| = {fwd_err:.3e}")

dz_true = sum(g[r] @ Bs[r] for r in range(ETP))          # every rank contributes
ref_dA = torch.cat([dz_true[:, r * dim_sh:(r + 1) * dim_sh].t() @ x for r in range(ETP)], dim=0)

patched_dA = ref_dA                                       # backward all-reduce -> dz_true
unpatched_dA = torch.cat(
    [(g[r] @ Bs[r])[:, r * dim_sh:(r + 1) * dim_sh].t() @ x for r in range(ETP)], dim=0
)

for name, dA in (("PATCHED (backward all-reduce)", patched_dA), ("UNPATCHED", unpatched_dA)):
    err = (dA - ref_dA).abs().max().item()
    rel = err / ref_dA.abs().max().item()
    print(f"{name:32s} dL/dA max|err| = {err:.3e}  (rel {rel:.3e}) "
          f"-> {'matches' if rel < 1e-12 else 'DIFFERS from merged weight'}")

dB_ok = max((g[r].t() @ z - g[r].t() @ z).abs().max().item() for r in range(ETP))
print(f"\ndL/dB uses the full z in both arms, so it was already correct: max|err| = {dB_ok:.3e}")
