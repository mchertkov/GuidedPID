# guided_torch/guided_torch.py
# Rebuilt GuidedPID API mirroring AdaPID architecture, using (A.7)
# Requires: your Ada module exposes BetaSchedulePWCTorch with alpha_K_gamma(t), a_minus(t), b_minus(t), c_minus(t)

from __future__ import annotations
import math, numpy as np, torch
from dataclasses import dataclass
from typing import List, Tuple
Tensor = torch.Tensor

# --------------------- Helpers: stable scalar maps ---------------------

def _tanh_stable(z: float) -> float:
    az = abs(z)
    if az < 1e-8:  # series
        return z - (z**3)/3.0
    # standard
    ez = math.exp(z)
    emz = math.exp(-z)
    return (ez - emz) / (ez + emz)

def _coth_stable(z: float) -> float:
    az = abs(z)
    if az < 1e-6:
        # series: coth z = 1/z + z/3 - z^3/45 + ...
        return (1.0 / z) + (z / 3.0)
    # coth z = 1 + 2/(e^{2z}-1)
    e2z_minus_1 = math.expm1(2.0 * z)  # stable for small z
    return 1.0 + 2.0 / e2z_minus_1

# --------------------- Types from Ada API ---------------------

Device = torch.device

# We assume your AdaPID torch API is in adapid_torch/adapid_torch.py
# and exposes BetaSchedulePWCTorch, GMMTorch.
try:
    from adapid_torch.adapid_torch import BetaSchedulePWCTorch, GMMTorch
except ImportError:
    # Fallback: direct import if user kept the flat adapid_torch.py
    from adapid_torch import BetaSchedulePWCTorch, GMMTorch  # type: ignore

# --------------------- Utility to ensure X is 2D ---------------------

def _as_batch(X: Tensor) -> Tensor:
    if X.ndim == 1:
        return X.unsqueeze(0)
    return X

# --------------------- ν-PWC schedule ---------------------

@dataclass
class NuPWC:
    """
    Piecewise-constant ν(t) on splits[0..S], values[k] on [splits[k], splits[k+1]).
    """
    values: Tensor   # shape (S,d)
    splits: Tensor   # shape (S+1,)

    @staticmethod
    def from_numpy(nu_vals: np.ndarray, splits: np.ndarray, device: Device) -> "NuPWC":
        assert nu_vals.ndim == 2
        S, d = nu_vals.shape
        assert splits.shape[0] == S+1
        return NuPWC(
            values=torch.tensor(nu_vals, dtype=torch.float32, device=device),
            splits=torch.tensor(splits,  dtype=torch.float32, device=device),
        )

    def value(self, t: float) -> Tensor:
        """
        ν(t) piecewise-constant:
          find k s.t. t in [splits[k], splits[k+1]).
        """
        t = float(t)
        s_np = self.splits.cpu().numpy()
        k = int(np.searchsorted(s_np, t, side="right") - 1)
        k = max(0, min(self.values.shape[0] - 1, k))
        return self.values[k]

    def nu(self, t: float) -> Tensor:
        return self.value(t)

# --------------------- APlusPWC: a^{(+)} PWC param ---------------------

@dataclass
class APlusPWC:
    """
    Stores piecewise index for a^{(+)}(t) over the same splits as BetaSchedulePWCTorch.
    For each piece [tL, tR], we store:
       (tL, tR, r, aL, aR),
    where r=√(2β_k), aL=a^{(+)}(tL^+), aR=a^{(+)}(tR^-).
    """
    pieces: List[Tuple[float,float,float,float,float]]
    splits: Tensor  # shape (S+1,)

    @staticmethod
    def build(betas: np.ndarray, splits: np.ndarray) -> "APlusPWC":
        """
        Given piecewise constant β_k on [splits[k],splits[k+1]),
        we construct a^{(+)} by matching right-edge boundaries piece by piece,
        with a^{(+)}(0^+)=+∞ as in the notes (Girsanov boundary).
        """
        S = len(betas)
        assert splits.shape[0] == S+1
        pieces = []

        # a(0^+)=+∞
        aL = float("inf")

        for k in range(S):
            tL, tR = float(splits[k]), float(splits[k+1])
            dur = tR - tL
            beta_k = float(betas[k])
            r = math.sqrt(2.0 * beta_k) if beta_k > 0.0 else 0.0

            if aL == float("inf"):
                if r == 0.0:
                    aR = 1.0 / dur if dur > 0.0 else float("inf")
                else:
                    z = r * dur
                    if abs(z) < 1e-6:
                        aR = 1.0 / dur if dur > 0.0 else float("inf")
                    else:
                        aR = r * _coth_stable(z)
            else:
                if r == 0.0:
                    aR = aL
                else:
                    z = r * dur
                    aR = r * (aL + r * math.tanh(z)) / (r + aL * math.tanh(z))

            pieces.append((tL, tR, r, aL, aR))
            aL = aR

        return APlusPWC(pieces=pieces, splits=torch.tensor(splits, dtype=torch.float32))

    def a_plus(self, t: float) -> float:
        t = float(t)
        if t <= self.pieces[0][0]:
            return float("inf")
        if t >= self.pieces[-1][1]:
            return self.pieces[-1][4]
        s_np = self.splits.cpu().numpy()
        j = int(np.searchsorted(s_np, t, side="right") - 1)
        j = max(0, min(len(self.pieces)-1, j))
        tL, tR, r, aL, aR = self.pieces[j]
        tau = t - tL
        if aL == float("inf"):
            return r * _coth_stable(r * tau) if r != 0.0 else (1.0/tau if tau>0.0 else float("inf"))
        z = r * tau
        if r == 0.0:
            return aL
        return r * (aL + r * math.tanh(z)) / (r + aL * math.tanh(z))

# --------------------- s^{(+)}(1^-) from (A.16–A.17) ---------------------

def _s_plus_terminal_from_betas_splits(
    betas: Tensor,
    splits: Tensor,
    nu_vals: Tensor,
    d: int,
    device: Device,
) -> Tensor:
    """
    Computes s^{(+)}(1^-) as in (A.16)-(A.17) directly from β_k, splits, ν_k.
    """
    S = betas.shape[0]
    s_plus = torch.zeros(d, dtype=torch.float32, device=device)

    for k in range(S):
        beta_k = float(betas[k])
        tL = float(splits[k])
        tR = float(splits[k+1])
        dur = tR - tL
        nu_k = nu_vals[k]

        if beta_k > 0.0:
            r = math.sqrt(2.0 * beta_k)
            if dur <= 0.0:
                decay = 1.0
            else:
                z = r * dur
                decay = math.exp(-r * dur) if abs(z) < 1e-6 else math.exp(-r * dur)
        else:
            decay = 1.0

        s_plus = decay * s_plus

        if k < S - 1:
            beta_R = float(betas[k+1])
            nu_next = nu_vals[k+1]
            dnu = (nu_next - nu_k)
            if beta_R > 0.0:
                rR = math.sqrt(2.0 * beta_R)
                aRk = rR
            else:
                aRk = 1.0/dur if dur>0.0 else 1.0
            s_plus = s_plus - float(aRk) * dnu

    return s_plus

def s_plus_terminal(beta_sched: BetaSchedulePWCTorch, nu_sched: NuPWC, d: int, device: Device) -> Tensor:
    betas = torch.as_tensor(beta_sched.betas if hasattr(beta_sched, "betas") else None)
    if betas is None or betas.numel() != (len(beta_sched.splits) - 1):
        print("[guided_torch] ERROR: BetaSchedulePWCTorch must expose original betas and splits.")
        raise RuntimeError("Missing betas or mismatched splits in BetaSchedulePWCTorch")
    return _s_plus_terminal_from_betas_splits(betas, torch.as_tensor(beta_sched.splits),
                                              nu_sched.values, d, device)

# --------------------- r^-, s^- PWC from (A.14–A.15) ---------------------

# ---------------------------------------
# r^-(t), s^-(t) per (A.14) + reverse jumps (A.15 + s^- jump)
# ---------------------------------------
@dataclass
class RSMinusPWC:
    """
    Piecewise description of r^-(t), s^-(t).

    For each piece j we store:
      (C_j[d], sR_j[d], cR_j(float), tL_j(float), tR_j(float))

    so that for t in [tL_j, tR_j]:

        r^-(t) = C_j * b^-(t)
        s^-(t) = sR_j + C_j * (c^-(t) - cR_j)
    """
    params: List[Tuple[Tensor, Tensor, float, float, float]]
    beta_sched: BetaSchedulePWCTorch
    splits: Tensor   # shape (S+1,)

    def _seg_idx(self, t: float) -> int:
        sp = self.splits.detach().cpu().numpy()
        import numpy as np
        i = int(np.searchsorted(sp, float(t), side="right") - 1)
        return max(0, min(i, len(self.params) - 1))

    def r_minus(self, t: float) -> Tensor:
        j = self._seg_idx(t)
        Cj, _, _, _, _ = self.params[j]
        b_t = float(self.beta_sched.b_minus(float(t)))
        return Cj * b_t

    def s_minus(self, t: float) -> Tensor:
        j = self._seg_idx(t)
        Cj, sRj, cRj, _, _ = self.params[j]
        c_t = float(self.beta_sched.c_minus(float(t)))
        return sRj + Cj * (c_t - cRj)


def build_rsminus_pwc(
    beta_sched: BetaSchedulePWCTorch,
    nu_sched: NuPWC,
    d: int,
    device: Device,
) -> RSMinusPWC:
    """
    Build RSMinusPWC storing, for each PWC piece j, an invariant vector C_j and
    right-anchored s_R, so that for any t in that piece we can reconstruct
    r^-(t), s^-(t) via (A.14):

        r_t^- = C_j * b^-(t)
        s_t^- = s_R + C_j * (c^-(t) - c_R)

    and include the splits tensor in the dataclass.
    """
    eps_val = 1e-32
    eps_t   = 1e-12

    # Make sure we have splits on the correct device
    splits_t = torch.as_tensor(beta_sched.splits, dtype=torch.float32, device=device)
    S = splits_t.numel() - 1

    params: List[Tuple[Tensor, Tensor, float, float, float]] = []

    # Initialize at right edge: r_R = 0, s_R = 0 in ℝ^d
    r_R = torch.zeros(d, dtype=torch.float32, device=device)
    s_R = torch.zeros(d, dtype=torch.float32, device=device)

    for j in reversed(range(S)):
        tL_raw = float(splits_t[j].item())
        tR_raw = float(splits_t[j+1].item())

        # Avoid exact endpoints for numerical safety
        tR_in = max(tL_raw, tR_raw - eps_t)
        tL_in = min(tR_raw, tL_raw + eps_t)

        # Right-end values
        bR = float(beta_sched.b_minus(tR_in))
        cR = float(beta_sched.c_minus(tR_in))
        bR_safe = bR if abs(bR) > eps_val else (math.copysign(eps_val, bR) if bR != 0.0 else eps_val)

        # Invariant for piece j
        Cj = r_R / bR_safe   # tensor (d,)

        # Store (C_j, s_R, c_R, tL_raw, tR_raw)
        params.insert(0, (Cj.clone(), s_R.clone(), cR, tL_raw, tR_raw))

        # Transport to left interior via (A.14)
        bL = float(beta_sched.b_minus(tL_in))
        cL = float(beta_sched.c_minus(tL_in))
        r_L_plus = Cj * bL
        s_L_plus = s_R + Cj * (cL - cR)

        # Reverse jump (A.15, including s^- jump) at t_L
        if j > 0:
            aL = float(beta_sched.a_minus(tL_in))
            dnu = (nu_sched.values[j-1] - nu_sched.values[j]).to(device)
            r_R = r_L_plus - (aL - bL) * dnu
            s_R = s_L_plus - (cL - bL) * dnu
        else:
            r_R, s_R = r_L_plus, s_L_plus

    return RSMinusPWC(params=params, beta_sched=beta_sched, splits=splits_t)


# ---------------------------------------
# Guided schedule wrapper
# ---------------------------------------
@dataclass
class GuidedPWCSchedule:
    beta_sched: BetaSchedulePWCTorch
    nu_sched: NuPWC
    rsminus: RSMinusPWC
    splus1: Tensor  # s^{(+)}(1^-)

    @staticmethod
    def build(betas, splits, nu_values, d: int, device: Device = torch.device("cpu")) -> "GuidedPWCSchedule":
        beta_sched = BetaSchedulePWCTorch(betas, splits)
        nu_sched   = NuPWC.from_numpy(nu_values, splits, device=device)
        rsminus    = build_rsminus_pwc(beta_sched, nu_sched, d=d, device=device)
        splus1     = s_plus_terminal(beta_sched, nu_sched, d=d, device=device)
        return GuidedPWCSchedule(beta_sched, nu_sched, rsminus, splus1)

    def a_minus(self, t: float) -> float: return float(self.beta_sched.a_minus(t))
    def b_minus(self, t: float) -> float: return float(self.beta_sched.b_minus(t))
    def c_minus(self, t: float) -> float: return float(self.beta_sched.c_minus(t))

    def K(self, t: float) -> float: 
        """
        Robust K(t) query. Changed from K (above) 11/15/25

        For very aggressive β-schedules, the exact Riccati-based K(t)
        can cross zero (or become numerically unstable) for extremely
        small t ~ 0^+. Analytically K(t) > 0, but numerically we hit a
        formal singularity.

        To avoid this, we:
          * Clamp t to a minimum t_eps near 0^+.
          * If K(t_eps) still fails, we adaptively nudge t forward
            in small steps until K(t) is valid or we reach t≈1.

        This keeps the GH-PID API unchanged while allowing exploration
        of very large β without spurious RuntimeError at t→0.
        """
        t0 = float(t)
        # minimum "0^+" time — you can adjust if needed
        t_eps = 1e-3

        # start from max(t, t_eps)
        t_eff = t0 if t0 > t_eps else t_eps

        # small forward step if we need to nudge further
        dt_probe = 1e-3

        while True:
            try:
                _, Kt, _ = self.beta_sched.alpha_K_gamma(float(t_eff))
                # if we get here, Kt is positive and finite
                return float(Kt)
            except RuntimeError as e:
                t_eff += dt_probe
                if t_eff >= 1.0:
                    raise RuntimeError(
                        f"GuidedPWCSchedule.K: could not find positive K(t) "
                        f"for t ≥ {t0:.6g}. Last error: {e}"
                    )


    def K_pos(self, t: float, eps: float = 1e-12) -> float:
        Kt = self.K(t)
        return Kt if (Kt > eps) else eps

    def psi(self, t: float) -> Tensor:
        return self.rsminus.s_minus(t) - self.splus1

    def mu_guide(self, X: Tensor, t: float) -> Tensor:
        """
        μ_guided(t;x) = ν_t + (b^-(t)/K(t)) (x - ν_t) + ψ(t)/K(t)
        NOTE: uses raw K(t) from beta_sched, exactly as AdaPID does.
        """
        Xb = X if X.ndim == 2 else X.unsqueeze(0)
        nu  = self.nu_sched.value(float(t))
        Kt  = self.K(float(t))          # <-- IMPORTANT: use raw K(t), not K_pos
        bm  = self.b_minus(float(t))
        psi = self.psi(float(t))
        eps = 1e-32
        return nu + (bm / (Kt + eps)) * (Xb - nu) + psi / (Kt + eps)

# ---------------------------------------
# ŷ with μ_guided (same fusion as Ada)
# ---------------------------------------
"""
# replaced 11/14/25
def yhat_gmm_guided(X: Tensor, t: float, sched: GuidedPWCSchedule, gmm: GMMTorch) -> Tensor:
    X = _as_batch(X).to(gmm.MU.device)
    m  = sched.mu_guide(X, t)
    Kt = sched.K(float(t))  # <-- IMPORTANT: use raw K(t), as in Ada

    MU, SIG, PI = gmm.MU, gmm.SIG, gmm.PI
    inv_sig2 = 1.0 / (SIG**2 + 1e-32)
    lam = (Kt * inv_sig2) / (inv_sig2 + Kt)

    dmu = MU.unsqueeze(0) - m.unsqueeze(1)
    r2  = (dmu**2).sum(-1)
    w   = torch.exp(-0.5 * r2 * lam.unsqueeze(0)) * PI.unsqueeze(0)

    num = inv_sig2.unsqueeze(0).unsqueeze(-1) * MU.unsqueeze(0) + Kt * m.unsqueeze(1)
    den = (inv_sig2 + Kt).unsqueeze(0).unsqueeze(-1)
    mu_tk = num / (den + 1e-32)

    w = w / (w.sum(dim=1, keepdim=True) + 1e-32)
    yhat = (w.unsqueeze(-1) * mu_tk).sum(dim=1)
    return yhat
"""

def yhat_gmm_guided(X: Tensor, t: float, sched: GuidedPWCSchedule, gmm: GMMTorch) -> Tensor:
    """
    Guided oracle ŷ(t;x;Γ) for a GMM target with possibly full covariance.

    Shapes:
      X       : (M,d)    or (d,) -> treated as (M,d)
      MU      : (N,d)
      SIG     : (N,) or (N,d) or (N,d,d)
                - if (N,d,d), interpreted as full covariance; we use its diagonal.
      PI      : (N,)
      output  : (M,d)
    """
    # Ensure batch shape (M,d)
    Xb = _as_batch(X).to(gmm.MU.device)      # (M,d)
    m  = sched.mu_guide(Xb, t)               # (M,d)
    Kt = sched.K(float(t))                   # scalar

    MU, SIG, PI = gmm.MU, gmm.SIG, gmm.PI.to(gmm.MU.device)  # MU: (N,d)

    # ---- Build diagonal variance matrix var_diag of shape (N,d) ----
    if SIG.ndim == 3:
        # Full covariance: take diagonal Σ_{n,kk}
        # SIG: (N,d,d) -> var_diag: (N,d)
        var_diag = torch.diagonal(SIG, dim1=1, dim2=2)
    elif SIG.ndim == 2:
        # Interpret as per-dim std devs
        var_diag = SIG ** 2                        # (N,d)
    elif SIG.ndim == 1:
        # One std per component; expand across dimensions
        var_diag = (SIG ** 2).unsqueeze(-1)        # (N,1)
    else:
        raise RuntimeError(f"yhat_gmm_guided: unexpected SIG shape {SIG.shape}")

    # Broadcast to match MU if needed
    if var_diag.shape != MU.shape:
        var_diag = var_diag.expand_as(MU)          # (N,d)

    # Per-component, per-dimension precision
    inv_var = 1.0 / (var_diag + 1e-32)             # (N,d)

    # ---------- weights w_{i,n} ----------
    # dmu_{i,n,k} = MU_{n,k} - m_{i,k}
    dmu = MU.unsqueeze(0) - m.unsqueeze(1)         # (M,N,d)
    # quadratic form with diagonal precision: sum_k inv_var_{n,k} (dmu_{i,n,k})^2
    quad = ((dmu**2) * inv_var.unsqueeze(0)).sum(-1)   # (M,N)
    # unnormalized weights
    w = torch.exp(-0.5 * quad) * PI.unsqueeze(0)       # (M,N)

    # ---------- posterior means μ_{i,n,k}(t) ----------
    inv = inv_var.unsqueeze(0)   # (1,N,d)
    MUe = MU.unsqueeze(0)        # (1,N,d)
    me  = m.unsqueeze(1)         # (M,1,d)

    # convex combination (component mean vs guided mean), dimension-wise
    num = inv * MUe + Kt * me    # (M,N,d)
    den = inv + Kt               # (1,N,d)
    mu_tk = num / (den + 1e-32)  # (M,N,d)

    # ---------- mixture over components ----------
    w = w / (w.sum(dim=1, keepdim=True) + 1e-32)   # (M,N)
    yhat = (w.unsqueeze(-1) * mu_tk).sum(dim=1)    # (M,d)

    # Final sanity: enforce shape (M,d)
    if yhat.ndim != 2 or yhat.shape[0] != Xb.shape[0]:
        raise RuntimeError(
            f"yhat_gmm_guided: unexpected shape {yhat.shape}, "
            f"expected (M,d)=({Xb.shape[0]}, {Xb.shape[1]})"
        )
    return yhat


# ---------------------------------------
# u*_guided and simulation
# ---------------------------------------
#@torch.no_grad()
def ustar_guided(X: Tensor, t: float, sched: GuidedPWCSchedule, gmm: GMMTorch) -> Tensor:
    r"""
    Guided optimal control u^*(t,x;Γ) (Eq. (*)):

        u_t^*(x;Γ) = -a^-(t) (x - ν_t)
                      + b^-(t) (ŷ(t;x;Γ) - ν_t)
                      + r^-(t).

    Shapes:
      X    : (M,d) or (d,) -> treated as (M,d)
      ν_t  : (d,)
      r^-  : (d,)
      ŷ    : (M,d)
      u    : (M,d)
    """
    # ensure shape (M, d)
    Xb = _as_batch(X).to(gmm.MU.device)   # (M,d)
    M, d = Xb.shape

    # scalar coefficients from the shared beta_sched
    a_m = float(sched.a_minus(float(t)))    # a^-(t)
    b_m = float(sched.b_minus(float(t)))    # b^-(t)

    # guidance ν_t and linear term r^-(t), both ∈ ℝ^d
    nu_t = sched.nu_sched.value(float(t)).to(Xb.device).view(1, d)   # (1,d)
    r_t  = sched.rsminus.r_minus(float(t)).to(Xb.device).view(1, d)  # (1,d)

    # guided oracle ŷ(t;x;Γ) ∈ ℝ^{M×d}
    yhat = yhat_gmm_guided(Xb, float(t), sched, gmm)                 # (M,d)

    if yhat.shape != Xb.shape:
        raise RuntimeError(
            f"ustar_guided: yhat shape {yhat.shape} "
            f"does not match X shape {Xb.shape}"
        )

    # Eq. (*): u = -a^- (x - ν_t) + b^- (ŷ - ν_t) + r^-
    u = -a_m * (Xb - nu_t) + b_m * (yhat - nu_t) + r_t               # (M,d)
    return u


#@torch.no_grad()
def simulate_guided_paths(
    sched: GuidedPWCSchedule,
    gmm: GMMTorch,
    M: int = 4000,
    T: int = 1200,
    seed: int = 0,
    device: Device = torch.device("cpu"),
    verbose: bool = False,
    check_every: int = 200,
) -> Tensor:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gmm = gmm.to(device)

    X = torch.zeros((M, gmm.d), device=device)
    dt = 1.0 / T
    sdt = math.sqrt(dt)

    for n in range(T):
        t_mid = (n + 0.5) / T
        u = ustar_guided(X, t_mid, sched, gmm)
        if not torch.isfinite(u).all():
            raise RuntimeError(f"Non-finite control at step {n}, t={t_mid:.6f}")
        X = X + u * dt + torch.randn_like(X) * sdt
        if not torch.isfinite(X).all():
            raise RuntimeError(f"Non-finite state at step {n}, t={t_mid:.6f}")

    return X
