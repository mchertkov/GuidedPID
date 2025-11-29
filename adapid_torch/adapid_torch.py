
# AdaPID PyTorch API (single file) — constant-β and PWC-β schedules
# Unit-diffusion SDE convention: dX_t = u*_t(X_t) dt + dW_t
from dataclasses import dataclass
import math
import torch

Tensor = torch.Tensor
Device = torch.device

def _expand_2d(x: Tensor) -> Tensor:
    return x.unsqueeze(0) if x.ndim == 1 else x

def set_seed(seed: int = 0):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------------------
# GMM (isotropic) — SIG is **standard deviation** per component
# -------------------------
@dataclass
class GMMTorch:
    MU: Tensor   # (K,d)
    SIG: Tensor  # (K,)  std per component
    PI: Tensor   # (K,)  weights summing to 1

    def to(self, device: Device) -> "GMMTorch":
        return GMMTorch(self.MU.to(device), self.SIG.to(device), self.PI.to(device))

    @property
    def K(self): return int(self.MU.shape[0])
    @property
    def d(self): return int(self.MU.shape[1])

def sample_gmm_torch(gmm: GMMTorch, n: int, generator=None) -> Tensor:
    if generator is None:
        generator = torch.Generator(device=gmm.MU.device)
    idx = torch.multinomial(gmm.PI, num_samples=n, replacement=True, generator=generator)
    mu  = gmm.MU[idx]
    sig = gmm.SIG[idx].unsqueeze(-1)
    z   = torch.randn((n, gmm.d), device=gmm.MU.device, generator=generator)
    return mu + z * sig

def logpdf_gmm_torch(gmm: GMMTorch, X: Tensor) -> Tensor:
    X = _expand_2d(X).to(gmm.MU.device)
    N, d = X.shape
    MU  = gmm.MU.unsqueeze(0)
    SIG = gmm.SIG.unsqueeze(0)
    PI  = gmm.PI.unsqueeze(0)
    diff2 = ((X.unsqueeze(1) - MU)**2).sum(-1)
    log_norm = -0.5 * d * math.log(2*math.pi) - d * torch.log(SIG + 1e-32)
    log_comp = log_norm - 0.5 * diff2 / (SIG**2 + 1e-32)
    lse = torch.logsumexp(log_comp + torch.log(PI + 1e-32), dim=1)
    return lse

# ---------- Constant-β closed forms ----------
def _a_minus_const(t: float, beta: float) -> float:
    if beta <= 0: 
        dt = max(0.0, 1.0 - float(t))
        return float('inf') if dt == 0.0 else 1.0/dt
    sb = math.sqrt(beta); dt = max(0.0, 1.0 - float(t))
    s = math.sinh(sb * dt)
    return float('inf') if s == 0.0 else sb * (math.cosh(sb*dt) / s)

def _b_minus_const(t: float, beta: float) -> float:
    if beta <= 0:
        dt = max(0.0, 1.0 - float(t))
        return float('inf') if dt == 0.0 else 1.0/dt
    sb = math.sqrt(beta); dt = max(0.0, 1.0 - float(t))
    s = math.sinh(sb * dt)
    return 0.0 if s == 0.0 else sb / s

def _c_minus_const(t: float, beta: float) -> float:
    return _a_minus_const(t, beta)

def _a_plus_1_const(beta: float) -> float:
    if beta <= 0: return 1.0
    sb = math.sqrt(beta); s = math.sinh(sb)
    return float('inf') if s == 0.0 else sb * (math.cosh(sb) / s)

class BetaScheduleConstTorch:
    def __init__(self, beta: float, device: Device = torch.device("cpu")):
        self.beta = float(beta); self.device = device
    def a_minus(self, t: float) -> float: return _a_minus_const(t, self.beta)
    def b_minus(self, t: float) -> float: return _b_minus_const(t, self.beta)
    def c_minus(self, t: float) -> float: return _c_minus_const(t, self.beta)
    def a_plus_1(self) -> float: return _a_plus_1_const(self.beta)
    def alpha_K_gamma(self, t: float):
        cm = self.c_minus(t); ap1 = self.a_plus_1()
        Kt = cm - ap1
        if not (Kt > 0.0 and math.isfinite(Kt)):
            raise RuntimeError(f"K(t) non-positive or non-finite at t={t:.6g}: {Kt}")
        alpha = self.b_minus(t) / Kt
        gamma = 1.0 / math.sqrt(Kt)
        return alpha, Kt, gamma

# ---------- PWC-β exact per-piece (NumPy parity) ----------
class BetaSchedulePWCTorch:
    """
    Exact PWC-β schedule translated from the NumPy API:
      • Build per-segment (a^-, b^-, c^-) by sweeping backward using the same closures.
      • Compute a^+(1) via the same forward sweep.
    """
    def __init__(self, betas, splits):
        import numpy as np, math
        betas  = np.asarray(betas,  float)
        splits = np.asarray(splits, float)
        assert splits[0] == 0.0 and splits[-1] == 1.0 and np.all(np.diff(splits) > 0)
        assert betas.size + 1 == splits.size
        self.betas, self.splits = betas, splits
        m = betas.size

        def last_piece(beta_i):
            def a_m(t): return _a_minus_const(t, float(beta_i))
            def b_m(t): return _b_minus_const(t, float(beta_i))
            def c_m(t): return _c_minus_const(t, float(beta_i))
            return a_m, b_m, c_m

        self._pieces = [None]*m
        iR = m-1
        aR, bR, cR = last_piece(betas[iR])
        self._pieces[iR] = (aR, bR, cR)

        aR0, bR0, cR0 = aR(float(splits[iR])), bR(float(splits[iR])), cR(float(splits[iR]))

        for j in reversed(range(m-1)):
            beta_j = float(betas[j]); rj = math.sqrt(beta_j)
            sLj, sRj = float(splits[j]), float(splits[j+1])
            aRj, bRj, cRj = aR0, bR0, cR0

            def make_piece(aRj, bRj, cRj, rj, beta_j, sRj):
                def a_m(t):
                    tau = max(0.0, sRj - float(t))
                    if rj == 0.0:  # β=0 on this piece
                        return aRj
                    th = math.tanh(rj * tau)
                    return rj * (aRj + rj * th) / (rj + aRj * th)
                def b_m(t):
                    at = a_m(t)
                    num = max(0.0, at*at - beta_j)
                    den = max(0.0, aRj*aRj - beta_j)
                    if den == 0.0: return bRj
                    return bRj * math.sqrt(num/den)
                def c_m(t):
                    at = a_m(t)
                    denom = (beta_j - aRj*aRj)
                    if denom == 0.0: return cRj
                    return cRj + (bRj*bRj)/denom * (aRj - at)
                return a_m, b_m, c_m

            aF, bF, cF = make_piece(aRj, bRj, cRj, rj, beta_j, sRj)
            self._pieces[j] = (aF, bF, cF)
            aR0, bR0, cR0 = aF(sLj), bF(sLj), cF(sLj)

        # Forward sweep to compute a^+(1)
        ap = None
        for k in range(m):
            r  = math.sqrt(float(betas[k]))
            dur = float(splits[k+1] - splits[k])
            if ap is None:
                if r == 0.0: ap = 1.0 if dur >= 1.0 else (float('inf') if dur == 0.0 else 1.0/dur)
                else:
                    s = math.sinh(r*dur); c = math.cosh(r*dur)
                    ap = float('inf') if s == 0.0 else r * (c/s)
            else:
                if r == 0.0:
                    ap = ap
                else:
                    rho = math.exp(-2.0*r*dur) * (ap - r)/(ap + r)
                    ap  = r * (1.0 + rho)/(1.0 - rho)
        self._a_plus_1 = float(ap)

    def _seg_idx(self, t: float) -> int:
        import numpy as np
        t = float(t)
        if t >= 1.0: return self.betas.size - 1
        i = int(np.searchsorted(self.splits, t, side="right") - 1)
        return max(0, min(i, self.betas.size - 1))

    def a_minus(self, t: float) -> float:
        i = self._seg_idx(t); return float(self._pieces[i][0](t))
    def b_minus(self, t: float) -> float:
        i = self._seg_idx(t); return float(self._pieces[i][1](t))
    def c_minus(self, t: float) -> float:
        i = self._seg_idx(t); return float(self._pieces[i][2](t))
    def a_plus_1(self) -> float:
        return self._a_plus_1

    def alpha_K_gamma(self, t: float):
        cm = self.c_minus(t); ap1 = self.a_plus_1()
        Kt = cm - ap1
        if not (Kt > 0.0 and math.isfinite(Kt)):
            raise RuntimeError(f"K(t) non-positive or non-finite at t={t:.6g}: {Kt}")
        alpha = self.b_minus(t) / Kt
        gamma = 1.0 / math.sqrt(Kt)
        return alpha, Kt, gamma

# ------------------------------
# Oracle ŷ(t;x) and control u*
# ------------------------------
def yhat_oracle_gmm_torch(X: Tensor, t: float, sched, gmm: GMMTorch) -> Tensor:
    X = _expand_2d(X).to(gmm.MU.device)
    alpha, Kt, _ = sched.alpha_K_gamma(t)
    m = alpha * X
    sig = gmm.SIG
    inv_sig2 = 1.0 / (sig**2 + 1e-32)
    lam = (Kt * inv_sig2) / (inv_sig2 + Kt)
    MU = gmm.MU
    dmu = MU.unsqueeze(0) - m.unsqueeze(1)
    r2 = (dmu**2).sum(-1)
    wexp = torch.exp(-0.5 * r2 * lam.unsqueeze(0)) * gmm.PI.unsqueeze(0)
    num = inv_sig2.unsqueeze(0).unsqueeze(-1) * MU.unsqueeze(0) + Kt * m.unsqueeze(1)
    den = (inv_sig2 + Kt).unsqueeze(0).unsqueeze(-1)
    mu_t = num / (den + 1e-32)
    Z = wexp.sum(dim=1, keepdim=True) + 1e-32
    yhat = (wexp.unsqueeze(-1) * mu_t).sum(dim=1) / Z
    return yhat

def control_u_star_torch(X: Tensor, t: float, sched, gmm: GMMTorch) -> Tensor:
    yhat = yhat_oracle_gmm_torch(X, t, sched, gmm)
    a_m = sched.a_minus(t); b_m = sched.b_minus(t)
    return b_m * yhat - a_m * _expand_2d(X).to(gmm.MU.device)

# ------------------------------
# Simulation (Euler–Maruyama, unit diffusion)
# ------------------------------
#@torch.no_grad()
def simulate_paths_torch(sched, gmm: GMMTorch, M: int = 4000, T: int = 1200, seed: int = 0, device: Device = torch.device("cpu")) -> Tensor:
    set_seed(seed)
    gmm = gmm.to(device)
    X = torch.zeros((M, gmm.d), device=device)
    dt = 1.0 / T; sqrt_dt = math.sqrt(dt)
    for n in range(T):
        t_mid = (n + 0.5) / T
        u = control_u_star_torch(X, t_mid, sched, gmm)
        X = X + u * dt + torch.randn_like(X) * sqrt_dt
    return X

# ------------------------------
# Sinkhorn W2 (balanced, uniform)
# ------------------------------
def sinkhorn_w2_torch(X: Tensor, Y: Tensor, reg: float = 0.02, iters: int = 300, tol: float = 1e-6):
    X = _expand_2d(X).contiguous(); Y = _expand_2d(Y).contiguous()
    n, d = X.shape; m, d2 = Y.shape; assert d == d2
    C = torch.cdist(X, Y)**2
    medianC = torch.median(C).clamp(min=1e-8).item()
    eps = float(reg if reg > 0 else 0.02 * medianC)
    K = torch.exp(-C / eps)
    u = torch.full((n,1), 1.0/n, device=X.device); v = torch.full((m,1), 1.0/m, device=Y.device)
    for _ in range(iters):
        u_prev = u
        Kv = K @ v; u = (1.0/n) / (Kv + 1e-32)
        Ku = K.t() @ u; v = (1.0/m) / (Ku + 1e-32)
        if torch.max(torch.abs(u - u_prev)) < tol: break
    P = u * K * v.t()
    w2_eps = torch.sum(P * C).item()
    return math.sqrt(max(w2_eps, 0.0)), {"method":"sinkhorn","eps":eps}

def gmm_from_numpy(mu_np, sig_np, pi_np, device: Device = torch.device("cpu")) -> GMMTorch:
    MU  = torch.as_tensor(mu_np, dtype=torch.float32, device=device)
    SIG = torch.as_tensor(sig_np, dtype=torch.float32, device=device)
    PI  = torch.as_tensor(pi_np,  dtype=torch.float32, device=device)
    PI  = PI / PI.sum()
    return GMMTorch(MU, SIG, PI)
