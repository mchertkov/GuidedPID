
# AdaPID PyTorch API (single file) — parity with NumPy API
# SDE convention: dX_t = u*_t(X_t) dt + dW_t  (unit diffusion; no sqrt(2))
# Equation pointers:
#   α(t), K(t), γ(t): AdaPID Eq. (2.x); alpha_K_gamma_from_schedule matches NumPy.  fileciteturn0file1
#   u*_t(x) = b^-_t ŷ(t;x) - a^-_t x; ŷ(t;x) oracle for GMM matches the NumPy closed form.  fileciteturn0file1

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
    """Exact i.i.d. samples from isotropic GMM."""
    if generator is None:
        generator = torch.Generator(device=gmm.MU.device)
    idx = torch.multinomial(gmm.PI, num_samples=n, replacement=True, generator=generator)  # (n,)
    mu  = gmm.MU[idx]                     # (n,d)
    sig = gmm.SIG[idx].unsqueeze(-1)      # (n,1)  std
    z   = torch.randn((n, gmm.d), device=gmm.MU.device, generator=generator)
    return mu + z * sig                   # use std, not sqrt(var)

def logpdf_gmm_torch(gmm: GMMTorch, X: Tensor) -> Tensor:
    """Log-density of isotropic GMM; SIG is std (σ)."""
    X = _expand_2d(X).to(gmm.MU.device)
    N, d = X.shape
    MU  = gmm.MU.unsqueeze(0)            # (1,K,d)
    SIG = gmm.SIG.unsqueeze(0)           # (1,K) std
    PI  = gmm.PI.unsqueeze(0)            # (1,K)
    diff2 = ((X.unsqueeze(1) - MU)**2).sum(-1)            # (N,K)
    log_norm = -0.5 * d * math.log(2*math.pi) - d * torch.log(SIG + 1e-32)  # (1,K)
    log_comp = log_norm - 0.5 * diff2 / (SIG**2 + 1e-32)                     # (N,K)
    lse = torch.logsumexp(log_comp + torch.log(PI + 1e-32), dim=1)           # (N,)
    return lse

# --------------------------------
# Constant-β schedule (matches NumPy)
# --------------------------------
def _a_minus_const(t: float, beta: float) -> float:
    if beta <= 0: return 1.0 / (1.0 - t)
    sb = math.sqrt(beta); return sb / math.tanh((1.0 - t) * sb)

def _b_minus_const(t: float, beta: float) -> float:
    if beta <= 0: return 1.0 / (1.0 - t)
    sb = math.sqrt(beta); return sb / math.sinh((1.0 - t) * sb)

def _c_minus_const(t: float, beta: float) -> float:
    return _a_minus_const(t, beta)

def _a_plus_1_const(beta: float) -> float:
    if beta <= 0: return 1.0
    sb = math.sqrt(beta); return sb / math.tanh(sb)

class BetaScheduleConstTorch:
    def __init__(self, beta: float, device: Device = torch.device("cpu")):
        self.beta = float(beta); self.device = device

    def a_minus(self, t: float) -> float: return _a_minus_const(t, self.beta)
    def b_minus(self, t: float) -> float: return _b_minus_const(t, self.beta)
    def c_minus(self, t: float) -> float: return _c_minus_const(t, self.beta)
    def a_plus_1(self) -> float: return _a_plus_1_const(self.beta)

    def alpha_K_gamma(self, t: float):
        # NumPy parity: raises if K<=0 there; here we just compute and rely on caller to use interior times.
        cm = self.c_minus(t); ap1 = self.a_plus_1()
        Kt = cm - ap1
        alpha = self.b_minus(t) / Kt
        gamma = 1.0 / math.sqrt(Kt)
        return alpha, Kt, gamma

# --------------------------------
# Oracle ŷ(t;x) for GMM — exact NumPy parity
# --------------------------------
def yhat_oracle_gmm_torch(X: Tensor, t: float, sched: BetaScheduleConstTorch, gmm: GMMTorch) -> Tensor:
    """
    NumPy-equivalent oracle:
      alpha, Kt, _ = alpha_K_gamma_from_schedule(...)
      m = alpha * X
      inv_sig2 = 1 / (σ^2)
      λ_k = (Kt * inv_sig2_k) / (inv_sig2_k + Kt)
      w_k ∝ π_k * exp(-0.5 * ||μ_k - m||^2 * λ_k)
      μ_tk = (inv_sig2_k * μ_k + Kt * m) / (inv_sig2_k + Kt)
      ŷ = sum_k [ w_k * μ_tk ] / sum_k [ w_k ]
    """
    X = _expand_2d(X).to(gmm.MU.device)       # (M,d)
    alpha, Kt, _ = sched.alpha_K_gamma(t)
    m = alpha * X                              # (M,d)

    sig = gmm.SIG                              # (K,) std
    inv_sig2 = 1.0 / (sig**2 + 1e-32)          # (K,)
    lam = (Kt * inv_sig2) / (inv_sig2 + Kt)    # (K,)

    MU = gmm.MU                                # (K,d)
    dmu = MU.unsqueeze(0) - m.unsqueeze(1)     # (M,K,d)
    r2 = (dmu**2).sum(-1)                      # (M,K)

    wexp = torch.exp(-0.5 * r2 * lam.unsqueeze(0)) * gmm.PI.unsqueeze(0)  # (M,K)

    num = inv_sig2.unsqueeze(0).unsqueeze(-1) * MU.unsqueeze(0) + Kt * m.unsqueeze(1)   # (M,K,d)
    den = (inv_sig2 + Kt).unsqueeze(0).unsqueeze(-1)                                    # (M,K,1)
    mu_t = num / (den + 1e-32)                    # (M,K,d)

    Z = wexp.sum(dim=1, keepdim=True) + 1e-32     # (M,1)
    yhat = (wexp.unsqueeze(-1) * mu_t).sum(dim=1) / Z   # (M,d)
    return yhat

# --------------------------------
# Optimal control u*_t(x)
# --------------------------------
def control_u_star_torch(X: Tensor, t: float, sched: BetaScheduleConstTorch, gmm: GMMTorch) -> Tensor:
    yhat = yhat_oracle_gmm_torch(X, t, sched, gmm)
    a_m = sched.a_minus(t); b_m = sched.b_minus(t)
    return b_m * yhat - a_m * _expand_2d(X).to(gmm.MU.device)

# --------------------------------
# Simulation (Euler–Maruyama, unit diffusion)
# --------------------------------
@torch.no_grad()
def simulate_paths_torch(sched: BetaScheduleConstTorch, gmm: GMMTorch, M: int = 4000, T: int = 1200, seed: int = 0, device: Device = torch.device("cpu")) -> Tensor:
    set_seed(seed)
    gmm = gmm.to(device)
    X = torch.zeros((M, gmm.d), device=device)
    dt = 1.0 / T; sqrt_dt = math.sqrt(dt)
    for n in range(T):
        t_mid = (n + 0.5) / T
        u = control_u_star_torch(X, t_mid, sched, gmm)
        X = X + u * dt + torch.randn_like(X) * sqrt_dt
    return X

# --------------------------------
# Sinkhorn W2 (balanced, uniform)
# --------------------------------
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
    SIG = torch.as_tensor(sig_np, dtype=torch.float32, device=device)   # std
    PI  = torch.as_tensor(pi_np,  dtype=torch.float32, device=device)
    PI  = PI / PI.sum()
    return GMMTorch(MU, SIG, PI)
