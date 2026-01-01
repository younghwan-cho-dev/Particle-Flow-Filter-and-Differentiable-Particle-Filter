import numpy as np
from numpy.random import Generator, default_rng
from typing import Optional, Dict, Tuple, Literal

from .numpy_ssm import NumpySSM


def _systematic_resample(weights: np.ndarray, rng: Generator) -> np.ndarray:
    """
    Systematic resampling.
    
    Args:
        weights: [N] Normalized weights (sum to 1)
        rng: NumPy random generator
        
    Returns:
        idx: [N] Resampled indices
    """
    N = len(weights)
    
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0 
    
    u0 = rng.uniform(0, 1.0 / N)
    u = u0 + np.arange(N) / N
    
    idx = np.searchsorted(cdf, u, side='left')
    idx = np.minimum(idx, N - 1)
    
    return idx


def _multinomial_resample(weights: np.ndarray, rng: Generator) -> np.ndarray:
    """
    Multinomial resampling.
    
    Args:
        weights: [N] Normalized weights (sum to 1)
        rng: NumPy random generator
        
    Returns:
        idx: [N] Resampled indices
    """
    N = len(weights)
    return rng.choice(N, size=N, replace=True, p=weights)


def particle_filter_numpy(
    ssm: NumpySSM,
    ys: np.ndarray,
    num_particles: int = 1000,
    resample: Literal["none", "ess", "always"] = "ess",
    ess_threshold: float = 0.5,
    resampling_method: Literal["systematic", "multinomial"] = "systematic",
    return_particles: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Bootstrap/SIR Particle Filter.
    
    Args:
        ssm: NumpySSM instance
        ys: [T, ny] Observations (y_1, ..., y_T)
        num_particles: Number of particles
        resample: Resampling policy ("none", "ess", "always")
        ess_threshold: ESS threshold for resampling (fraction of N)
        resampling_method: "systematic" or "multinomial"
        return_particles: If True, return particle history
        seed: Random seed
        
    Returns:
        ms: [T+1, nx] Filtered means (m_0, m_1, ..., m_T)
        Ps: [T+1, nx, nx] Filtered covariances
        info: Dict with diagnostics
    """
    ys = np.asarray(ys)
    T = ys.shape[0]
    N = num_particles
    nx = ssm.nx
    
    rng = default_rng(seed)
    
    x = ssm.sample_initial(N, rng)
    
    log_N = np.log(N)
    logw = -log_N * np.ones(N)
    
    w0 = np.ones(N) / N
    m0 = np.sum(w0[:, None] * x, axis=0) 
    xc0 = x - m0[None, :]
    P0 = np.einsum('n,ni,nj->ij', w0, xc0, xc0)
    P0 = 0.5 * (P0 + P0.T)
    
    ms = [m0]
    Ps = [P0]
    
    ess_hist = []
    maxw_hist = []
    ent_hist = []
    resampled_hist = []
    unique_anc_hist = []
    logZ_hist = []
    loglik_cum = 0.0
    
    if return_particles:
        x_hist = [x.copy()]
        w_hist = [np.exp(logw)]
    else:
        x_hist = None
        w_hist = None
    
    for t in range(1, T + 1):
        y_t = ys[t - 1]  

        x = ssm.sample_transition(x, rng) 
        
        loglik_i = ssm.observation_loglik(x, y_t)
        
        logZt = _logsumexp(logw + loglik_i)
        logZ_hist.append(logZt)
        loglik_cum += logZt
        
        logw = logw + loglik_i
        logw = logw - _logsumexp(logw) 
        w = np.exp(logw)
        
        ess = 1.0 / np.sum(w ** 2)
        ess_hist.append(ess)
        maxw_hist.append(np.max(w))
        ent_hist.append(-np.sum(w * np.log(w + 1e-12)))
        
        m_filt = np.sum(w[:, None] * x, axis=0)
        xc = x - m_filt[None, :]
        P_filt = np.einsum('n,ni,nj->ij', w, xc, xc)
        P_filt = 0.5 * (P_filt + P_filt.T)
        
        ms.append(m_filt)
        Ps.append(P_filt)
        
        if resample == "always":
            do_resample = True
        elif resample == "ess":
            do_resample = ess < ess_threshold * N
        elif resample == "none":
            do_resample = False
        else:
            raise ValueError(f"Unknown resample policy: {resample}")
        
        if do_resample:
            if resampling_method == "systematic":
                idx = _systematic_resample(w, rng)
            elif resampling_method == "multinomial":
                idx = _multinomial_resample(w, rng)
            else:
                raise ValueError(f"Unknown resampling_method: {resampling_method}")
            
            x = x[idx]
            logw = -log_N * np.ones(N)
            
            unique_anc = len(np.unique(idx))
            unique_anc_hist.append(float(unique_anc))
            resampled_hist.append(True)
        else:
            unique_anc_hist.append(float(N))
            resampled_hist.append(False)
        
        if return_particles:
            x_hist.append(x.copy())
            w_hist.append(np.exp(logw))
    
    ms = np.stack(ms, axis=0)  
    Ps = np.stack(Ps, axis=0)  
    
    info = {
        "ess": np.array(ess_hist),                    
        "max_weight": np.array(maxw_hist),            
        "entropy": np.array(ent_hist),                
        "resampled": np.array(resampled_hist),        
        "unique_ancestors": np.array(unique_anc_hist),
        "logZ": np.array(logZ_hist),                  
        "loglik_cum": loglik_cum,
    }
    
    if return_particles:
        info["particles"] = np.stack(x_hist, axis=0) 
        info["weights"] = np.stack(w_hist, axis=0)   
    
    return ms, Ps, info


def _logsumexp(x: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp.
    
    Args:
        x: [N] array
        
    Returns:
        log(sum(exp(x)))
    """
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))
