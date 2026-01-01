"""
Particle Flow Particle Filter (PF-PF) with Exact Daum-Huang Flow.
Based on Li & Coates (2017): "Particle Filtering With Invertible Particle Flow"
"""

import numpy as np
from numpy.random import Generator, default_rng
from typing import Optional, Dict, Tuple, Callable, Literal

from .numpy_ssm import NumpySSM
from .numpy_edh_flow import edh_flow, generate_lambda_schedule


def _systematic_resample(weights: np.ndarray, rng: Generator) -> np.ndarray:
    N = len(weights)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    u0 = rng.uniform(0, 1.0 / N)
    u = u0 + np.arange(N) / N
    idx = np.searchsorted(cdf, u, side='right')
    idx = np.minimum(idx, N - 1)
    return idx


def _logsumexp(x: np.ndarray) -> float:
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def _ekf_predict(
    m: np.ndarray,
    P: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EKF prediction step for covariance estimation.
    
    Args:
        m: [nx] Prior mean
        P: [nx, nx] Prior covariance
        A: [nx, nx] Dynamics matrix (F_jac)
        Q: [nx, nx] Process noise covariance
        
    Returns:
        m_pred: [nx] Predicted mean
        P_pred: [nx, nx] Predicted covariance
    """
    m_pred = A @ m
    P_pred = A @ P @ A.T + Q
    P_pred = 0.5 * (P_pred + P_pred.T) 
    return m_pred, P_pred


def _ekf_update(
    m: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    h_fn: Callable[[np.ndarray], np.ndarray],
    H_jac_fn: Callable[[np.ndarray], np.ndarray],
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EKF update step for covariance estimation (Joseph form).
    
    Args:
        m: [nx] Predicted mean
        P: [nx, nx] Predicted covariance
        y: [ny] Observation
        h_fn: Observation function
        H_jac_fn: Jacobian of observation function
        R: [ny, ny] Observation noise covariance
        
    Returns:
        m_upd: [nx] Updated mean
        P_upd: [nx, nx] Updated covariance
    """
    nx = len(m)
    
    H = H_jac_fn(m)
    h_m = h_fn(m[None, :])[0] 
    
    v = y - h_m 
    
    S = H @ P @ H.T + R 
    
    # Kalman gain
    K = np.linalg.solve(S.T, H @ P.T).T 
    
    # Updated mean
    m_upd = m + K @ v
    
    # Updated covariance (Joseph form for stability)
    I = np.eye(nx)
    IKH = I - K @ H
    P_upd = IKH @ P @ IKH.T + K @ R @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)
    
    return m_upd, P_upd


def particle_filter_edh(
    ssm: NumpySSM,
    ys: np.ndarray,
    h_fn: Callable[[np.ndarray], np.ndarray],
    H_jac_fn: Callable[[np.ndarray], np.ndarray],
    num_particles: int = 1000,
    num_flow_steps: int = 29,
    flow_step_ratio: float = 1.2,
    resample: Literal["none", "ess", "always"] = "ess",
    ess_threshold: float = 0.5,
    resampling_method: Literal["systematic", "multinomial"] = "systematic",
    return_particles: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Particle Flow Particle Filter with EDH flow.
    Args:
        ssm: NumpySSM instance (must have linear dynamics)
        ys: [T, ny] Observations
        h_fn: Observation function h(x), [N, nx] -> [N, ny]
        H_jac_fn: Jacobian of h(x), [nx] -> [ny, nx]
        num_particles: Number of particles (default 1000)
        num_flow_steps: Number of EDH flow steps (default 29)
        flow_step_ratio: Ratio between consecutive step sizes (default 1.2)
        resample: Resampling policy (default "ess")
        ess_threshold: ESS threshold for resampling (default 0.5)
        resampling_method: "systematic" or "multinomial"
        return_particles: If True, return particle history
        seed: Random seed
        
    Returns:
        ms: [T+1, nx] Filtered means
        Ps: [T+1, nx, nx] Filtered covariances
        info: Dict with diagnostics
    """
    ys = np.asarray(ys)
    T = ys.shape[0]
    N = num_particles
    
    A = ssm.A  # [nx, nx]
    Q = ssm.Q  # [nx, nx]
    R = ssm.R  # [ny, ny]
    m0 = ssm.m0  # [nx]
    P0 = ssm.P0  # [nx, nx]
    nx = ssm.nx
    ny = ys.shape[1] if ys.ndim > 1 else 1
    
    rng = default_rng(seed)
    
    x = ssm.sample_initial(N, rng)  # [N, nx]
    
    log_N = np.log(N)
    logw = -log_N * np.ones(N)
    
    m_ekf = m0.copy()
    P_ekf = P0.copy()
    
    w0 = np.ones(N) / N
    m0_filt = np.sum(w0[:, None] * x, axis=0)
    xc0 = x - m0_filt[None, :]
    P0_filt = np.einsum('n,ni,nj->ij', w0, xc0, xc0)
    P0_filt = 0.5 * (P0_filt + P0_filt.T)
    
    ms = [m0_filt]
    Ps = [P0_filt]
    
    ess_hist = []
    maxw_hist = []
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
    
    x_prev = x.copy()
    
    for t in range(1, T + 1):
        y_t = ys[t - 1]  # [ny]
        if y_t.ndim == 0:
            y_t = np.array([y_t])
        
        m_pred, P_pred = _ekf_predict(m_ekf, P_ekf, A, Q)
        
        x_prior = ssm.sample_transition(x, rng)
        
        eta_bar_0 = np.mean(x_prior, axis=0) 
        
        x_posterior = edh_flow(
            particles=x_prior,
            P=P_pred,
            R=R,
            z=y_t,
            h_fn=h_fn,
            H_jac_fn=H_jac_fn,
            eta_bar_0=eta_bar_0,
            num_steps=num_flow_steps,
            step_ratio=flow_step_ratio,
        )
        
        mean_trans = x @ A.T  
        diff_prior = x_prior - mean_trans
        Q_inv = np.linalg.inv(Q)
        Q_logdet = np.linalg.slogdet(Q)[1]
        log_p_prior = -0.5 * (
            nx * np.log(2 * np.pi) + Q_logdet +
            np.sum(diff_prior @ Q_inv * diff_prior, axis=1)
        )
        
        diff_posterior = x_posterior - mean_trans
        log_p_posterior_trans = -0.5 * (
            nx * np.log(2 * np.pi) + Q_logdet +
            np.sum(diff_posterior @ Q_inv * diff_posterior, axis=1)
        )
        
        h_vals = h_fn(x_posterior) 
        residual = y_t[None, :] - h_vals
        R_inv = np.linalg.inv(R)
        R_logdet = np.linalg.slogdet(R)[1]
        log_p_obs = -0.5 * (
            ny * np.log(2 * np.pi) + R_logdet +
            np.sum(residual @ R_inv * residual, axis=1)
        )
        
        log_w_increment = log_p_posterior_trans + log_p_obs - log_p_prior
        
        logw = logw + log_w_increment
        
        # Log marginal likelihood increment
        logZt = _logsumexp(logw)
        logZ_hist.append(logZt)
        loglik_cum += logZt
        

        logw = logw - _logsumexp(logw)
        w = np.exp(logw)
        
        ess = 1.0 / np.sum(w ** 2)
        ess_hist.append(ess)
        maxw_hist.append(np.max(w))
        
        m_filt = np.sum(w[:, None] * x_posterior, axis=0)
        xc = x_posterior - m_filt[None, :]
        P_filt = np.einsum('n,ni,nj->ij', w, xc, xc)
        P_filt = 0.5 * (P_filt + P_filt.T)
        
        ms.append(m_filt)
        Ps.append(P_filt)
        
        # Update EKF covariance using standard EKF update
        _, P_ekf_updated = _ekf_update(m_pred, P_pred, y_t, h_fn, H_jac_fn, R)
        
        # Use particle filtered mean as EKF mean (to keep EKF synced with particles)
        m_ekf = m_filt.copy()
        P_ekf = P_ekf_updated
        
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
            else:
                idx = rng.choice(N, size=N, replace=True, p=w)
            
            x_posterior = x_posterior[idx]
            logw = -log_N * np.ones(N)
            
            unique_anc = len(np.unique(idx))
            unique_anc_hist.append(float(unique_anc))
            resampled_hist.append(True)
        else:
            unique_anc_hist.append(float(N))
            resampled_hist.append(False)
        
        x = x_posterior
        x_prev = x.copy()
        
        if return_particles:
            x_hist.append(x.copy())
            w_hist.append(np.exp(logw))
    
    ms = np.stack(ms, axis=0)
    Ps = np.stack(Ps, axis=0)
    
    info = {
        "ess": np.array(ess_hist),
        "max_weight": np.array(maxw_hist),
        "resampled": np.array(resampled_hist),
        "unique_ancestors": np.array(unique_anc_hist),
        "logZ": np.array(logZ_hist),
        "loglik_cum": loglik_cum,
    }
    
    if return_particles:
        info["particles"] = np.stack(x_hist, axis=0)
        info["weights"] = np.stack(w_hist, axis=0)
    
    return ms, Ps, info
