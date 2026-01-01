"""
Exact Daum-Huang (EDH) Particle Flow in NumPy.
Based on Li & Coates (2017): "Particle Filtering With Invertible Particle Flow"
"""

import numpy as np
from typing import Callable, Tuple, Optional


def generate_lambda_schedule(
    num_steps: int = 29,
    ratio: float = 1.2,
) -> np.ndarray:
    """
    Generate exponentially spaced step sizes for particle flow.
    
    Args:
        num_steps: Number of flow steps (default 29)
        ratio: Ratio between consecutive step sizes (default 1.2)
        
    Returns:
        lambdas: [num_steps] array of lambda values (cumulative)
        step_sizes: [num_steps] array of step sizes (epsilon_j)
    """
    
    if abs(ratio - 1.0) < 1e-10:
        # Uniform steps
        step_sizes = np.ones(num_steps) / num_steps
    else:
        eps_1 = (1 - ratio) / (1 - ratio**num_steps)
        step_sizes = eps_1 * (ratio ** np.arange(num_steps))
    
    lambdas = np.cumsum(step_sizes)
    
    lambdas[-1] = 1.0
    
    return lambdas, step_sizes


def edh_flow_parameters(
    eta_bar: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    z: np.ndarray,
    h_fn: Callable[[np.ndarray], np.ndarray],
    H_jac_fn: Callable[[np.ndarray], np.ndarray],
    lam: float,
    eta_bar_0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute EDH flow parameters.
    
    Args:
        eta_bar: [nx] Current mean of particle cloud
        P: [nx, nx] Predicted covariance
        R: [ny, ny] Observation noise covariance
        z: [ny] Current observation
        h_fn: Observation function h(x) -> y
        H_jac_fn: Jacobian function H(x) -> [ny, nx]
        lam: Current pseudo-time \lambda ∈ [0, 1]
        eta_bar_0: [nx] Initial mean (at \lambda=0)
        
    Returns:
        A: [nx, nx] Flow matrix
        b: [nx] Flow offset
    """
    nx = len(eta_bar)
    ny = len(z)
    
    H = H_jac_fn(eta_bar) 
    h_bar = h_fn(eta_bar[None, :])[0]  
    e = h_bar - H @ eta_bar  
    
    HPHT = H @ P @ H.T 
    S = lam * HPHT + R 
    
    S_inv_H = np.linalg.solve(S, H)
    
    A = -0.5 * P @ H.T @ S_inv_H 

    I = np.eye(nx)
    I_plus_lam_A = I + lam * A 
    I_plus_2lam_A = I + 2 * lam * A 
    
    R_inv_residual = np.linalg.solve(R, z - e)
    
    term1 = I_plus_lam_A @ P @ H.T @ R_inv_residual

    term2 = A @ eta_bar_0 
    
    b = I_plus_2lam_A @ (term1 + term2) 
    
    return A, b


def edh_flow_step(
    particles: np.ndarray,
    eta_bar: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    step_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a sinlge step of EDH flow.
    
    Args:
        particles: [N, nx] Current particles
        eta_bar: [nx] Current mean
        A: [nx, nx] Flow matrix
        b: [nx] Flow offset
        step_size: \epsilon (step size)
        
    Returns:
        particles_new: [N, nx] Updated particles
        eta_bar_new: [nx] Updated mean
    """
    
    I_plus_eps_A = np.eye(A.shape[0]) + step_size * A
    eps_b = step_size * b
    
    particles_new = particles @ I_plus_eps_A.T + eps_b[None, :]
    eta_bar_new = eta_bar @ I_plus_eps_A.T + eps_b
    
    return particles_new, eta_bar_new


def edh_flow(
    particles: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    z: np.ndarray,
    h_fn: Callable[[np.ndarray], np.ndarray],
    H_jac_fn: Callable[[np.ndarray], np.ndarray],
    eta_bar_0: Optional[np.ndarray] = None,
    num_steps: int = 29,
    step_ratio: float = 1.2,
) -> np.ndarray:
    """
    Apply full EDH flow to evolve particles from prior to posterior.
    
    Args:
        particles: [N, nx] Particles from predictive prior
        P: [nx, nx] Predicted covariance
        R: [ny, ny] Observation noise covariance
        z: [ny] Current observation
        h_fn: Observation function h(x) -> y, accepts [N, nx] -> [N, ny]
        H_jac_fn: Jacobian function H(x) -> [ny, nx], accepts [nx] -> [ny, nx]
        eta_bar_0: [nx] Initial mean for flow (EKF predicted mean). If None, use particle mean.
        num_steps: Number of flow steps (default 29)
        step_ratio: Ratio between consecutive step sizes (default 1.2)
        
    Returns:
        particles_new: [N, nx] Particles after flow (approximating posterior)
    """
    N, nx = particles.shape
    
    lambdas, step_sizes = generate_lambda_schedule(num_steps, step_ratio)
    
    if eta_bar_0 is None:
        eta_bar_0 = np.mean(particles, axis=0)
    eta_bar = eta_bar_0.copy()
    
    # Flow through pseudo-time
    lam = 0.0
    for j in range(num_steps):
        eps_j = step_sizes[j]
        lam_j = lam + eps_j 
        
        A, b = edh_flow_parameters(
            eta_bar, P, R, z, h_fn, H_jac_fn, lam_j, eta_bar_0
        )
        
        particles, eta_bar = edh_flow_step(particles, eta_bar, A, b, eps_j)
        
        lam = lam_j
    
    return particles


def edh_flow_with_weights(
    particles_prior: np.ndarray,
    particles_posterior: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    z: np.ndarray,
    h_fn: Callable[[np.ndarray], np.ndarray],
    x_prev: np.ndarray,
    A_dyn: np.ndarray,
) -> np.ndarray:
    """
    Compute importance weights after EDH flow.
    
    Args:
        particles_prior: [N, nx] Particles before flow (η_0)
        particles_posterior: [N, nx] Particles after flow (η_1)
        P: [nx, nx] Predicted covariance
        Q: [nx, nx] Process noise covariance  
        R: [ny, ny] Observation noise covariance
        z: [ny] Current observation
        h_fn: Observation function
        x_prev: [N, nx] Previous state particles
        A_dyn: [nx, nx] Dynamics matrix
        
    Returns:
        log_weights: [N] Unnormalized log importance weights
    """
    N, nx = particles_prior.shape
    ny = len(z)
    
    mean_prior = x_prev @ A_dyn.T  
    diff_prior = particles_prior - mean_prior
    
    Q_inv = np.linalg.inv(Q)
    Q_logdet = np.linalg.slogdet(Q)[1]
    
    log_p_prior = -0.5 * (
        nx * np.log(2 * np.pi) + Q_logdet +
        np.sum(diff_prior @ Q_inv * diff_prior, axis=1)
    ) 

    diff_posterior = particles_posterior - mean_prior
    
    log_p_posterior_given_prev = -0.5 * (
        nx * np.log(2 * np.pi) + Q_logdet +
        np.sum(diff_posterior @ Q_inv * diff_posterior, axis=1)
    ) 
    
    h_vals = h_fn(particles_posterior) 
    residual = z[None, :] - h_vals  
    
    R_inv = np.linalg.inv(R)
    R_logdet = np.linalg.slogdet(R)[1]
    
    log_p_obs = -0.5 * (
        ny * np.log(2 * np.pi) + R_logdet +
        np.sum(residual @ R_inv * residual, axis=1)
    )  
    
    log_weights = log_p_posterior_given_prev + log_p_obs - log_p_prior
    
    return log_weights
