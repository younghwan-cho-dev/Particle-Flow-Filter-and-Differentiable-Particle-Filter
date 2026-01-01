"""
Factory functions for NumPy-based State Space Models.

Models:
1. 1D Random Walk
2. Multivariate Linear Gaussian
3. Range-Bearing (nonlinear obs with Student-t noise)
4. Acoustic Tracking (nonlinear obs with Gaussian noise)
"""

import numpy as np
from numpy.random import Generator
from scipy import stats
from typing import Optional, Tuple

from .numpy_ssm import NumpySSM

def make_random_walk_ssm_np(
    Q: float = 0.2,
    R: float = 0.5,
    m0: float = 0.3,
    P0: float = 1.2,
) -> NumpySSM:
    """
    1D Random Walk model.
    
    Args:
        Q: Process noise variance
        R: Observation noise variance
        m0: Initial mean
        P0: Initial variance
        
    Returns:
        NumpySSM instance
    """
    A = np.array([[1.0]])
    Q_mat = np.array([[Q]])
    C = np.array([[1.0]])
    R_mat = np.array([[R]])
    m0_vec = np.array([m0])
    P0_mat = np.array([[P0]])
    
    return NumpySSM(
        A=A, Q=Q_mat, m0=m0_vec, P0=P0_mat,
        C=C, R=R_mat,
    )


def make_linear_gaussian_ssm_np(
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
) -> NumpySSM:
    """
    Multivariate Linear Gaussian SSM.
    Args:
        A: [nx, nx] State transition matrix
        C: [ny, nx] Observation matrix
        Q: [nx, nx] Process noise covariance
        R: [ny, ny] Observation noise covariance
        m0: [nx] Initial mean
        P0: [nx, nx] Initial covariance
        
    Returns:
        NumpySSM instance
    """
    return NumpySSM(
        A=np.asarray(A),
        Q=np.asarray(Q),
        m0=np.asarray(m0),
        P0=np.asarray(P0),
        C=np.asarray(C),
        R=np.asarray(R),
    )

def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _h_range_bearing(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Range-bearing observation function.
    
    Args:
        x: [N, 4] states with columns [px, py, vx, vy]
           
    Returns:
        y: [N, 2] or [2] observations [range, bearing]
    """
    single = x.ndim == 1
    if single:
        x = x[None, :]
    
    px, py = x[:, 0], x[:, 1]
    r = np.sqrt(px**2 + py**2 + eps)
    th = np.arctan2(py, px)
    
    y = np.stack([r, th], axis=-1)
    
    if single:
        return y[0]
    return y


def make_range_bearing_ssm_np(
    dt: float = 0.01,
    q_diag: float = 0.01,
    nu: float = 2.0,
    s_r: float = 0.01,
    s_th: float = 0.01,
    m0: Tuple[float, ...] = (1.0, 0.5, 0.01, 0.01),
    P0_diag: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.1),
    eps: float = 1e-6,
) -> NumpySSM:
    """
    Range-Bearing SSM with Student-t measurement noise.
    
    Args:
        dt: Time step
        q_diag: Process noise diagonal value
        nu: Degrees of freedom for Student-t
        s_r: Scale for range noise
        s_th: Scale for bearing noise
        m0: Initial state mean (px, py, vx, vy)
        P0_diag: Initial state variance diagonal
        eps: Small constant for numerical stability
        
    Returns:
        NumpySSM instance
    """
    nx = 4
    ny = 2
    
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ])
    
    # Process noise
    Q = np.diag(np.full(nx, q_diag))
    Q = 0.5 * (Q + Q.T) + eps * np.eye(nx)
    
    # Initial state
    m0_vec = np.array(m0)
    P0_mat = np.diag(np.array(P0_diag))
    P0_mat = 0.5 * (P0_mat + P0_mat.T) + eps * np.eye(nx)
    
    def obs_loglik_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Log likelihood of Student-t range-bearing.
        
        Args:
            x: [N, 4] particles
            y: [2] observation [range, bearing]
            
        Returns:
            loglik: [N] log-likelihoods
        """
        y_pred = _h_range_bearing(x)
        
        res_r = y[0] - y_pred[:, 0]
        res_th = _wrap_angle(y[1] - y_pred[:, 1])
        
        loglik_r = stats.t.logpdf(res_r, df=nu, loc=0, scale=s_r)
        loglik_th = stats.t.logpdf(res_th, df=nu, loc=0, scale=s_th)
        
        return loglik_r + loglik_th
    
    def obs_sample_fn(x: np.ndarray, rng: Generator) -> np.ndarray:
        """
        Sample y from Student-t range-bearing.
        
        Args:
            x: [4] single state
            rng: NumPy random generator
            
        Returns:
            y: [2] observation [range, bearing]
        """
        y_mean = _h_range_bearing(x) 
        
        u_r = rng.uniform()
        u_th = rng.uniform()
        noise_r = stats.t.ppf(u_r, df=nu, loc=0, scale=s_r)
        noise_th = stats.t.ppf(u_th, df=nu, loc=0, scale=s_th)
        
        r = y_mean[0] + noise_r
        th = _wrap_angle(y_mean[1] + noise_th)
        
        return np.array([r, th])
    
    ssm = NumpySSM(
        A=A, Q=Q, m0=m0_vec, P0=P0_mat,
        obs_loglik_fn=obs_loglik_fn,
        obs_sample_fn=obs_sample_fn,
    )
    ssm.ny = ny 
    
    return ssm


def make_acoustic_ssm_np(
    num_targets: int = 4,
    num_sensors_per_side: int = 5,
    sensor_range: Tuple[float, float] = (0.0, 40.0),
    dt: float = 1.0,
    source_amplitude: float = 10.0,
    obs_noise_std: float = 0.1,
    m0_positions: Optional[np.ndarray] = None,
    m0_velocities: Optional[np.ndarray] = None,
    dist_offset: float = 0.1,
    deterministic_init: bool = False,
    use_paper_Q: bool = True,
    eps: float = 1e-6,
) -> NumpySSM:
    """
    Multi-target Acoustic Tracking SSM from Li & Coates (2017).
    
    State: x = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...] for num_targets
    State dimension: num_targets * 4
        
    Args:
        num_targets: Number of acoustic sources
        num_sensors_per_side: Grid size (total sensors = num_sensors_per_side**2)
        sensor_range: (min, max) for sensor grid
        dt: Time step
        source_amplitude: Amplitude (Amp=10 in paper)
        obs_noise_std: Observation noise std (0.1 in paper, variance=0.01)
        m0_positions: [num_targets, 2] Initial positions
        m0_velocities: [num_targets, 2] Initial velocities
        dist_offset: d0, prevents division by zero
        deterministic_init: If True, small init variance (for simulation)
        use_paper_Q: If True, use paper's larger Q for filtering. If False, use Q_real.
        eps: Numerical stability constant
        
    Returns:
        NumpySSM instance with additional attributes:
            Q_filter: Process noise for filtering (larger)
            Q_real: Process noise for simulation (smaller)
            sigma0: Initial std [10, 10, 1, 1] per target for filter init
    """
    nx = num_targets * 4
    num_sensors = num_sensors_per_side ** 2
    ny = num_sensors
    
    sensor_1d = np.linspace(sensor_range[0], sensor_range[1], num_sensors_per_side)
    sensor_xx, sensor_yy = np.meshgrid(sensor_1d, sensor_1d)
    sensor_locs = np.stack([sensor_xx.ravel(), sensor_yy.ravel()], axis=1)

    A_single = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ])
    A = np.kron(np.eye(num_targets), A_single)

    Gamma_single = np.array([
        [1/3, 0,   0.5, 0  ],
        [0,   1/3, 0,   0.5],
        [0.5, 0,   1,   0  ],
        [0,   0.5, 0,   1  ],
    ])
    Q_real_single = 0.05 * Gamma_single
    Q_real = np.kron(np.eye(num_targets), Q_real_single)
    Q_real = 0.5 * (Q_real + Q_real.T) + eps * np.eye(nx)
    
 
    Q_filter_single = np.array([
        [3.0,  0.0,  0.1,  0.0 ],
        [0.0,  3.0,  0.0,  0.1 ],
        [0.1,  0.0,  0.03, 0.0 ],
        [0.0,  0.1,  0.0,  0.03],
    ])
    Q_filter = np.kron(np.eye(num_targets), Q_filter_single)
    Q_filter = 0.5 * (Q_filter + Q_filter.T) + eps * np.eye(nx)
    

    if use_paper_Q:
        Q = Q_filter
    else:
        Q = Q_real
    
    if m0_positions is None:
        m0_positions = np.array([
            [12, 6],
            [32, 32],
            [20, 13],
            [15, 35],
        ][:num_targets])
    if m0_velocities is None:
        m0_velocities = np.array([
            [0.001, 0.001],
            [-0.001, -0.005],
            [-0.1, 0.01],
            [0.002, 0.002],
        ][:num_targets])
    
    m0_vec = np.zeros(nx)
    for i in range(num_targets):
        m0_vec[4*i] = m0_positions[i, 0]
        m0_vec[4*i + 1] = m0_positions[i, 1]
        m0_vec[4*i + 2] = m0_velocities[i, 0]
        m0_vec[4*i + 3] = m0_velocities[i, 1]

    sigma0 = np.tile([10.0, 10.0, 1.0, 1.0], num_targets)
    
    if deterministic_init:
        P0_diag_single = np.array([1e-12, 1e-12, 1e-12, 1e-12])
    else:
        P0_diag_single = np.diag(Q_filter_single)
    P0_diag = np.tile(P0_diag_single, num_targets)
    P0 = np.diag(P0_diag)
    P0 = 0.5 * (P0 + P0.T) + eps * np.eye(nx)
    

    R = (obs_noise_std ** 2) * np.eye(ny)
    R_chol = np.linalg.cholesky(R + eps * np.eye(ny))
    
    def _h_acoustic(x: np.ndarray) -> np.ndarray:
        """
        Acoustic observation function.
        
        Args:
            x: [N, nx] particles or [nx] single state
            
        Returns:
            y: [N, ny] or [ny] observations
        """
        single = x.ndim == 1
        if single:
            x = x[None, :]
        
        N = x.shape[0]
        

        positions = np.zeros((N, num_targets, 2))
        for i in range(num_targets):
            positions[:, i, 0] = x[:, 4*i]
            positions[:, i, 1] = x[:, 4*i + 1]
        
        diff = positions[:, :, None, :] - sensor_locs[None, None, :, :] 
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))  
        
   
        contributions = source_amplitude / (dist + dist_offset) 
        y_mean = np.sum(contributions, axis=1)
        
        if single:
            return y_mean[0]
        return y_mean
    
    def _H_jac_acoustic(x: np.ndarray) -> np.ndarray:
        """
        Jacobian of acoustic observation function.
        
        
        Args:
            x: [nx] vector
            
        Returns:
            H: [ny, nx] Jacobian
        """
        positions = np.zeros((num_targets, 2))
        for i in range(num_targets):
            positions[i, 0] = x[4*i]
            positions[i, 1] = x[4*i + 1]
        
        diff = positions[:, None, :] - sensor_locs[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + eps)
        
        denom = dist * ((dist + dist_offset) ** 2) 
        grad_coeff = -source_amplitude / denom  
        grad_pos = grad_coeff[:, :, None] * diff 
        
        H = np.zeros((ny, nx))
        
        for i in range(num_targets):
            for j in range(num_sensors):
                H[j, 4*i] = grad_pos[i, j, 0]
                H[j, 4*i + 1] = grad_pos[i, j, 1]
        
        return H
    
    ssm = NumpySSM(
        A=A, Q=Q, m0=m0_vec, P0=P0,
        obs_fn=_h_acoustic,
        R=R,
    )
    
    ssm.h_fn = _h_acoustic
    ssm.H_jac_fn = _H_jac_acoustic
    
    ssm.Q_filter = Q_filter  # Larger Q for filtering
    ssm.Q_real = Q_real      # Smaller Q for simulation
    ssm.sigma0 = sigma0      
    ssm.sensor_locs = sensor_locs 
    ssm.source_amplitude = source_amplitude
    ssm.dist_offset = dist_offset
    
    return ssm


def simulate_acoustic_with_Q_real(
    ssm: NumpySSM,
    T: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate acoustic model using Q_real (smaller noise for realistic trajectories).
    
    The SSM uses Q_filter for filtering, but simulation should use Q_real.
    
    Args:
        ssm: Acoustic SSM (must have Q_real attribute)
        T: Number of time steps
        seed: Random seed
        
    Returns:
        xs: [T+1, nx] True states (x[0] is initial)
        ys: [T, ny] Observations
    """
    if not hasattr(ssm, 'Q_real'):
        raise ValueError("SSM must have Q_real attribute (use make_acoustic_ssm_np)")
    
    rng = np.random.default_rng(seed)
    nx = ssm.nx
    ny = ssm.ny
    
    # Simulate states with Q_real
    xs = [ssm.m0.copy()]
    x = ssm.m0.copy()
    for t in range(T):
        x = ssm.A @ x + rng.multivariate_normal(np.zeros(nx), ssm.Q_real)
        xs.append(x)
    xs = np.array(xs)
    
    # Generate observations
    ys = []
    for t in range(1, T + 1):
        y_mean = ssm.h_fn(xs[t:t+1])[0]
        y = y_mean + rng.multivariate_normal(np.zeros(ny), ssm.R)
        ys.append(y)
    ys = np.array(ys)
    
    return xs, ys


def make_ssm_np(name: str, **kwargs) -> NumpySSM:
    """
    Create SSM by name.
    
    Args:
        name: One of "random_walk", "linear_gaussian", "range_bearing", "acoustic"
        **kwargs: Model-specific parameters
        
    Returns:
        NumpySSM instance
    """
    factories = {
        "random_walk": make_random_walk_ssm_np,
        "linear_gaussian": make_linear_gaussian_ssm_np,
        "range_bearing": make_range_bearing_ssm_np,
        "acoustic": make_acoustic_ssm_np,
    }
    
    if name not in factories:
        raise ValueError(f"Unknown model: {name}. Choose from {list(factories.keys())}")
    
    return factories[name](**kwargs)
