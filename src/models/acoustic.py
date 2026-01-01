"""
Multi-target Acoustic Tracking State Space Model.

State: num_targets × [x, y, vx, vy]
Observations: num_sensors amplitude measurements
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional

from ..prob_ssm import ProbSSM

tfd = tfp.distributions


def make_acoustic_tracking_ssm(
    num_targets: int = 4,
    num_sensors_per_side: int = 5,
    sensor_range: tuple = (0.0, 40.0),
    dt: float = 1.0,
    source_amplitude: float = 10.0,  # Amp in paper
    obs_noise_std: float = 0.1,      # measurement noise std
    m0_positions: Optional[np.ndarray] = None,
    m0_velocities: Optional[np.ndarray] = None,
    dist_offset: float = 0.1,  # d0 
    deterministic_init: bool = False,  # True for simulation, False for filtering
    eps: float = 1e-6,
    dtype=tf.float32,
):
    """
    Multi-target Acoustic Tracking SSM from Li & Coates (2017).
    
    State x = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...] for num_targets targets
    State dimension: num_targets * 4
    
    Args:
        num_targets: Number of acoustic sources
        num_sensors_per_side: Sensors arranged on grid (total = num_sensors_per_side^2)
        sensor_range: (min, max) range for sensor grid
        dt: Time step
        source_amplitude: Amplitude s_i (same for all sources), Amp=10 in paper
        obs_noise_std: Observation noise std (measvar_real=0.01, so std=0.1)
        m0_positions: Initial target positions [num_targets, 2]
        m0_velocities: Initial target velocities [num_targets, 2]
        dist_offset: d0 in paper
        deterministic_init: If True, use deterministic init (for simulation).
                           If False, use uncertain Gaussian prior (for filtering).
        
    Returns:
        ProbSSM instance
    """
    nx = num_targets * 4  # [x, y, vx, vy] per target
    num_sensors = num_sensors_per_side ** 2
    ny = num_sensors
    
    sensor_1d = np.linspace(sensor_range[0], sensor_range[1], num_sensors_per_side)
    sensor_xx, sensor_yy = np.meshgrid(sensor_1d, sensor_1d)
    sensor_locs = np.stack([sensor_xx.ravel(), sensor_yy.ravel()], axis=1)
    sensor_locs = tf.constant(sensor_locs, dtype=dtype)  # [num_sensors, 2]
    
    # Single target block: [x, y, vx, vy]
    A_single = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    A = np.kron(np.eye(num_targets), A_single)  # Block diagonal
    A = tf.constant(A, dtype=dtype)
    
    # Process noise covariance Q
    Q_single = (1.0 / 20.0) * np.array([
        [1/3, 0,   1/2, 0  ],
        [0,   1/3, 0,   1/2],
        [1/2, 0,   1,   0  ],
        [0,   1/2, 0,   1  ]
    ], dtype=np.float32)
    Q = np.kron(np.eye(num_targets), Q_single)  # Block diagonal
    Q = tf.constant(Q, dtype=dtype)
    Q_chol = tf.linalg.cholesky(Q + eps * tf.eye(nx, dtype=dtype))
    
    # Initial state
    if m0_positions is None:
        m0_positions = np.array([
            [12, 6],
            [32, 32],
            [20, 13],
            [15, 35]
        ][:num_targets], dtype=np.float32)
    if m0_velocities is None:
        m0_velocities = np.array([
            [1e-3, 1e-3],
            [-1e-3, -5e-3],
            [-0.1, 0.01],
            [2e-3, 2e-3]
        ][:num_targets], dtype=np.float32)
    
    # Interleave
    m0_list = []
    for i in range(num_targets):
        m0_list.extend([m0_positions[i, 0], m0_positions[i, 1],
                        m0_velocities[i, 0], m0_velocities[i, 1]])
    m0 = tf.constant(m0_list, dtype=dtype)
    
    # Initial covariance P0
    # Paper: sigma0 = 10 * [1, 1, 0.1, 0.1] 
    # So [100, 100, 1, 1]
    if deterministic_init:
        P0_diag_single = np.array([1e-12, 1e-12, 1e-12, 1e-12], dtype=np.float32)
    else:
        P0_diag_single = np.array([100.0, 100.0, 1.0, 1.0], dtype=np.float32)
    P0_diag_full = np.tile(P0_diag_single, num_targets)
    P0 = tf.linalg.diag(tf.constant(P0_diag_full, dtype=dtype))
    P0_chol = tf.linalg.cholesky(P0 + eps * tf.eye(nx, dtype=dtype))
    
    # Observation noise
    R = (obs_noise_std ** 2) * tf.eye(ny, dtype=dtype)
    R_chol = tf.linalg.cholesky(R + eps * tf.eye(ny, dtype=dtype))
    
    s_amp = tf.constant(source_amplitude, dtype=dtype)
    d0 = tf.constant(dist_offset, dtype=dtype)
    
    
    def init_dist():
        return tfd.MultivariateNormalTriL(loc=m0, scale_tril=P0_chol)
    
    def transition_dist(x_prev, u, t):
        mean = tf.linalg.matvec(A, x_prev)
        return tfd.MultivariateNormalTriL(loc=mean, scale_tril=Q_chol)
    
    def h_mean_fn(x, t=None):
        """
        Observation function: amplitude at each sensor.
        
        x: [nx] state vector
        Returns: [ny] observations
        """

        positions = []
        for i in range(num_targets):
            px = x[4*i]
            py = x[4*i + 1]
            positions.append(tf.stack([px, py]))
        positions = tf.stack(positions, axis=0)  # [num_targets, 2]
        
        diff = positions[:, None, :] - sensor_locs[None, :, :]
        dist = tf.sqrt(tf.reduce_sum(diff ** 2, axis=-1))  # [num_targets, num_sensors]
        
        contributions = s_amp / (dist + d0)  # [num_targets, num_sensors]
        y_mean = tf.reduce_sum(contributions, axis=0)  # [num_sensors]
        
        return y_mean
    
    def observation_dist(x_t, t):
        y_mean = h_mean_fn(x_t, t)
        return tfd.MultivariateNormalTriL(loc=y_mean, scale_tril=R_chol)
    
    def H_jac_fn(x, t=None):
        """
        Jacobian of observation function w.r.t. state.
        
        Returns: [ny, nx] Jacobian matrix
        """
        # Extract positions
        positions = []
        for i in range(num_targets):
            px = x[4*i]
            py = x[4*i + 1]
            positions.append(tf.stack([px, py]))
        positions = tf.stack(positions, axis=0)  # [num_targets, 2]
        
        diff = positions[:, None, :] - sensor_locs[None, :, :]  # [num_targets, num_sensors, 2]
        dist = tf.sqrt(tf.reduce_sum(diff ** 2, axis=-1))  # [num_targets, num_sensors]

        denom = dist * ((dist + d0) ** 2) + eps
        grad_coeff = -s_amp / denom  # [num_targets, num_sensors]
        grad_pos = grad_coeff[:, :, None] * diff  # [num_targets, num_sensors, 2]
        
        H = tf.zeros([ny, nx], dtype=dtype)
        
        indices = []
        values = []
        for i in range(num_targets):
            for j in range(num_sensors):
                indices.append([j, 4*i])
                values.append(grad_pos[i, j, 0])
                indices.append([j, 4*i + 1])
                values.append(grad_pos[i, j, 1])
        
        indices = tf.constant(indices, dtype=tf.int32)
        values = tf.stack(values)
        H = tf.tensor_scatter_nd_update(H, indices, values)
        
        return H
    
    def f_mean_fn(x, u, t):
        return tf.linalg.matvec(A, x)
    
    def F_jac_fn(x, u, t):
        return A
    
    return ProbSSM(
        init_dist=init_dist,
        transition_dist=transition_dist,
        observation_dist=observation_dist,
        f_mean=f_mean_fn,
        F_jac=F_jac_fn,
        h_mean=h_mean_fn,
        H_jac=H_jac_fn,
        state_dim=nx,
        obs_dim=ny,
        dtype=dtype,
    )