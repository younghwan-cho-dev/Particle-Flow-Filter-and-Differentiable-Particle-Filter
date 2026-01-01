import tensorflow as tf
import numpy as np


#deprecated
def systematic_resample(weights, rng):
    """
    Systematic resampling.
    """
    N = tf.shape(weights)[0]
    N_float = tf.cast(N, weights.dtype)
    
    u0 = rng.uniform([], 0.0, 1.0 / N_float, dtype=weights.dtype)
    u = u0 + tf.cast(tf.range(N), weights.dtype) / N_float
    
    cdf = tf.cumsum(weights)
    
    idx = tf.searchsorted(cdf, u, side="left")
    idx = tf.minimum(idx, N - 1)
    return tf.cast(idx, tf.int32)

#deprecated
def multinomial_resample(weights, rng, N=None):
    """
    Multinomial resampling.
    """
    if N is None:
        N = tf.shape(weights)[0]
    
    logw = tf.math.log(weights + 1e-12)
    idx = tf.random.categorical(
        tf.expand_dims(logw, axis=0), N, dtype=tf.int32
    )[0]
    return idx


def compute_particle_mean_cov(particles, weights=None, eps=1e-6):
    """
    Compute weighted mean and covariance from particles.
    
    Args:
        particles: [N, nx] particle states
        weights: [N] normalized weights (uniform if None)
        eps: regularization for covariance
        
    Returns:
        mean: [nx]
        cov: [nx, nx]
    """
    N = tf.shape(particles)[0]
    nx = particles.shape[-1]
    dtype = particles.dtype
    
    if weights is None:
        weights = tf.ones([N], dtype=dtype) / tf.cast(N, dtype)
    
    # Weighted mean
    mean = tf.reduce_sum(weights[:, None] * particles, axis=0)
    
    # Weighted covariance
    centered = particles - mean[None, :]
    cov = tf.einsum('n,ni,nj->ij', weights, centered, centered)
    
    # Symmetrize and regularize
    cov = 0.5 * (cov + tf.transpose(cov)) + eps * tf.eye(nx, dtype=dtype)
    
    return mean, cov


def compute_ess(weights):
    """
    Compute Effective Sample Size.
    
    Args:
        weights: [N] normalized weights
        
    Returns:
        ess: scalar, effective sample size
    """
    return 1.0 / tf.reduce_sum(tf.square(weights))


def compute_rmse(true_states, estimated_states, indices=None):
    """
    Compute Root Mean Square Error.
    
    Args:
        true_states: [T, nx] true states
        estimated_states: [T, nx] estimated states
        indices: optional list of state indices to use
        
    Returns:
        rmse: scalar
    """
    if indices is not None:
        true_states = tf.gather(true_states, indices, axis=-1)
        estimated_states = tf.gather(estimated_states, indices, axis=-1)
    
    mse = tf.reduce_mean(tf.square(true_states - estimated_states))
    return tf.sqrt(mse)


def compute_nees(true_states, estimated_means, estimated_covs, eps=1e-6):
    """
    Compute Normalized Estimation Error Squared (NEES).
    
    Args:
        true_states: [T, nx] true states
        estimated_means: [T, nx] estimated means
        estimated_covs: [T, nx, nx] estimated covariances
        
    Returns:
        nees: [T] NEES values
    """
    errors = true_states - estimated_means  # [T, nx]
    
    nees_list = []
    for t in range(int(errors.shape[0])):
        e_t = errors[t]  # [nx]
        P_t = estimated_covs[t]  # [nx, nx]
        
        # Regularize and invert
        nx = P_t.shape[0]
        P_t = 0.5 * (P_t + tf.transpose(P_t)) + eps * tf.eye(nx, dtype=P_t.dtype)
        L = tf.linalg.cholesky(P_t)
        P_inv_e = tf.linalg.cholesky_solve(L, e_t[:, None])[:, 0]
        
        nees_t = tf.reduce_sum(e_t * P_inv_e)
        nees_list.append(nees_t)
    
    return tf.stack(nees_list)


def plot_tracking_results(
    xs_true, 
    xs_est, 
    num_targets=4,
    figsize=(10, 10),
    title="Tracking Results"
):
    """
    Args:
        xs_true: [T, nx] true states
        xs_est: [T, nx] estimated states
        num_targets: number of targets
        figsize: figure size
        title: plot title
        
    Returns:
        fig, ax: matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['tab:red', 'tab:green', 'tab:cyan', 'tab:purple']
    
    for i in range(num_targets):
        px_idx = 4 * i
        py_idx = 4 * i + 1
        
        # True trajectory
        ax.plot(
            xs_true[:, px_idx], xs_true[:, py_idx],
            color=colors[i % len(colors)], linewidth=1.5,
            label=f'Target {i} (true)'
        )
        ax.scatter(
            xs_true[:, px_idx], xs_true[:, py_idx],
            color=colors[i % len(colors)], s=10, alpha=0.5
        )
        
        # Estimated trajectory
        ax.plot(
            xs_est[:, px_idx], xs_est[:, py_idx],
            color=colors[i % len(colors)], linewidth=1.5, linestyle='--',
            label=f'Target {i} (est)'
        )
        
        # Mark start and end
        ax.scatter(
            xs_true[0, px_idx], xs_true[0, py_idx],
            color=colors[i % len(colors)], s=100, marker='o', edgecolors='black'
        )
        ax.scatter(
            xs_true[-1, px_idx], xs_true[-1, py_idx],
            color=colors[i % len(colors)], s=100, marker='X', edgecolors='black'
        )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig, ax


def plot_ess_history(ess_values, threshold=None, figsize=(10, 4)):
    """
    Plot ESS history.
    
    Args:
        ess_values: [T] ESS values
        threshold: optional threshold line
        figsize: figure size
        
    Returns:
        fig, ax
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(ess_values, 'b-', linewidth=1.5, label='ESS')
    
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('ESS')
    ax.set_title('Effective Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax