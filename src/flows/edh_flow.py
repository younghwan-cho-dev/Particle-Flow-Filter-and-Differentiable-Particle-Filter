
# EDH (Exact Daum-Huang) Flow Implementation.

import tensorflow as tf
import numpy as np

def compute_edh_flow_params(lam, P, H, R, m, y, h_m, eps=1e-6):
    """
    Compute EDH flow parameters.

    Shapes:
        P: [nx,nx], H: [ny,nx], R: [ny,ny], m:[nx], y:[ny], h_m:[ny]
    """
    dtype = P.dtype
    nx = tf.shape(P)[0]
    ny = tf.shape(R)[0]
    I_nx = tf.eye(nx, dtype=dtype)

    lam = tf.cast(lam, dtype)
    eps = tf.cast(eps, dtype)

    HPHt = H @ P @ tf.transpose(H)  # [ny,ny]

    S_lam = lam * HPHt + R
    S_lam = 0.5 * (S_lam + tf.transpose(S_lam)) + eps * tf.eye(ny, dtype=dtype)

    L_S = tf.linalg.cholesky(S_lam)

    X = tf.linalg.cholesky_solve(L_S, H)  # [ny,nx]
    A_lam = -0.5 * (P @ tf.transpose(H) @ X)  # [nx,nx]

    Hm = tf.linalg.matvec(H, m)      # [ny]
    e = h_m - Hm                     # [ny]
    y_minus_e = y - e                # [ny]

    R_stable = 0.5 * (R + tf.transpose(R)) + eps * tf.eye(ny, dtype=dtype)
    L_R = tf.linalg.cholesky(R_stable)
    v = tf.linalg.cholesky_solve(L_R, y_minus_e[:, None])  # [ny,1
    v = tf.squeeze(v, axis=-1)                             # [ny]
    PHt = P @ tf.transpose(H)                              # [nx,ny]
    PHt_Rinv_y = tf.linalg.matvec(PHt, v)                  # [nx]

    A_m = tf.linalg.matvec(A_lam, m)                       # [nx]

    I_plus_lamA = I_nx + lam * A_lam
    I_plus_2lamA = I_nx + 2.0 * lam * A_lam

    inner = tf.linalg.matvec(I_plus_lamA, PHt_Rinv_y) + A_m
    b_lam = tf.linalg.matvec(I_plus_2lamA, inner)

    return A_lam, b_lam


def edh_flow_step_batch(x_batch, lam, P, H, R, m, y, h_m, eps=1e-6):
    A_lam, b_lam = compute_edh_flow_params(lam, P, H, R, m, y, h_m, eps)

    # Correct batch affine application:
    # x: [N,nx], A: [nx,nx]  ->  x Aᵀ: [N,nx]
    dx_dlam_batch = tf.linalg.matmul(x_batch, A_lam, transpose_b=True) + b_lam[None, :]
    return dx_dlam_batch


def generate_lambda_schedule(num_steps=29, dtype=tf.float32):
    """
    Exponentially spaced λ schedule from 0 to 1:
    """
    k = tf.range(num_steps + 1, dtype=dtype)
    N = tf.cast(num_steps, dtype)
    e4 = tf.exp(tf.cast(4.0, dtype))
    lambdas = (tf.exp(tf.cast(4.0, dtype) * k / N) - 1.0) / (e4 - 1.0)
    return lambdas


def integrate_edh_flow(x0_batch, P, H, R, m, y, h_m, num_steps=29, eps=1e-6):
    """
    Integrate EDH flow using Euler steps on an exponential \lambda grid.

    Returns:
        x1_batch: [N,nx]
        diagnostics: dict including flow magnitudes and (optional) logdet increments
    """
    dtype = x0_batch.dtype
    lambdas = generate_lambda_schedule(num_steps, dtype=dtype)  # [num_steps+1]

    x_batch = tf.identity(x0_batch)

    flow_magnitudes = []
    logdet_increments = []

    for k in range(num_steps):
        lam_k = lambdas[k]
        lam_k1 = lambdas[k + 1]
        d_lam = lam_k1 - lam_k

        A_lam, b_lam = compute_edh_flow_params(lam_k, P, H, R, m, y, h_m, eps)

        dx_dlam = tf.linalg.matmul(x_batch, A_lam, transpose_b=True) + b_lam[None, :]
        x_batch = x_batch + d_lam * dx_dlam

        flow_magnitudes.append(tf.reduce_mean(tf.norm(dx_dlam, axis=-1)))

        nx = tf.shape(P)[0]
        I = tf.eye(nx, dtype=dtype)
        M = I + d_lam * A_lam
        sign, logabsdet = tf.linalg.slogdet(M)

        logdet_increments.append(logabsdet)

    diagnostics = {
        "flow_magnitudes": tf.stack(flow_magnitudes),
        "lambda_schedule": lambdas,
        "logdet_increments": tf.stack(logdet_increments),
    }
    return x_batch, diagnostics



def compute_edh_flow_jacobian_determinant(lam_start, lam_end, P, H, R, eps=1e-6):
    """
    Args:
        lam_start, lam_end: integration bounds
        P, H, R: flow parameters
        
    Returns:
        log_det_jacobian: scalar
    """
    ny = R.shape[0]
    dtype = P.dtype
    
    # Single step approximation at midpoint
    lam_mid = 0.5 * (lam_start + lam_end)
    d_lam = lam_end - lam_start
    
    # Compute A at midpoint
    HPHt = H @ P @ tf.transpose(H)
    S_lam = lam_mid * HPHt + R
    S_lam = 0.5 * (S_lam + tf.transpose(S_lam)) + eps * tf.eye(ny, dtype=dtype)
    L_S = tf.linalg.cholesky(S_lam)
    S_lam_inv = tf.linalg.cholesky_solve(L_S, tf.eye(ny, dtype=dtype))
    
    A_lam = -0.5 * P @ tf.transpose(H) @ S_lam_inv @ H
    
    trace_A = tf.linalg.trace(A_lam)
    log_det_jacobian = d_lam * trace_A
    
    return log_det_jacobian


def compute_A_matrix_conditioning(lam, P, H, R, eps=1e-6):
    """
    Compute conditioning diagnostics for A(lambda)) matrix.
    
    Args:
        lam: pseudo-time parameter
        P, H, R: flow parameters
        
    Returns:
        dict with eigenvalues, condition number, spectral radius
    """
    ny = R.shape[0]
    dtype = P.dtype
    
    HPHt = H @ P @ tf.transpose(H)
    S_lam = lam * HPHt + R
    S_lam = 0.5 * (S_lam + tf.transpose(S_lam)) + eps * tf.eye(ny, dtype=dtype)
    L_S = tf.linalg.cholesky(S_lam)
    S_lam_inv = tf.linalg.cholesky_solve(L_S, tf.eye(ny, dtype=dtype))
    
    A_lam = -0.5 * P @ tf.transpose(H) @ S_lam_inv @ H
    
    # Eigenvalue analysis
    eigenvalues = tf.linalg.eigvalsh(A_lam)
    
    # Condition number
    abs_eigs = tf.abs(eigenvalues)
    max_eig = tf.reduce_max(abs_eigs)
    min_eig = tf.reduce_min(abs_eigs) + eps
    cond_number = max_eig / min_eig
    
    # Spectral radius
    spectral_radius = max_eig
    
    return {
        'eigenvalues': eigenvalues,
        'condition_number': cond_number,
        'spectral_radius': spectral_radius,
        'trace': tf.linalg.trace(A_lam),
    }