"""
Kalman Filter for Linear Gaussian State Space Models.
"""

import tensorflow as tf


def kalman_filter(ys, A, B, C, D, m0, P0, eps=1e-6, dtype=tf.float32):
    """
    Kalman Filter for Linear Gaussian SSM.

    Args:
        ys: [T, ny] observations
        A: [nx, nx] state transition matrix
        B: [nx, nv] process noise matrix (Q = B @ B^T)
        C: [ny, nx] observation matrix
        D: [ny, nw] observation noise matrix (R = D @ D^T)
        m0: [nx] initial mean
        P0: [nx, nx] initial covariance
        eps: numerical stability constant
        dtype: TensorFlow dtype
        
    Returns:
        ms: [T+1, nx] filtered means (m0 plus filtered means)
        Ps: [T+1, nx, nx] filtered covariances
    """
    ys = tf.convert_to_tensor(ys, dtype)
    A = tf.convert_to_tensor(A, dtype)
    B = tf.convert_to_tensor(B, dtype)
    C = tf.convert_to_tensor(C, dtype)
    D = tf.convert_to_tensor(D, dtype)
    m0 = tf.convert_to_tensor(m0, dtype)
    P0 = tf.convert_to_tensor(P0, dtype)

    T = tf.shape(ys)[0]
    nx = A.shape[-1]
    ny = C.shape[-2]

    Q = B @ tf.transpose(B)
    R = D @ tf.transpose(D)
    I = tf.eye(nx, dtype=dtype)

    ms = [m0]
    Ps = [P0]

    for t in range(int(ys.shape[0])):
        m_prev, P_prev = ms[-1], Ps[-1]

        # Predict
        m_pred = tf.linalg.matvec(A, m_prev)
        P_pred = A @ P_prev @ tf.transpose(A) + Q

        # Update
        y_pred = tf.linalg.matvec(C, m_pred)
        v = ys[t] - y_pred
        S = C @ P_pred @ tf.transpose(C) + R

        # Ensure symmetry & positive definiteness
        S = 0.5 * (S + tf.transpose(S)) + eps * tf.eye(ny, dtype=dtype)
        L = tf.linalg.cholesky(S)

        # Compute Kalman gain via Cholesky solve
        PCt = P_pred @ tf.transpose(C)
        U = tf.linalg.triangular_solve(L, tf.transpose(PCt), lower=True)
        V = tf.linalg.triangular_solve(L, U, lower=True, adjoint=True)
        K = tf.transpose(V)

        # Update mean
        m_filt = m_pred + tf.linalg.matvec(K, v)

        # Joseph form covariance update (numerically stable)
        KC = K @ C
        P_filt = (I - KC) @ P_pred @ tf.transpose(I - KC) + K @ R @ tf.transpose(K)
        P_filt = 0.5 * (P_filt + tf.transpose(P_filt))

        ms.append(m_filt)
        Ps.append(P_filt)

    ms = tf.stack(ms, axis=0)
    Ps = tf.stack(Ps, axis=0)
    return ms, Ps