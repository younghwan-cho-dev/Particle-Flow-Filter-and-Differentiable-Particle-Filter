"""
Extended Kalman Filter (EKF) for nonlinear state space models.
"""

import tensorflow as tf


def ekf_filter(
    ys,
    f, F_jac,
    h, H_jac,
    Q, R,
    m0, P0,
    u_seq=None,
    eps=1e-6,
    dtype=tf.float32
):
    """
    Extended Kalman Filter (EKF) with additive noise.

    Args:
        ys: [T, ny] observations
        f: callable f(x, u, t) -> [nx] dynamics mean function
        F_jac: callable F(x, u, t) -> [nx, nx] Jacobian of f
        h: callable h(x, t) -> [ny] observation mean function
        H_jac: callable H(x, t) -> [ny, nx] Jacobian of h
        Q: [nx, nx] process noise covariance (or callable Q(t))
        R: [ny, ny] observation noise covariance (or callable R(t))
        m0: [nx] initial mean
        P0: [nx, nx] initial covariance
        u_seq: optional [T, nu] control inputs
        eps: numerical stability constant
        dtype: TensorFlow dtype

    Returns:
        ms: [T+1, nx] filtered means
        Ps: [T+1, nx, nx] filtered covariances
    """
    ys = tf.convert_to_tensor(ys, dtype)
    m0 = tf.convert_to_tensor(m0, dtype)
    P0 = tf.convert_to_tensor(P0, dtype)

    T = int(ys.shape[0])
    ny = int(ys.shape[1])
    nx = int(m0.shape[0])

    I = tf.eye(nx, dtype=dtype)

    def get_Q(t):
        return Q(t) if callable(Q) else tf.convert_to_tensor(Q, dtype)

    def get_R(t):
        return R(t) if callable(R) else tf.convert_to_tensor(R, dtype)

    ms = [m0]
    Ps = [P0]

    for t in range(T):
        m_prev, P_prev = ms[t], Ps[t]
        u_t = None if u_seq is None else tf.convert_to_tensor(u_seq[t], dtype)

        m_pred = tf.convert_to_tensor(f(m_prev, u_t, t), dtype)
        F = tf.convert_to_tensor(F_jac(m_prev, u_t, t), dtype)
        Qt = get_Q(t)
        P_pred = F @ P_prev @ tf.transpose(F) + Qt
        P_pred = 0.5 * (P_pred + tf.transpose(P_pred))

        y_pred = tf.convert_to_tensor(h(m_pred, t), dtype)
        H = tf.convert_to_tensor(H_jac(m_pred, t), dtype)
        Rt = get_R(t)

        v = ys[t] - y_pred
        S = H @ P_pred @ tf.transpose(H) + Rt

        # SPD guard
        S = 0.5 * (S + tf.transpose(S)) + eps * tf.eye(ny, dtype=dtype)
        L = tf.linalg.cholesky(S)

        # K = P_pred H^T S^{-1} via Cholesky solves
        PHt = P_pred @ tf.transpose(H)
        U = tf.linalg.triangular_solve(L, tf.transpose(PHt), lower=True)
        V = tf.linalg.triangular_solve(L, U, lower=True, adjoint=True)
        K = tf.transpose(V)

        m_filt = m_pred + tf.linalg.matvec(K, v)

        # Joseph covariance update
        KH = K @ H
        P_filt = (I - KH) @ P_pred @ tf.transpose(I - KH) + K @ Rt @ tf.transpose(K)
        P_filt = 0.5 * (P_filt + tf.transpose(P_filt))

        ms.append(m_filt)
        Ps.append(P_filt)

    return tf.stack(ms, axis=0), tf.stack(Ps, axis=0)