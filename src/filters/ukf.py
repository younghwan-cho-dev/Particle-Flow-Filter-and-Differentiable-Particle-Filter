import tensorflow as tf

def _sigma_points(m, P, alpha=1e-3, beta=2.0, kappa=0.0, eps=1e-6, dtype=tf.float32):
    """
    Returns:
      X:  [2n+1, n] sigma points
      Wm: [2n+1]    mean weights
      Wc: [2n+1]    covariance weights
    """
    m = tf.convert_to_tensor(m, dtype)
    P = tf.convert_to_tensor(P, dtype)
    n = int(m.shape[0])

    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    gamma = tf.sqrt(tf.cast(c, dtype))

    P = 0.5 * (P + tf.transpose(P)) + eps * tf.eye(n, dtype=dtype)
    L = tf.linalg.cholesky(P)  # [n,n]

    X0 = m[None, :]                                # [1,n]
    cols = tf.transpose(L)                         # [n,n] rows are columns of L
    Xi_plus  = m[None, :] + gamma * cols           # [n,n]
    Xi_minus = m[None, :] - gamma * cols           # [n,n]
    X = tf.concat([X0, Xi_plus, Xi_minus], axis=0) # [2n+1, n]

    Wm0 = lam / c
    Wc0 = lam / c + (1.0 - alpha**2 + beta)
    Wi  = 1.0 / (2.0 * c)

    Wm = tf.concat([tf.constant([Wm0], dtype),
                    tf.fill([2*n], tf.constant(Wi, dtype))], axis=0)
    Wc = tf.concat([tf.constant([Wc0], dtype), 
                    tf.fill([2*n], tf.constant(Wi, dtype))], axis=0)
    return X, Wm, Wc

def ukf_filter(
    ys,
    f, h,             # callables: f(x,u,t)->[nx], h(x,t)->[ny]
    Q, R,             # [nx,nx], [ny,ny] (or callables Q(t), R(t))
    m0, P0,
    u_seq=None,
    alpha=1e-3, beta=2.0, kappa=0.0,
    eps=1e-6,
    dtype=tf.float32
):
    """
    Unscented Kalman Filter (UKF), additive noise.

    Returns:
      ms: [T+1, nx]
      Ps: [T+1, nx, nx]
    """
    ys = tf.convert_to_tensor(ys, dtype)
    m0 = tf.convert_to_tensor(m0, dtype)
    P0 = tf.convert_to_tensor(P0, dtype)

    T  = int(ys.shape[0])
    ny = int(ys.shape[1])
    nx = int(m0.shape[0])

    def get_Q(t):
        Qt = Q(t) if callable(Q) else Q
        Qt = tf.convert_to_tensor(Qt, dtype)
        Qt = tf.ensure_shape(Qt, [nx, nx])
        return Qt

    def get_R(t):
        Rt = R(t) if callable(R) else R
        Rt = tf.convert_to_tensor(Rt, dtype)
        Rt = tf.ensure_shape(Rt, [ny, ny])
        return Rt

    ms = [m0]
    Ps = [P0]

    for t in range(T):
        m_prev, P_prev = ms[-1], Ps[-1]
        u_t = None if u_seq is None else tf.convert_to_tensor(u_seq[t], dtype)
        Qt = get_Q(t)
        Rt = get_R(t)

        X, Wm, Wc = _sigma_points(m_prev, P_prev, alpha, beta, kappa, eps, dtype=dtype)  # [2n+1,nx]
        # propagate sigma points
        X_prop = tf.stack(
            [tf.convert_to_tensor(f(X[i], u_t, t), dtype) for i in range(2 * nx + 1)],
            axis=0
        )  # [2n+1,nx]
        m_pred = tf.reduce_sum(Wm[:, None] * X_prop, axis=0)  # [nx]

        dX = X_prop - m_pred[None, :]                         # [2n+1,nx]
        P_pred = tf.matmul(tf.transpose(dX) * Wc[None, :], dX) + Qt
        P_pred = 0.5 * (P_pred + tf.transpose(P_pred)) + eps * tf.eye(nx, dtype=dtype)

        X2, Wm2, Wc2 = _sigma_points(m_pred, P_pred, alpha, beta, kappa, eps, dtype=dtype)  # [2n+1,nx]

        Y = tf.stack(
            [tf.convert_to_tensor(h(X2[i], t), dtype) for i in range(2 * nx + 1)],
            axis=0
        )  # [2n+1,ny]

        y_pred = tf.reduce_sum(Wm2[:, None] * Y, axis=0)  # [ny]
        dY = Y - y_pred[None, :]                          # [2n+1,ny]

        S = tf.matmul(tf.transpose(dY) * Wc2[None, :], dY) + Rt
        S = 0.5 * (S + tf.transpose(S)) + eps * tf.eye(ny, dtype=dtype)

        dX2 = X2 - m_pred[None, :]                        # [2n+1,nx]

        Pxy = tf.matmul(tf.transpose(dX2) * Wc2[None, :], dY)  # [nx,ny]

        L = tf.linalg.cholesky(S)
        Z = tf.linalg.cholesky_solve(L, tf.transpose(Pxy))  # [ny,nx] = S^{-1} Pxy^T
        K = tf.transpose(Z)                                 # [nx,ny]

        v = ys[t] - y_pred
        m_filt = m_pred + tf.linalg.matvec(K, v)

        P_filt = P_pred - K @ S @ tf.transpose(K)
        P_filt = 0.5 * (P_filt + tf.transpose(P_filt)) + eps * tf.eye(nx, dtype=dtype)

        ms.append(m_filt)
        Ps.append(P_filt)

    return tf.stack(ms, axis=0), tf.stack(Ps, axis=0)