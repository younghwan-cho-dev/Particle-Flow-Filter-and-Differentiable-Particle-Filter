"""
Range-Bearing State Space Model with Student-t measurement noise.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..prob_ssm import ProbSSM

tfd = tfp.distributions

PI = tf.constant(3.141592653589793, tf.float32)


def wrap_angle(a):
    return (a + PI) % (2.0 * PI) - PI


def h_range_bearing(x, eps=1e-12):
    """
    Observation function: x -> (range, bearing).
    
    Args:
        x: [4] state vector [px, py, vx, vy]
        
    Returns:
        y: [2] observation [range, bearing]
    """
    px, py = x[0], x[1]
    r = tf.sqrt(px*px + py*py + eps)
    th = tf.atan2(py, px)
    return tf.stack([r, th], axis=0)


def H_jac_range_bearing(x, eps=1e-12):
    """
    Jacobian for range-bearing observation.
    
    Args:
        x: [4] state vector
        
    Returns:
        H: [2, 4] Jacobian matrix
    """
    px, py = x[0], x[1]
    r2 = px*px + py*py + eps
    r = tf.sqrt(r2)

    dr_dpx = px / r
    dr_dpy = py / r
    dth_dpx = -py / r2
    dth_dpy = px / r2

    row0 = tf.stack([dr_dpx, dr_dpy, 0.0, 0.0], axis=0)
    row1 = tf.stack([dth_dpx, dth_dpy, 0.0, 0.0], axis=0)
    return tf.stack([row0, row1], axis=0)


def make_rb_studentt_ssm(
    dt=0.01,
    q_diag=0.01,
    nu=2.0,
    s_r=0.05,
    s_th=0.05,
    m0=(1.0, 0.5, 0.01, 0.01),
    P0_diag=(0.1, 0.1, 0.1, 0.1),
    dtype=tf.float32,
    eps=1e-6,
):
    """
    Range-Bearing SSM with Student-t measurement noise.
    
    Args:
        dt: Time step
        q_diag: Process noise diagonal value
        nu: Degrees of freedom for Student-t
        s_r: Scale for range noise
        s_th: Scale for bearing noise
        m0: Initial mean
        P0_diag: Initial covariance diagonal
        dtype: TensorFlow dtype
        eps: Numerical stability constant
        
    Returns:
        ProbSSM instance
    """
    dt = tf.convert_to_tensor(dt, dtype)
    A = tf.constant(np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32), dtype=dtype)

    nx = 4

    Q = tf.linalg.diag(tf.fill([nx], tf.cast(q_diag, dtype)))
    Q = 0.5 * (Q + tf.transpose(Q)) + eps * tf.eye(nx, dtype=dtype)

    m0 = tf.convert_to_tensor(m0, dtype)
    P0 = tf.linalg.diag(tf.convert_to_tensor(P0_diag, dtype))
    P0 = 0.5 * (P0 + tf.transpose(P0)) + eps * tf.eye(nx, dtype=dtype)

    def init_dist():
        L0 = tf.linalg.cholesky(P0)
        return tfd.MultivariateNormalTriL(loc=m0, scale_tril=L0)

    def transition_dist(x_prev, u, t):
        mean = tf.linalg.matvec(A, x_prev)
        Lq = tf.linalg.cholesky(Q)
        return tfd.MultivariateNormalTriL(loc=mean, scale_tril=Lq)

    class RangeBearingStudentT(tfd.Distribution):
        def __init__(self, x, df, s_r, s_th, validate_args=False, 
                     name="RangeBearingStudentT"):
            parameters = dict(locals())
            super().__init__(
                dtype=dtype,
                reparameterization_type=tfd.NOT_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=True,
                parameters=parameters,
                name=name,
            )
            self._x = x
            self._df = tf.convert_to_tensor(df, dtype)
            self._s_r = tf.convert_to_tensor(s_r, dtype)
            self._s_th = tf.convert_to_tensor(s_th, dtype)

            self._dist_r = tfd.StudentT(
                df=self._df, loc=tf.zeros([], dtype), scale=self._s_r
            )
            self._dist_th = tfd.StudentT(
                df=self._df, loc=tf.zeros([], dtype), scale=self._s_th
            )

        def _event_shape(self):
            return tf.TensorShape([2])

        def _batch_shape(self):
            return tf.TensorShape([])

        def _sample_n(self, n, seed=None):
            y0 = h_range_bearing(self._x)
            nr = self._dist_r.sample(sample_shape=[n], seed=seed)
            nth = self._dist_th.sample(sample_shape=[n], seed=seed)

            r = y0[0] + nr
            th = wrap_angle(y0[1] + nth)
            return tf.stack([r, th], axis=-1)

        def _log_prob(self, y):
            y = tf.convert_to_tensor(y, dtype)
            y0 = h_range_bearing(self._x)
            vr = y[..., 0] - y0[0]
            vth = wrap_angle(y[..., 1] - y0[1])
            return self._dist_r.log_prob(vr) + self._dist_th.log_prob(vth)

    def observation_dist(x_t, t):
        return RangeBearingStudentT(x=x_t, df=nu, s_r=s_r, s_th=s_th)

    # --- EKF/UKF hooks ---
    def f_mean(x, u, t):
        return tf.linalg.matvec(A, x)

    def F_jac(x, u, t):
        return A

    def h_mean(x, t):
        return h_range_bearing(x)

    def H_jac(x, t):
        return H_jac_range_bearing(x)

    return ProbSSM(
        init_dist=init_dist,
        transition_dist=transition_dist,
        observation_dist=observation_dist,
        f_mean=f_mean,
        F_jac=F_jac,
        h_mean=h_mean,
        H_jac=H_jac,
        state_dim=4,
        obs_dim=2,
        dtype=dtype,
    )