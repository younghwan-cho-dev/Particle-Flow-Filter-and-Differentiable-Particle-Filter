"""
Linear Gaussian State Space Model.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from ..prob_ssm import ProbSSM

tfd = tfp.distributions


def make_lgssm(A, B, C, D, m0, P0, eps=1e-6, dtype=tf.float32):
    """
    Create a Linear Gaussian State Space Model.
    
    Args:
        A: [nx, nx] state transition matrix
        B: [nx, nv] process noise matrix (Q = B @ B^T)
        C: [ny, nx] observation matrix
        D: [ny, nw] observation noise matrix (R = D @ D^T)
        m0: [nx] initial mean
        P0: [nx, nx] initial covariance
        eps: numerical stability constant
        dtype: TensorFlow dtype
        
    Returns:
        ProbSSM instance
    """
    A = tf.convert_to_tensor(A, dtype)
    B = tf.convert_to_tensor(B, dtype)
    C = tf.convert_to_tensor(C, dtype)
    D = tf.convert_to_tensor(D, dtype)
    m0 = tf.convert_to_tensor(m0, dtype)
    P0 = tf.convert_to_tensor(P0, dtype)

    nx = A.shape[-1]
    ny = C.shape[-2]

    Q = B @ tf.transpose(B)
    R = D @ tf.transpose(D)

    P0_chol = tf.linalg.cholesky(P0 + eps * tf.eye(nx, dtype=dtype))
    Q_chol = tf.linalg.cholesky(Q + eps * tf.eye(nx, dtype=dtype))
    R_chol = tf.linalg.cholesky(R + eps * tf.eye(ny, dtype=dtype))

    def init_dist():
        return tfd.MultivariateNormalTriL(loc=m0, scale_tril=P0_chol)

    def transition_dist(x_prev, u, t):
        return tfd.MultivariateNormalTriL(
            loc=tf.linalg.matvec(A, x_prev), 
            scale_tril=Q_chol
        )

    def observation_dist(x_t, t):
        return tfd.MultivariateNormalTriL(
            loc=tf.linalg.matvec(C, x_t), 
            scale_tril=R_chol
        )
    
    # EKF/UKF hooks (linear case)
    def f_mean(x, u, t):
        return tf.linalg.matvec(A, x)
    
    def F_jac(x, u, t):
        return A
    
    def h_mean(x, t):
        return tf.linalg.matvec(C, x)
    
    def H_jac(x, t):
        return C

    return ProbSSM(
        init_dist=init_dist,
        transition_dist=transition_dist,
        observation_dist=observation_dist,
        f_mean=f_mean,
        F_jac=F_jac,
        h_mean=h_mean,
        H_jac=H_jac,
        state_dim=int(nx),
        obs_dim=int(ny),
        dtype=dtype,
    )