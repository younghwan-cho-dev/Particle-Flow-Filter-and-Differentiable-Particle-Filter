"""
Probabilistic State Space Model (ProbSSM) base class.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass
from typing import Callable, Optional, Any

tfd = tfp.distributions

@dataclass
class ProbSSM:
    """
    Probabilistic State Space Model.

    Attributes:
        init_dist: Callable returning initial state distribution
        transition_dist: Callable(x_prev, u, t) returning transition distribution
        observation_dist: Callable(x_t, t) returning observation distribution
        f_mean: Optional mean function for transition (for EKF/UKF)
        F_jac: Optional Jacobian of f w.r.t. x (for EKF)
        h_mean: Optional mean function for observation (for EKF/UKF)
        H_jac: Optional Jacobian of h w.r.t. x (for EKF)
        state_dim: State dimension (optional metadata)
        obs_dim: Observation dimension (optional metadata)
        dtype: TensorFlow dtype
    """
    # Required: Probabilistic State Space Model
    init_dist: Callable[[], tfd.Distribution]
    transition_dist: Callable[[tf.Tensor, Optional[tf.Tensor], int], tfd.Distribution]
    observation_dist: Callable[[tf.Tensor, int], tfd.Distribution]

    # Optional: for EKF/UKF and flow filters
    f_mean: Optional[Callable[[tf.Tensor, Optional[tf.Tensor], int], tf.Tensor]] = None
    F_jac: Optional[Callable[[tf.Tensor, Optional[tf.Tensor], int], tf.Tensor]] = None
    h_mean: Optional[Callable[[tf.Tensor, int], tf.Tensor]] = None
    H_jac: Optional[Callable[[tf.Tensor, int], tf.Tensor]] = None

    # Model metadata
    state_dim: Optional[int] = None
    obs_dim: Optional[int] = None
    
    dtype: Any = tf.float32

    def simulate(self, T: int, u_seq: Optional[tf.Tensor] = None, seed: int = 0):
        """
        Simulate from the SSM.
        
        Args:
            T: Number of time steps
            u_seq: Optional [T, nu] control inputs
            seed: Random seed
            
        Returns:
            xs: [T+1, nx] states (x0..xT)
            ys: [T, ny] observations (y1..yT), where y_t is generated from x_t
        """
        seed_stream = tfp.util.SeedStream(seed, salt="ProbSSM.simulate")

        x0 = tf.cast(self.init_dist().sample(seed=seed_stream()), self.dtype)
        xs_ta = tf.TensorArray(self.dtype, size=T+1, clear_after_read=False)
        ys_ta = tf.TensorArray(self.dtype, size=T, clear_after_read=False)

        xs_ta = xs_ta.write(0, x0)
        x_prev = x0

        for t in range(1, T+1):
            u_t = None if u_seq is None else tf.convert_to_tensor(u_seq[t-1], self.dtype)

            x_dist = self.transition_dist(x_prev, u_t, t)
            x_t = tf.cast(x_dist.sample(seed=seed_stream()), self.dtype)

            y_dist = self.observation_dist(x_t, t)
            y_t = tf.cast(y_dist.sample(seed=seed_stream()), self.dtype)

            xs_ta = xs_ta.write(t, x_t)
            ys_ta = ys_ta.write(t-1, y_t)

            x_prev = x_t

        xs = xs_ta.stack()
        ys = ys_ta.stack()
        return xs, ys