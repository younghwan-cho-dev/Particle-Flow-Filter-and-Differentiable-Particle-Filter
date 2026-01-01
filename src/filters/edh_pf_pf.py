"""
EDH PF-PF (Particle Flow Particle Filter with EDH flow).
"""

import tensorflow as tf
from typing import Optional

from ..utils import systematic_resample, compute_particle_mean_cov
from ..flows.edh_flow import (
    integrate_edh_flow,
    compute_A_matrix_conditioning,
)

def _sample_transition_batch(ssm, x_prev, u_t, t, rng, dtype):
    try:
        dist = ssm.transition_dist(x_prev, u_t, t) 
        seed32 = tf.cast(rng.make_seeds(2)[0], tf.int32)   # shape [2], int32
        eta0 = tf.cast(dist.sample(seed=seed32), dtype)
        logp_eta0 = tf.cast(dist.log_prob(eta0), dtype)
        return eta0, logp_eta0
    except Exception:
        N = tf.shape(x_prev)[0]
        eta_ta = tf.TensorArray(dtype, size=N)
        lp_ta = tf.TensorArray(dtype, size=N)
        for i in tf.range(N):
            di = ssm.transition_dist(x_prev[i], u_t, t)
            seed32 = tf.cast(rng.make_seeds(2)[0], tf.int32)
            xi = tf.cast(di.sample(seed=seed32), dtype)
            eta_ta = eta_ta.write(i, xi)
            lp_ta = lp_ta.write(i, tf.cast(di.log_prob(xi), dtype))
        return eta_ta.stack(), lp_ta.stack()


def _logprob_transition_batch(ssm, x_prev, x_new, u_t, t, dtype):
    try:
        dist = ssm.transition_dist(x_prev, u_t, t)
        return tf.cast(dist.log_prob(x_new), dtype)
    except Exception:
        N = tf.shape(x_prev)[0]
        lp_ta = tf.TensorArray(dtype, size=N)
        for i in tf.range(N):
            di = ssm.transition_dist(x_prev[i], u_t, t)
            lp_ta = lp_ta.write(i, tf.cast(di.log_prob(x_new[i]), dtype))
        return lp_ta.stack()


def _loglik_obs_batch(ssm, x, y, t, dtype):
    """
    Compute log p(y | x^i) in batch if possible, else per-particle.
    """
    try:
        dist = ssm.observation_dist(x, t)
        return tf.cast(dist.log_prob(y), dtype)
    except Exception:
        N = tf.shape(x)[0]
        lp_ta = tf.TensorArray(dtype, size=N)
        for i in tf.range(N):
            di = ssm.observation_dist(x[i], t)
            lp_ta = lp_ta.write(i, tf.cast(di.log_prob(y), dtype))
        return lp_ta.stack()


def edh_pf_pf(
    ssm,
    ys,
    u_seq=None,
    num_particles: int = 500,
    num_flow_steps: int = 29,
    resample: str = "ess",
    ess_threshold: float = 0.5,
    resampling_method: str = "systematic",
    return_particles: bool = False,
    seed: Optional[int] = None,
    dtype=None,
):
    if dtype is None:
        dtype = getattr(ssm, "dtype", tf.float32)

    ys = tf.convert_to_tensor(ys, dtype)
    T = int(ys.shape[0])
    N = int(num_particles)

    rng = (
        tf.random.Generator.from_seed(seed)
        if seed is not None
        else tf.random.Generator.from_non_deterministic_state()
    )

    seed32 = tf.cast(rng.make_seeds(2)[0], tf.int32)
    x = tf.cast(ssm.init_dist().sample([N], seed=seed32), dtype) 
    nx = int(x.shape[-1])

    logw = -tf.math.log(tf.cast(N, dtype)) * tf.ones([N], dtype=dtype)

    m0, P0 = compute_particle_mean_cov(x, None)
    ms = [m0]
    Ps = [P0]

    # Diagnostics
    ess_hist = []
    maxw_hist = []
    resampled_hist = []
    unique_anc_hist = []
    logZ_hist = []
    flow_mag_hist = []
    cond_number_hist = []
    loglik_cum = tf.constant(0.0, dtype=dtype)

    if return_particles:
        x_hist = [x]
        w_hist = [tf.exp(logw)]
    else:
        x_hist = None
        w_hist = None

    ny = int(ys.shape[-1])

    for t in range(1, T + 1):
        y_t = ys[t - 1]
        u_t = None if u_seq is None else tf.convert_to_tensor(u_seq[t - 1], dtype)

        x_prev = x
        w_prev = tf.exp(logw)  # normalized

        eta0, logp_eta0 = _sample_transition_batch(ssm, x_prev, u_t, t, rng, dtype)  # [N,nx], [N]

        m_pred, P_pred = compute_particle_mean_cov(eta0, w_prev)

        H = ssm.H_jac(m_pred, t)     # [ny,nx]
        h_m = ssm.h_mean(m_pred, t)  # [ny]

        obs_dist = ssm.observation_dist(m_pred, t)
        if hasattr(obs_dist, "scale_tril"):
            R_chol = obs_dist.scale_tril
            R = R_chol @ tf.transpose(R_chol)
        elif hasattr(obs_dist, "covariance") and callable(obs_dist.covariance):
            R = obs_dist.covariance()
        elif hasattr(obs_dist, "covariance_matrix"):
            R = obs_dist.covariance_matrix
        else:
            R = tf.eye(ny, dtype=dtype) * tf.cast(0.01, dtype)

        x_flow, flow_diag = integrate_edh_flow(
            x0_batch=eta0,
            P=P_pred,
            H=H,
            R=R,
            m=m_pred,
            y=y_t,
            h_m=h_m,
            num_steps=num_flow_steps,
        )

        # flow diagnostics
        if "flow_magnitudes" in flow_diag:
            flow_mag_hist.append(tf.reduce_mean(flow_diag["flow_magnitudes"]))
        else:
            flow_mag_hist.append(tf.reduce_mean(tf.norm(x_flow - eta0, axis=-1)))

        cond_diag = compute_A_matrix_conditioning(0.5, P_pred, H, R)
        cond_number_hist.append(cond_diag["condition_number"])

        loglik = _loglik_obs_batch(ssm, x_flow, y_t, t, dtype)                
        logp_xflow = _logprob_transition_batch(ssm, x_prev, x_flow, u_t, t, dtype)  

        log_inc = logp_xflow + loglik - logp_eta0

        logZt = tf.reduce_logsumexp(logw + log_inc)
        logZ_hist.append(logZt)
        loglik_cum += logZt

        logw = logw + log_inc
        logw = logw - tf.reduce_logsumexp(logw)
        w = tf.exp(logw)

        ess = 1.0 / tf.reduce_sum(tf.square(w))
        ess_hist.append(ess)
        maxw_hist.append(tf.reduce_max(w))

        m_filt, P_filt = compute_particle_mean_cov(x_flow, w)
        ms.append(m_filt)
        Ps.append(P_filt)

        if resample == "always":
            do_resample = True
        elif resample == "ess":
            do_resample = bool((ess < tf.cast(ess_threshold * N, dtype)).numpy())
        elif resample == "none":
            do_resample = False
        else:
            raise ValueError(f"Unknown resample policy: {resample}")

        if do_resample:
            if resampling_method == "systematic":
                idx = systematic_resample(w, rng) 
            elif resampling_method == "multinomial":
                idx = tf.random.categorical(tf.expand_dims(logw, axis=0), N, dtype=tf.int32)[0]
            else:
                raise ValueError(f"Unknown resampling_method: {resampling_method}")

            x = tf.gather(x_flow, idx, axis=0)
            logw = -tf.math.log(tf.cast(N, dtype)) * tf.ones([N], dtype=dtype)

            unique_anc = tf.shape(tf.unique(idx).y)[0]
            unique_anc_hist.append(tf.cast(unique_anc, dtype))
            resampled_hist.append(tf.constant(True))
        else:
            x = x_flow
            unique_anc_hist.append(tf.cast(N, dtype))
            resampled_hist.append(tf.constant(False))

        if return_particles:
            x_hist.append(x)
            w_hist.append(tf.exp(logw))

    ms = tf.stack(ms, axis=0)
    Ps = tf.stack(Ps, axis=0)

    info = {
        "ess": tf.stack(ess_hist, axis=0),
        "max_weight": tf.stack(maxw_hist, axis=0),
        "resampled": tf.stack(resampled_hist, axis=0),
        "unique_ancestors": tf.stack(unique_anc_hist, axis=0),
        "logZ": tf.stack(logZ_hist, axis=0),
        "loglik_cum": loglik_cum,
        "flow_magnitude": tf.stack(flow_mag_hist, axis=0),
        "condition_number": tf.stack(cond_number_hist, axis=0),
    }
    if return_particles:
        info["particles"] = tf.stack(x_hist, axis=0)
        info["weights"] = tf.stack(w_hist, axis=0)

    return ms, Ps, info
