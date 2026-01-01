import tensorflow as tf
from typing import Optional

from ..utils import systematic_resample


def particle_filter(
    ssm,
    ys,
    u_seq=None,
    num_particles=1000,
    resample="ess",
    ess_threshold=0.5,
    resampling_method="systematic",
    return_particles=False,
    seed=None,
    dtype=None,
):
    """
    Bootstrap/SIR particle filter for a ProbSSM.

    Model:
        x_0 ~ init_dist()
        x_t ~ transition_dist(x_{t-1}, u_t, t)
        y_t ~ observation_dist(x_t, t)

    Args:
        ssm: ProbSSM instance
        ys: [T, ny] observations (y1..yT)
        u_seq: optional [T, nu] inputs aligned with ys
        num_particles: number of particles N
        resample: "none" | "ess" | "always"
        ess_threshold: threshold for ESS resampling (fraction of N)
        resampling_method: "systematic" | "multinomial"
        return_particles: whether to return particle/weight history
        seed: random seed
        dtype: TensorFlow dtype
        
    Returns:
        ms: [T+1, nx] filtered means
        Ps: [T+1, nx, nx] filtered covariances
        info: dict with diagnostics
    """
    if dtype is None:
        dtype = getattr(ssm, "dtype", tf.float32)

    ys = tf.convert_to_tensor(ys, dtype)
    T = int(ys.shape[0])
    N = int(num_particles)

    rng = tf.random.Generator.from_seed(seed) if seed is not None \
        else tf.random.Generator.from_non_deterministic_state()

    # Initialize particles
    x = tf.cast(ssm.init_dist().sample([N], seed=seed), dtype)
    nx = int(x.shape[-1])

    logw = -tf.math.log(tf.cast(N, dtype)) * tf.ones([N], dtype=dtype)

    # Initial mean/cov
    w0 = tf.ones([N], dtype=dtype) / tf.cast(N, dtype)
    m0 = tf.reduce_sum(w0[:, None] * x, axis=0)
    xc0 = x - m0[None, :]
    P0 = tf.einsum("n,ni,nj->ij", w0, xc0, xc0)
    P0 = 0.5 * (P0 + tf.transpose(P0))

    ms = [m0]
    Ps = [P0]

    # Diagnostics
    ess_hist = []
    maxw_hist = []
    ent_hist = []
    resampled_hist = []
    unique_anc_hist = []
    logZ_hist = []
    loglik_cum = tf.constant(0.0, dtype=dtype)

    if return_particles:
        x_hist = [x]
        w_hist = [tf.exp(logw)]
    else:
        x_hist = None
        w_hist = None

    for t in range(1, T + 1):
        y_t = ys[t - 1]
        u_t = None if u_seq is None else tf.convert_to_tensor(u_seq[t - 1], dtype)

        # Get transition distribution parameters at each particle
        # tf.map_fn too slow
        means = tf.map_fn(
            lambda x_i: ssm.transition_dist(x_i, u_t, t).mean(),
            x,
            fn_output_signature=dtype
        )  # [N, nx]
        
        dist_ref = ssm.transition_dist(x[0], u_t, t)
        if hasattr(dist_ref, 'scale_tril'):
            scale_tril = dist_ref.scale_tril  # [nx, nx]
            # Sample standard normal and transform
            z = tf.random.normal([N, nx], dtype=dtype)  # [N, nx]
            x = means + tf.linalg.matvec(scale_tril, z)  # [N, nx]
        else:
            x = tf.stack([
                tf.cast(ssm.transition_dist(x[i], u_t, t).sample(), dtype)
                for i in range(N)
            ], axis=0)

        loglik_i = tf.vectorized_map(
            lambda x_ti: tf.cast(ssm.observation_dist(x_ti, t).log_prob(y_t), dtype),
            x
        )

        logZt = tf.reduce_logsumexp(logw + loglik_i)
        logZ_hist.append(logZt)
        loglik_cum += logZt

        # Update weights
        logw = logw + loglik_i
        logw = logw - tf.reduce_logsumexp(logw)
        w = tf.exp(logw)

        ess = 1.0 / tf.reduce_sum(tf.square(w))
        ess_hist.append(ess)
        maxw_hist.append(tf.reduce_max(w))
        ent_hist.append(-tf.reduce_sum(w * tf.math.log(w + tf.constant(1e-12, dtype=dtype))))

        m_filt = tf.reduce_sum(w[:, None] * x, axis=0)
        xc = x - m_filt[None, :]
        P_filt = tf.einsum("n,ni,nj->ij", w, xc, xc)
        P_filt = 0.5 * (P_filt + tf.transpose(P_filt))

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
                idx = tf.random.categorical(
                    tf.expand_dims(logw, axis=0), N, dtype=tf.int32
                )[0]
            else:
                raise ValueError(f"Unknown resampling_method: {resampling_method}")

            x = tf.gather(x, idx, axis=0)
            logw = -tf.math.log(tf.cast(N, dtype)) * tf.ones([N], dtype=dtype)

            unique_anc = tf.shape(tf.unique(idx).y)[0]
            unique_anc_hist.append(tf.cast(unique_anc, dtype))
            resampled_hist.append(tf.constant(True))
        else:
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
        "entropy": tf.stack(ent_hist, axis=0),
        "resampled": tf.stack(resampled_hist, axis=0),
        "unique_ancestors": tf.stack(unique_anc_hist, axis=0),
        "logZ": tf.stack(logZ_hist, axis=0),
        "loglik_cum": loglik_cum,
    }
    if return_particles:
        info["particles"] = tf.stack(x_hist, axis=0)
        info["weights"] = tf.stack(w_hist, axis=0)

    return ms, Ps, info