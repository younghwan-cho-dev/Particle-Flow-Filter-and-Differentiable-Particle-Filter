import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from src.filters import kalman_filter, ekf_filter, ukf_filter
from src import ProbSSM
from src_numpy.numpy_pf import particle_filter_numpy
from src_numpy.numpy_ssm import NumpySSM


def assert_close(name: str, a, b, atol: float, rtol: float):
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    max_abs = tf.reduce_max(tf.abs(a - b)).numpy()
    max_rel = tf.reduce_max(tf.abs(a - b) / (tf.abs(b) + 1e-12)).numpy()
    
    passed = np.allclose(a.numpy(), b.numpy(), atol=atol, rtol=rtol)
    if not passed:
        pytest.fail(
            f"{name}: FAILED\n"
            f"  max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}\n"
            f"  required: atol={atol:.0e}, rtol={rtol:.0e}"
        )


def assert_symmetric(P: np.ndarray, name: str, atol: float = 1e-10):
    """Check that matrix P is symmetric."""
    diff = np.max(np.abs(P - P.T))
    if diff > atol:
        pytest.fail(f"{name}: Not symmetric, max|P - P.T| = {diff:.3e}")


def assert_positive_definite(P: np.ndarray, name: str, tol: float = 1e-10):
    """Check that matrix P is positive definite via eigenvalues."""
    eigvals = np.linalg.eigvalsh(P)
    min_eig = np.min(eigvals)
    if min_eig < tol:
        pytest.fail(f"{name}: Not positive definite, min eigenvalue = {min_eig:.3e}")


def assert_covariance_valid(Ps: np.ndarray, name: str, atol: float = 1e-10):
    """Check symmetry and positive-definiteness for a sequence of covariances.
    
    Args:
        Ps: [T, nx, nx] array of covariance matrices
        name: identifier for error messages
        atol: tolerance for symmetry check
    """
    T = Ps.shape[0]
    for t in range(T):
        assert_symmetric(Ps[t], f"{name}[t={t}]", atol=atol)
        assert_positive_definite(Ps[t], f"{name}[t={t}]")


def check_convergence(Ps: np.ndarray, tol: float = 1e-6, window: int = 10) -> bool:
    """Check if covariance sequence has converged in the last `window` steps.
    
    Args:
        Ps: [T, nx, nx] array of covariance matrices
        tol: convergence tolerance
        window: number of final steps to check
        
    Returns:
        True if converged, False otherwise
    """
    T = Ps.shape[0]
    if T < window + 1:
        return False
    
    for t in range(T - window, T):
        diff = np.max(np.abs(Ps[t] - Ps[t-1]))
        if diff > tol:
            return False
    return True


def assert_convergence(Ps: np.ndarray, name: str, tol: float = 1e-6, window: int = 10):
    T = Ps.shape[0]
    if T < window + 1:
        pytest.fail(f"{name}: Not enough timesteps ({T}) for convergence check (need {window+1})")
    
    max_diffs = []
    for t in range(T - window, T):
        diff = np.max(np.abs(Ps[t] - Ps[t-1]))
        max_diffs.append(diff)
        
    if max(max_diffs) > tol:
        pytest.fail(
            f"{name}: Covariance did not converge.\n"
            f"  Last {window} diffs: {[f'{d:.3e}' for d in max_diffs]}\n"
            f"  Required tol: {tol:.0e}"
        )


def make_stable_linear_params(nx: int, ny: int, seed: int = 0, dtype=tf.float64):
    rng = np.random.default_rng(seed)
    
    # Make a stable A (spectral radius < 1)
    A = rng.normal(size=(nx, nx)) * 0.2
    A = A / max(1.2, np.max(np.abs(np.linalg.eigvals(A))))
    
    C = rng.normal(size=(ny, nx))
    Q = np.diag(rng.uniform(0.05, 0.2, size=nx))
    R = np.diag(rng.uniform(0.05, 0.2, size=ny))
    m0 = rng.normal(size=nx)
    P0 = np.eye(nx)
    
    # Cholesky factors for KF interface (BB^T = Q, DD^T = R)
    B = np.linalg.cholesky(Q)
    D = np.linalg.cholesky(R)
    
    return {
        # NumPy versions
        "A_np": A,
        "C_np": C,
        "Q_np": Q,
        "R_np": R,
        "B_np": B,
        "D_np": D,
        "m0_np": m0,
        "P0_np": P0,
        # TensorFlow versions
        "A": tf.convert_to_tensor(A, dtype=dtype),
        "C": tf.convert_to_tensor(C, dtype=dtype),
        "Q": tf.convert_to_tensor(Q, dtype=dtype),
        "R": tf.convert_to_tensor(R, dtype=dtype),
        "B": tf.convert_to_tensor(B, dtype=dtype),
        "D": tf.convert_to_tensor(D, dtype=dtype),
        "m0": tf.convert_to_tensor(m0, dtype=dtype),
        "P0": tf.convert_to_tensor(P0, dtype=dtype),
        # Metadata
        "nx": nx,
        "ny": ny,
        "dtype": dtype,
    }


def simulate_linear_ssm_np(T: int, A, Q, C, R, m0, P0, seed: int = 0):
    """Simulate from linear Gaussian SSM (NumPy).
    
    Returns:
        xs: [T+1, nx] states (x_0, ..., x_T)
        ys: [T, ny] observations (y_1, ..., y_T)
    """
    rng = np.random.default_rng(seed)
    nx = A.shape[0]
    ny = C.shape[0]
    
    Q_chol = np.linalg.cholesky(Q)
    R_chol = np.linalg.cholesky(R)
    P0_chol = np.linalg.cholesky(P0)
    
    xs = np.zeros((T + 1, nx))
    ys = np.zeros((T, ny))
    
    xs[0] = m0 + P0_chol @ rng.normal(size=nx)
    
    for t in range(T):
        xs[t + 1] = A @ xs[t] + Q_chol @ rng.normal(size=nx)
        ys[t] = C @ xs[t + 1] + R_chol @ rng.normal(size=ny)
    
    return xs, ys

@pytest.fixture
def dtype():
    return tf.float64


@pytest.fixture
def random_walk_1d_tf_params(dtype):
    """1D random walk parameters for TensorFlow filters. This is for KF/EKF/UKF"""
    Qval, Rval = 0.2, 0.5
    
    A = tf.constant([[1.0]], dtype=dtype)
    C = tf.constant([[1.0]], dtype=dtype)
    Q = tf.constant([[Qval]], dtype=dtype)
    R = tf.constant([[Rval]], dtype=dtype)
    B = tf.constant([[np.sqrt(Qval)]], dtype=dtype)
    D = tf.constant([[np.sqrt(Rval)]], dtype=dtype)
    m0 = tf.constant([0.3], dtype=dtype)
    P0 = tf.constant([[1.2]], dtype=dtype)
    
    return {
        "A": A, "C": C, "Q": Q, "R": R, "B": B, "D": D,
        "m0": m0, "P0": P0,
        "Qval": Qval, "Rval": Rval,
        "nx": 1, "ny": 1, "dtype": dtype,
    }


@pytest.fixture
def random_walk_1d_np_params():
    """1D random walk parameters for NumPy PF."""
    Qval, Rval = 0.2, 0.5
    
    return {
        "A": np.array([[1.0]]),
        "C": np.array([[1.0]]),
        "Q": np.array([[Qval]]),
        "R": np.array([[Rval]]),
        "m0": np.array([0.3]),
        "P0": np.array([[1.2]]),
        "Qval": Qval, "Rval": Rval,
        "nx": 1, "ny": 1,
    }


@pytest.fixture
def stable_2d_tf_params(dtype):
    """Randomly generated stable 2D system for TensorFlow filters."""
    return make_stable_linear_params(nx=2, ny=2, seed=42, dtype=dtype)


@pytest.fixture
def stable_2d_np_params():
    """Randomly generated stable 2D system for NumPy PF."""
    params = make_stable_linear_params(nx=2, ny=2, seed=42, dtype=tf.float64)
    return {
        "A": params["A_np"],
        "C": params["C_np"],
        "Q": params["Q_np"],
        "R": params["R_np"],
        "m0": params["m0_np"],
        "P0": params["P0_np"],
        "nx": 2, "ny": 2,
    }


@pytest.fixture
def linear_multidim_tf_params(dtype):
    """Higher-dimensional linear system for TensorFlow filters."""
    return make_stable_linear_params(nx=4, ny=2, seed=0, dtype=dtype)


@pytest.fixture
def linear_multidim_np_params():
    """Higher-dimensional linear system for NumPy PF."""
    params = make_stable_linear_params(nx=4, ny=2, seed=0, dtype=tf.float64)
    return {
        "A": params["A_np"],
        "C": params["C_np"],
        "Q": params["Q_np"],
        "R": params["R_np"],
        "m0": params["m0_np"],
        "P0": params["P0_np"],
        "nx": 4, "ny": 2,
    }


def make_prob_ssm(params, dtype=tf.float64):
    """Create ProbSSM from parameters dict."""
    A, C = params["A"], params["C"]
    Q, R = params["Q"], params["R"]
    m0, P0 = params["m0"], params["P0"]
    
    def init_dist():
        P0_chol = tf.linalg.cholesky(P0)
        return tfd.MultivariateNormalTriL(loc=m0, scale_tril=P0_chol)
    
    def transition_dist(x_prev, u, t):
        mean = tf.linalg.matvec(A, x_prev)
        Q_chol = tf.linalg.cholesky(Q)
        return tfd.MultivariateNormalTriL(loc=mean, scale_tril=Q_chol)
    
    def observation_dist(x_t, t):
        mean = tf.linalg.matvec(C, x_t)
        R_chol = tf.linalg.cholesky(R)
        return tfd.MultivariateNormalTriL(loc=mean, scale_tril=R_chol)
    
    return ProbSSM(
        init_dist=init_dist,
        transition_dist=transition_dist,
        observation_dist=observation_dist,
        dtype=dtype
    )


def make_numpy_ssm(params):
    return NumpySSM(
        A=params["A"],
        C=params["C"],
        Q=params["Q"],
        R=params["R"],
        m0=params["m0"],
        P0=params["P0"],
    )


# KF Validity Tests
class TestKFValidity:
    
    T = 100  # timesteps for convergence tests
    CONV_TOL = 1e-6
    CONV_WINDOW = 10
    
    def test_covariance_convergence_1d_random_walk(self, random_walk_1d_tf_params):
        """KF covariance should converge to steady-state for 1D random walk."""
        params = random_walk_1d_tf_params
        dtype = params["dtype"]
        
        # Simulate observations
        xs_np, ys_np = simulate_linear_ssm_np(
            T=self.T,
            A=params["A"].numpy(),
            Q=params["Q"].numpy(),
            C=params["C"].numpy(),
            R=params["R"].numpy(),
            m0=params["m0"].numpy(),
            P0=params["P0"].numpy(),
            seed=123
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        # Run KF
        ms_kf, Ps_kf = kalman_filter(
            ys, params["A"], params["B"], params["C"], params["D"],
            params["m0"], params["P0"], eps=0.0, dtype=dtype
        )
        Ps_np = Ps_kf.numpy()
        
        assert_convergence(Ps_np, "KF P (1D RW)", tol=self.CONV_TOL, window=self.CONV_WINDOW)
    
    def test_covariance_convergence_2d(self, stable_2d_tf_params):
        """KF covariance should converge for stable 2D system."""
        params = stable_2d_tf_params
        dtype = params["dtype"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=self.T,
            A=params["A_np"],
            Q=params["Q_np"],
            C=params["C_np"],
            R=params["R_np"],
            m0=params["m0_np"],
            P0=params["P0_np"],
            seed=456
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        ms_kf, Ps_kf = kalman_filter(
            ys, params["A"], params["B"], params["C"], params["D"],
            params["m0"], params["P0"], eps=0.0, dtype=dtype
        )
        Ps_np = Ps_kf.numpy()
        assert_convergence(Ps_np, "KF P (2D)", tol=self.CONV_TOL, window=self.CONV_WINDOW)
    
    def test_covariance_symmetry_positive_definite_1d(self, random_walk_1d_tf_params):
        """KF covariances should remain symmetric and positive definite."""
        params = random_walk_1d_tf_params
        dtype = params["dtype"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=50,
            A=params["A"].numpy(),
            Q=params["Q"].numpy(),
            C=params["C"].numpy(),
            R=params["R"].numpy(),
            m0=params["m0"].numpy(),
            P0=params["P0"].numpy(),
            seed=789
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        ms_kf, Ps_kf = kalman_filter(
            ys, params["A"], params["B"], params["C"], params["D"],
            params["m0"], params["P0"], eps=0.0, dtype=dtype
        )
        assert_covariance_valid(Ps_kf.numpy(), "KF P (1D)")
        
    
    def test_covariance_symmetry_positive_definite_multidim(self, linear_multidim_tf_params):
        """KF covariances should remain symmetric and PD for multi-dim system."""
        params = linear_multidim_tf_params
        dtype = params["dtype"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=50,
            A=params["A_np"],
            Q=params["Q_np"],
            C=params["C_np"],
            R=params["R_np"],
            m0=params["m0_np"],
            P0=params["P0_np"],
            seed=101
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        ms_kf, Ps_kf = kalman_filter(
            ys, params["A"], params["B"], params["C"], params["D"],
            params["m0"], params["P0"], eps=0.0, dtype=dtype
        )
        assert_covariance_valid(Ps_kf.numpy(), "KF P (multidim)")
        

# EKF Tests

class TestEKF:
    
    def test_ekf_equals_kf_1d_random_walk(self, random_walk_1d_tf_params):
        """EKF should equal KF exactly for linear 1D system."""
        params = random_walk_1d_tf_params
        dtype = params["dtype"]
        A, C = params["A"], params["C"]
        Q, R = params["Q"], params["R"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=100,
            A=A.numpy(), Q=Q.numpy(), C=C.numpy(), R=R.numpy(),
            m0=params["m0"].numpy(), P0=params["P0"].numpy(),
            seed=111
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        # EKF functions for linear system
        def f(x, u, t):
            return tf.linalg.matvec(A, x)
        
        def F_jac(x, u, t):
            return A
        
        def h(x, t):
            return tf.linalg.matvec(C, x)
        
        def H_jac(x, t):
            return C
        
        ms_kf, Ps_kf = kalman_filter(
            ys, A, params["B"], C, params["D"],
            params["m0"], params["P0"], eps=0.0, dtype=dtype
        )
        ms_ekf, Ps_ekf = ekf_filter(
            ys, f, F_jac, h, H_jac, Q, R,
            params["m0"], params["P0"], u_seq=None, eps=0.0, dtype=dtype
        )
        
        assert_close("EKF mean vs KF (1D)", ms_ekf, ms_kf, atol=1e-9, rtol=1e-9)
        assert_close("EKF cov vs KF (1D)", Ps_ekf, Ps_kf, atol=1e-8, rtol=1e-8)
        
    
    def test_ekf_equals_kf_multidim(self, linear_multidim_tf_params):
        """EKF should equal KF exactly for linear multi-dim system."""
        params = linear_multidim_tf_params
        dtype = params["dtype"]
        A, C = params["A"], params["C"]
        Q, R = params["Q"], params["R"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=100,
            A=params["A_np"], Q=params["Q_np"],
            C=params["C_np"], R=params["R_np"],
            m0=params["m0_np"], P0=params["P0_np"],
            seed=222
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        def f(x, u, t):
            return tf.linalg.matvec(A, x)
        
        def F_jac(x, u, t):
            return A
        
        def h(x, t):
            return tf.linalg.matvec(C, x)
        
        def H_jac(x, t):
            return C

        ms_kf, Ps_kf = kalman_filter(
            ys, A, params["B"], C, params["D"],
            params["m0"], params["P0"], eps=0.0, dtype=dtype
        )
        ms_ekf, Ps_ekf = ekf_filter(
            ys, f, F_jac, h, H_jac, Q, R,
            params["m0"], params["P0"], u_seq=None, eps=0.0, dtype=dtype
        )
        
        assert_close("EKF mean vs KF (multi)", ms_ekf, ms_kf, atol=1e-9, rtol=1e-9)
        assert_close("EKF cov vs KF (multi)", Ps_ekf, Ps_kf, atol=1e-8, rtol=1e-8)
    
    
    def test_ekf_covariance_valid(self, linear_multidim_tf_params):
        """EKF covariances should remain symmetric and PD."""
        params = linear_multidim_tf_params
        dtype = params["dtype"]
        A, C = params["A"], params["C"]
        Q, R = params["Q"], params["R"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=50,
            A=params["A_np"], Q=params["Q_np"],
            C=params["C_np"], R=params["R_np"],
            m0=params["m0_np"], P0=params["P0_np"],
            seed=333
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        def f(x, u, t):
            return tf.linalg.matvec(A, x)
        
        def F_jac(x, u, t):
            return A
        
        def h(x, t):
            return tf.linalg.matvec(C, x)
        
        def H_jac(x, t):
            return C

        ms_ekf, Ps_ekf = ekf_filter(
            ys, f, F_jac, h, H_jac, Q, R,
            params["m0"], params["P0"], u_seq=None, eps=0.0, dtype=dtype
        )
        assert_covariance_valid(Ps_ekf.numpy(), "EKF P")


# UKF Tests

class TestUKF:
    
    def test_ukf_equals_kf_1d_random_walk(self, random_walk_1d_tf_params):
        """UKF should equal KF for linear 1D system."""
        params = random_walk_1d_tf_params
        dtype = params["dtype"]
        A, C = params["A"], params["C"]
        Q, R = params["Q"], params["R"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=100,
            A=A.numpy(), Q=Q.numpy(), C=C.numpy(), R=R.numpy(),
            m0=params["m0"].numpy(), P0=params["P0"].numpy(),
            seed=444
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        def f(x, u, t):
            return tf.linalg.matvec(A, x)
        
        def h(x, t):
            return tf.linalg.matvec(C, x)
        
        ms_kf, Ps_kf = kalman_filter(
            ys, A, params["B"], C, params["D"],
            params["m0"], params["P0"], eps=0.0, dtype=dtype
        )
        ms_ukf, Ps_ukf = ukf_filter(
            ys, f, h, Q, R,
            params["m0"], params["P0"], u_seq=None, eps=0.0, dtype=dtype
        )
        
        assert_close("UKF mean vs KF (1D)", ms_ukf, ms_kf, atol=1e-9, rtol=1e-9)
        assert_close("UKF cov vs KF (1D)", Ps_ukf, Ps_kf, atol=1e-8, rtol=1e-8)
        
    
    def test_ukf_equals_kf_multidim(self, linear_multidim_tf_params):
        """UKF should equal KF for linear multi-dim system."""
        params = linear_multidim_tf_params
        dtype = params["dtype"]
        A, C = params["A"], params["C"]
        Q, R = params["Q"], params["R"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=100,
            A=params["A_np"], Q=params["Q_np"],
            C=params["C_np"], R=params["R_np"],
            m0=params["m0_np"], P0=params["P0_np"],
            seed=555
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        def f(x, u, t):
            return tf.linalg.matvec(A, x)
        
        def h(x, t):
            return tf.linalg.matvec(C, x)
        
        ms_kf, Ps_kf = kalman_filter(
            ys, A, params["B"], C, params["D"],
            params["m0"], params["P0"], eps=0.0, dtype=dtype
        )
        ms_ukf, Ps_ukf = ukf_filter(
            ys, f, h, Q, R,
            params["m0"], params["P0"], u_seq=None, eps=0.0, dtype=dtype
        )
        
        assert_close("UKF mean vs KF (multi)", ms_ukf, ms_kf, atol=1e-9, rtol=1e-9)
        assert_close("UKF cov vs KF (multi)", Ps_ukf, Ps_kf, atol=1e-8, rtol=1e-8)
        
    
    def test_ukf_covariance_valid(self, linear_multidim_tf_params):
        """UKF covariances should remain symmetric and PD."""
        params = linear_multidim_tf_params
        dtype = params["dtype"]
        A, C = params["A"], params["C"]
        Q, R = params["Q"], params["R"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=50,
            A=params["A_np"], Q=params["Q_np"],
            C=params["C_np"], R=params["R_np"],
            m0=params["m0_np"], P0=params["P0_np"],
            seed=666
        )
        ys = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        def f(x, u, t):
            return tf.linalg.matvec(A, x)
        
        def h(x, t):
            return tf.linalg.matvec(C, x)
        
        ms_ukf, Ps_ukf = ukf_filter(
            ys, f, h, Q, R,
            params["m0"], params["P0"], u_seq=None, eps=0.0, dtype=dtype
        )
        assert_covariance_valid(Ps_ukf.numpy(), "UKF P")


# PF Tests

class TestPF:
    
    NUM_PARTICLES = 10**6
    MEAN_ATOL = 1e-2
    MEAN_RTOL = 1e-2
    COV_ATOL = 1e-1
    COV_RTOL = 1e-1
    
    @pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
    def test_pf_approx_kf_1d_random_walk(
        self, random_walk_1d_tf_params, random_walk_1d_np_params, resampling_method
    ):
        """PF should approximate KF for linear 1D system."""
        tf_params = random_walk_1d_tf_params
        np_params = random_walk_1d_np_params
        dtype = tf_params["dtype"]
        
        # Simulate observations (use same seed for both)
        xs_np, ys_np = simulate_linear_ssm_np(
            T=100,
            A=np_params["A"], Q=np_params["Q"],
            C=np_params["C"], R=np_params["R"],
            m0=np_params["m0"], P0=np_params["P0"],
            seed=777
        )
        ys_tf = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        ms_kf, Ps_kf = kalman_filter(
            ys_tf, tf_params["A"], tf_params["B"], tf_params["C"], tf_params["D"],
            tf_params["m0"], tf_params["P0"], eps=0.0, dtype=dtype
        )
        
        numpy_ssm = make_numpy_ssm(np_params)
        ms_pf, Ps_pf, info = particle_filter_numpy(
            numpy_ssm, ys_np,
            num_particles=self.NUM_PARTICLES,
            resampling_method=resampling_method,
            seed=42
        )
        
        assert_close(
            f"PF mean vs KF (1D, {resampling_method})",
            ms_pf, ms_kf.numpy(),
            atol=self.MEAN_ATOL, rtol=self.MEAN_RTOL
        )
        assert_close(
            f"PF cov vs KF (1D, {resampling_method})",
            Ps_pf, Ps_kf.numpy(),
            atol=self.COV_ATOL, rtol=self.COV_RTOL
        )

    
    @pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
    def test_pf_approx_kf_multidim(
        self, linear_multidim_tf_params, linear_multidim_np_params, resampling_method
    ):
        """PF should approximate KF for linear multi-dim system."""
        tf_params = linear_multidim_tf_params
        np_params = linear_multidim_np_params
        dtype = tf_params["dtype"]
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=100,
            A=np_params["A"], Q=np_params["Q"],
            C=np_params["C"], R=np_params["R"],
            m0=np_params["m0"], P0=np_params["P0"],
            seed=888
        )
        ys_tf = tf.convert_to_tensor(ys_np, dtype=dtype)
        
        ms_kf, Ps_kf = kalman_filter(
            ys_tf, tf_params["A"], tf_params["B"], tf_params["C"], tf_params["D"],
            tf_params["m0"], tf_params["P0"], eps=0.0, dtype=dtype
        )
        
        numpy_ssm = make_numpy_ssm(np_params)
        ms_pf, Ps_pf, info = particle_filter_numpy(
            numpy_ssm, ys_np,
            num_particles=self.NUM_PARTICLES,
            resampling_method=resampling_method,
            seed=42
        )
        
        assert_close(
            f"PF mean vs KF (multi, {resampling_method})",
            ms_pf, ms_kf.numpy(),
            atol=self.MEAN_ATOL, rtol=self.MEAN_RTOL
        )
        assert_close(
            f"PF cov vs KF (multi, {resampling_method})",
            Ps_pf, Ps_kf.numpy(),
            atol=self.COV_ATOL, rtol=self.COV_RTOL
        )
        
    
    def test_pf_weight_validity(self, random_walk_1d_np_params):
        """PF weights should be valid (sum to 1, non-negative)."""
        params = random_walk_1d_np_params
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=50,
            A=params["A"], Q=params["Q"],
            C=params["C"], R=params["R"],
            m0=params["m0"], P0=params["P0"],
            seed=999
        )
        
        numpy_ssm = make_numpy_ssm(params)
        ms_pf, Ps_pf, info = particle_filter_numpy(
            numpy_ssm, ys_np,
            num_particles=1000,
            return_particles=True,
            seed=42
        )
        
        weights = info["weights"]  # [T+1, N]
        
        # Check non-negative
        assert np.all(weights >= 0), "Weights contain negative values"
        
        # Check sum to 1 (approximately)
        weight_sums = np.sum(weights, axis=1)  # [T+1]
        assert np.allclose(weight_sums, 1.0, atol=1e-10), \
            f"Weights don't sum to 1: min={weight_sums.min()}, max={weight_sums.max()}"

    
    def test_pf_reproducibility(self, random_walk_1d_np_params):
        """PF with same seed should give identical results."""
        params = random_walk_1d_np_params
        
        xs_np, ys_np = simulate_linear_ssm_np(
            T=50,
            A=params["A"], Q=params["Q"],
            C=params["C"], R=params["R"],
            m0=params["m0"], P0=params["P0"],
            seed=1010
        )
        
        numpy_ssm = make_numpy_ssm(params)
        
        # Run twice with same seed
        ms_pf1, Ps_pf1, _ = particle_filter_numpy(
            numpy_ssm, ys_np, num_particles=1000, seed=42
        )
        ms_pf2, Ps_pf2, _ = particle_filter_numpy(
            numpy_ssm, ys_np, num_particles=1000, seed=42
        )
        
        # Should be exactly equal
        np.testing.assert_array_equal(ms_pf1, ms_pf2, err_msg="PF means differ with same seed")
        np.testing.assert_array_equal(Ps_pf1, Ps_pf2, err_msg="PF covs differ with same seed")
        
        # Run with different seed - should differ
        ms_pf3, Ps_pf3, _ = particle_filter_numpy(
            numpy_ssm, ys_np, num_particles=1000, seed=99
        )
        assert not np.allclose(ms_pf1, ms_pf3), "PF results identical with different seeds"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
