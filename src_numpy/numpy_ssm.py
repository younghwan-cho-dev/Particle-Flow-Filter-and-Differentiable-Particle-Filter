import numpy as np
from numpy.random import Generator, default_rng
from scipy import stats
from typing import Optional, Callable, Union, Tuple


class NumpySSM:
    
    def __init__(
        self,
        # Linear dynamics (required)
        A: np.ndarray,
        Q: np.ndarray,
        m0: np.ndarray,
        P0: np.ndarray,
        
        # Observation - Option 1: Linear Gaussian
        C: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        
        # Observation - Option 2: Nonlinear Gaussian
        obs_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        
        # Observation - Option 3: Custom log-likelihood
        obs_loglik_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        
        # For simulation with custom obs
        obs_sample_fn: Optional[Callable[[np.ndarray, Generator], np.ndarray]] = None,
    ):
        """
        Args:
            A: [nx, nx] State transition matrix
            Q: [nx, nx] Process noise covariance
            m0: [nx] Initial state mean
            P0: [nx, nx] Initial state covariance
            C: [ny, nx] Observation matrix (for linear obs)
            R: [ny, ny] Observation noise covariance (for linear/nonlinear Gaussian)
            obs_fn: Callable, x [N, nx] -> y_mean [N, ny] (for nonlinear Gaussian)
            obs_loglik_fn: Callable, (x [N, nx], y [ny]) -> loglik [N] (for custom)
            obs_sample_fn: Callable, (x [nx], rng) -> y [ny] (for custom simulation)
        """
        self.A = np.asarray(A, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.m0 = np.asarray(m0, dtype=np.float64)
        self.P0 = np.asarray(P0, dtype=np.float64)
        
        self.nx = self.A.shape[0]
        
        self.Q_chol = np.linalg.cholesky(self.Q)
        self.P0_chol = np.linalg.cholesky(self.P0)
        
        self._setup_observation_mode(C, R, obs_fn, obs_loglik_fn, obs_sample_fn)
    
    def _setup_observation_mode(
        self,
        C: Optional[np.ndarray],
        R: Optional[np.ndarray],
        obs_fn: Optional[Callable],
        obs_loglik_fn: Optional[Callable],
        obs_sample_fn: Optional[Callable],
    ):
        
        if C is not None:
            # Mode 1: Linear Gaussian
            self.obs_mode = "linear"
            self.C = np.asarray(C, dtype=np.float64)
            self.R = np.asarray(R, dtype=np.float64)
            self.ny = self.C.shape[0]
            self.R_chol = np.linalg.cholesky(self.R)
            # Precompute for log-likelihood
            self.R_inv = np.linalg.inv(self.R)
            self.R_logdet = np.linalg.slogdet(self.R)[1]
            
        elif obs_fn is not None and obs_loglik_fn is None:
            # Mode 2: Nonlinear Gaussian
            self.obs_mode = "nonlinear_gaussian"
            self.obs_fn = obs_fn
            self.R = np.asarray(R, dtype=np.float64)
            self.ny = self.R.shape[0]
            self.R_chol = np.linalg.cholesky(self.R)
            self.R_inv = np.linalg.inv(self.R)
            self.R_logdet = np.linalg.slogdet(self.R)[1]
            
        elif obs_loglik_fn is not None:
            # Mode 3: Custom log-likelihood
            self.obs_mode = "custom"
            self.obs_loglik_fn = obs_loglik_fn
            self.obs_sample_fn = obs_sample_fn
            self.ny = None 
            
        else:
            raise ValueError(
                "Must provide one of: C (linear), obs_fn (nonlinear Gaussian), "
                "or obs_loglik_fn (custom)"
            )
    
    def sample_initial(self, N: int, rng: Generator) -> np.ndarray:
        """
        Sample N particles from initial distribution.
        
        Args:
            N: Number of particles
            rng: NumPy random generator
            
        Returns:
            x: [N, nx] Initial particles
        """
        noise = rng.standard_normal((N, self.nx))
        return self.m0[None, :] + noise @ self.P0_chol.T
    
    def sample_transition(self, x: np.ndarray, rng: Generator) -> np.ndarray:
        """
        Sample x_t from transition distribution given x_{t-1}.
        
        Args:
            x: [N, nx] Current particles
            rng: NumPy random generator
            
        Returns:
            x_new: [N, nx] Propagated particles
        """
        N = x.shape[0]
        mean = x @ self.A.T  
        noise = rng.standard_normal((N, self.nx))
        return mean + noise @ self.Q_chol.T
    
    def observation_loglik(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute observation loglikelihood for all particles.
        
        Args:
            x: [N, nx] Particles
            y: [ny] Observation
            
        Returns:
            loglik: [N] Loglik
        """
        if self.obs_mode == "linear":
            return self._loglik_linear(x, y)
        elif self.obs_mode == "nonlinear_gaussian":
            return self._loglik_nonlinear_gaussian(x, y)
        elif self.obs_mode == "custom":
            return self.obs_loglik_fn(x, y)
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
    
    def _loglik_linear(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log-likelihood for linear Gaussian observation."""
        y_pred = x @ self.C.T
        residual = y[None, :] - y_pred 

        solved = np.linalg.solve(self.R_chol, residual.T) 
        mahal_sq = np.sum(solved ** 2, axis=0)
        
        log_prob = -0.5 * (
            self.ny * np.log(2 * np.pi) + self.R_logdet + mahal_sq
        )
        return log_prob
    
    def _loglik_nonlinear_gaussian(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log-likelihood for nonlinear Gaussian observation."""
        y_pred = self.obs_fn(x)
        residual = y[None, :] - y_pred 
        
        solved = np.linalg.solve(self.R_chol, residual.T)
        mahal_sq = np.sum(solved ** 2, axis=0)
        
        log_prob = -0.5 * (
            self.ny * np.log(2 * np.pi) + self.R_logdet + mahal_sq
        )
        return log_prob
    
    def sample_observation(self, x: np.ndarray, rng: Generator) -> np.ndarray:
        """
        
        Args:
            x: [nx] Single state vector
            rng: NumPy random generator
            
        Returns:
            y: [ny] Observation
        """
        if self.obs_mode == "linear":
            y_mean = self.C @ x
            noise = rng.standard_normal(self.ny)
            return y_mean + self.R_chol @ noise
            
        elif self.obs_mode == "nonlinear_gaussian":
            y_mean = self.obs_fn(x[None, :])[0] 
            noise = rng.standard_normal(self.ny)
            return y_mean + self.R_chol @ noise
            
        elif self.obs_mode == "custom":
            if self.obs_sample_fn is None:
                raise ValueError(
                    "obs_sample_fn required for simulation with custom observation"
                )
            return self.obs_sample_fn(x, rng)
        
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
    
    def simulate(
        self,
        T: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate trajectory and observations.
        
        Args:
            T: Number of time steps
            seed: Random seed (optional)
            
        Returns:
            xs: [T+1, nx] State trajectory (x_0, ..., x_T)
            ys: [T, ny] Observations (y_1, ..., y_T)
        """
        rng = default_rng(seed)
        
        xs = np.zeros((T + 1, self.nx))

        noise = rng.standard_normal(self.nx)
        xs[0] = self.m0 + self.P0_chol @ noise
        

        if self.ny is None:
            y_test = self.sample_observation(xs[0], rng)
            self.ny = len(y_test)
        
        ys = np.zeros((T, self.ny))
        
        # Simulate
        for t in range(1, T + 1):
            noise = rng.standard_normal(self.nx)
            xs[t] = self.A @ xs[t - 1] + self.Q_chol @ noise
            
            ys[t - 1] = self.sample_observation(xs[t], rng)
        
        return xs, ys
    
    def __repr__(self) -> str:
        return (
            f"NumpySSM(nx={self.nx}, ny={self.ny}, obs_mode='{self.obs_mode}')"
        )
