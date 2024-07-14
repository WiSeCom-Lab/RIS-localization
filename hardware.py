import numpy as np
import scipy

def MutualCoupling(N, ideal=True, c_min=0.01, c_max=0.4):
    if ideal is True:
        M = np.eye(N, dtype="complex128")
    else:
        # v = np.concatenate((np.array([1]), np.array(c_min+(c_max-c_min)*np.random.rand(N-1))), dtype="complex128")
        # M = scipy.linalg.toeplitz(v)
        M = c_min+(c_max-c_min)*np.random.rand(N,N)
        M = M - np.diag(np.diag(M)) + np.eye(N, dtype="complex128")
    return M

def PhaseGainError(N, ideal=True, g_error=0.05, p_error=20):
    if ideal is True:
        gamma = np.eye(N, dtype="complex128")
    else:
        gain = 1 + g_error*np.random.randn(N)
        phase = (p_error/180)*np.pi*np.random.randn(N)
        gamma = np.diag(gain*np.exp(1j*phase))
    return gamma

def RISmodel(phases, ideal=True, beta_min=0.2, gamma=0.43*np.pi, alpha=1.6):
    if ideal is True:
        coefficients = np.exp(1j*phases)
    else:
        coefficients = ((1-beta_min)*(((np.sin(phases-gamma)+1)/2)**alpha)+beta_min)*np.exp(1j*phases)
    return coefficients