import scipy.fft
import warnings
import numpy as np


def compute_padding(M, N, J):
    """
         Precomputes the future padded size. If 2^J=M or 2^J=N,
         border effects are unavoidable in this case, and it is
         likely that the input has either a compact support,
         either is periodic.

         Parameters
         ----------
         M, N : int
             input size

         Returns
         -------
         M, N : int
             padded size
    """
    M_padded = ((M + 2 ** J) // 2 ** J + 1) * 2 ** J
    N_padded = ((N + 2 ** J) // 2 ** J + 1) * 2 ** J

    return M_padded, N_padded
    
def sqrt(x):
    """
        Compute the square root of an array
        This suppresses any warnings due to invalid input, unless the array is
        real and has negative values. This fixes the erroneous warnings
        introduced by an Intel SVM bug for large single-precision arrays. For
        more information, see:
            https://github.com/numpy/numpy/issues/11448
            https://github.com/ContinuumIO/anaconda-issues/issues/9129
        Parameters
        ----------
        x : numpy array
            An array for which we would like to compute the square root.
        Returns
        -------
        y : numpy array
            The square root of the array.
    """
    if np.isrealobj(x) and (x < 0).any():
        warnings.warn("Negative value encountered in sqrt", RuntimeWarning,
            stacklevel=1)
    old_settings = np.seterr(invalid='ignore')
    y = np.sqrt(x)
    np.seterr(**old_settings)

    return y
