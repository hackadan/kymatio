import numpy as np
from scipy.fft import fft2, ifft2
from utils import sqrt

def circular_harmonic_filter_bank(M, N, J, L, sigma_0, fourier=True):
    """
        Computes a set of 2D circular Harmonic Wavelets of scales j = [0, ..., J]
        and first orders l = [0, ..., L].

        Parameters
        ----------
        M, N : int
            spatial sizes
        J : int
            maximal scale of the wavelets
        L : int
            maximal first order of the wavelets
        sigma_0 : float
            width parameter of mother solid harmonic wavelet
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        filters : list of ndarray
            the element number l of the list is a torch array array of size
            (J+1, 2l+1, M, N, 2) containing the (J+1)x(2l+1) wavelets of order l.
    """
    filters = []
    for l in range(L + 1):
        filters_l = np.zeros((J + 1, M, N), dtype='complex64')
        for j in range(J+1):
            sigma = sigma_0 * 1.5 ** j
            filters_l[j,...] = circular_harmonic_2d(M, N, sigma, l, fourier=fourier)
        filters.append(filters_l)
    return filters


def circular_harmonic_2d(M, N, sigma, l, fourier=True):
    """
        Computes a set of 2D circular Harmonic Wavelets.
	A circular harmonic wavelet has two integer orders l >= 0 and -l <= m <= l
	In spherical coordinates (r, theta, phi), a circular harmonic wavelet is
	the product of a polynomial Gaussian r^l exp(-0.5 r^2 / sigma^2)
	with a spherical harmonic function Y_{l,m} (theta, phi).

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            width parameter of the solid harmonic wavelets
        l : int
            first integer order of the wavelets
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        solid_harm : ndarray, type complex64
            numpy array of size (2l+1, M, N, 0) and type complex64 containing
            the 2l+1 wavelets of order (l , m) with -l <= m <= l.
            It is ifftshifted such that the origin is at the point [., 0, 0, 0]
    """
    circ_harm = np.zeros((M, N), np.complex64)
    grid = np.fft.ifftshift(
        np.mgrid[-M // 2 : -M // 2 + M,
                 -N // 2:-N // 2 + N].astype('float32'),
        axes=(1,2))
    _sigma = sigma

    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        _sigma = 1. / sigma

    # r-coeffs for the scaling of the solution
    r_square = (grid ** 2).sum(0)
    r_power_l = sqrt(r_square ** l)

    gaussian = np.exp(-0.5 * r_square / _sigma**2).astype('complex64')

    if l == 0:
        if fourier:
            return gaussian.reshape((1, M, N))
        return gaussian.reshape((1, M, N)) / (
                                          (2 * np.pi) * _sigma ** 2)

    polynomial_gaussian = r_power_l * gaussian / _sigma ** l
    # polynomial_gaussian = gaussian

    # polar, azimuthal = get_3d_angles(grid)
    theta = np.arctan2(grid[0], grid[1])

    # for i_m, m in enumerate(range(-l, l + 1)):
        #i.e. 1j amounts to the complex-coeff i, m is each candidate frequency, theta is all angles within the image/grid.
    circ_harm = np.exp(1j * l * theta) * polynomial_gaussian

    # if l % 2 == 0:
    #     norm_factor = 1. / (2 * np.pi * np.sqrt(l + 0.5) * 
    #                                         double_factorial(l + 1))
    # else :
    #     norm_factor = 1. / (2 ** (0.5 * ( l + 3)) * 
    #                         np.sqrt(np.pi * (2 * l + 1)) * 
    #                         factorial((l + 1) / 2))

    norm_factor = 1 / (2 * np.pi)
    
    if fourier:
        norm_factor *= (2 * np.pi) * (-1j) ** l
    else:
        norm_factor /= _sigma ** 2

    circ_harm *= norm_factor

    return circ_harm
    

def filter_bank(M, N, J, L=8):
    """
        Builds in Fourier the Morlet filters used for the scattering transform.
        Each single filter is provided as a dictionary with the following keys:
        * 'j' : scale
        * 'theta' : angle used
        Parameters
        ----------
        M, N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
        Returns
        -------
        filters : list
            A two list of dictionary containing respectively the low-pass and
             wavelet filters.
        Notes
        -----
        The design of the filters is optimized for the value L = 8.
    """
    filters = {}
    filters['psi'] = []

    for j in range(J):
        for theta in range(L):
            psi = {'levels': [], 'j': j, 'theta': theta}
            psi_signal = morlet_2d(M, N, 0.8 * 2**j,
                (int(L-L/2-1)-theta) * np.pi / L,
                3.0 / 4.0 * np.pi /2**j, 4.0/L)
            psi_signal_fourier = np.real(fft2(psi_signal))
            # drop the imaginary part, it is zero anyway
            psi_levels = []
            for res in range(min(j + 1, max(J - 1, 1))):
                psi_levels.append(periodize_filter_fft(psi_signal_fourier, res))
            psi['levels'] = psi_levels
            filters['psi'].append(psi)

    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0)
    phi_signal_fourier = np.real(fft2(phi_signal))
    # drop the imaginary part, it is zero anyway
    filters['phi'] = {'levels': [], 'j': J}
    for res in range(J):
        filters['phi']['levels'].append(
            periodize_filter_fft(phi_signal_fourier, res))

    return filters


def periodize_filter_fft(x, res):
    """
        Parameters
        ----------
        x : numpy array
            signal to periodize in Fourier
        res :
            resolution to which the signal is cropped.

        Returns
        -------
        crop : numpy array
            It returns a crop version of the filter, assuming that
             the convolutions will be done via compactly supported signals.
    """
    M = x.shape[0]
    N = x.shape[1]

    crop = np.zeros((M // 2 ** res, N // 2 ** res), x.dtype)

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** (-res)))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** (-res)))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x,:] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x,mask)

    for k in range(int(M / 2 ** res)):
        for l in range(int(N / 2 ** res)):
            for i in range(int(2 ** res)):
                for j in range(int(2 ** res)):
                    crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]

    return crop


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0):
    """
        Computes a 2D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab /= norm_factor

    return gab


__all__ = ['filter_bank']
