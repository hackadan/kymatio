import tensorflow as tf
from collections import namedtuple
import numpy as np

BACKEND_NAME = 'tensorflow'

from ...backend.tensorflow_backend import TensorFlowBackend

class Pad(object):
    def __init__(self, pad_size, input_size):
        """
            Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.
            Parameters
            ----------
            pad_size : list of 4 integers
                size of padding to apply.
            input_size : list of 2 integers
                size of the original signal
        """
        self.pad_size = pad_size
        self.input_size = input_size

    def __call__(self, x):
        pad_size = list(self.pad_size)

        # Clone to avoid passing on modifications.
        new_pad_size = list(pad_size)

        # This handles the case where the padding is equal to the image size.
        if pad_size[0] == self.input_size[0]:
            new_pad_size[0] -= 1
            new_pad_size[1] -= 1
        if pad_size[2] == self.input_size[1]:
            new_pad_size[2] -= 1
            new_pad_size[3] -= 1

        paddings = [[0, 0]] * len(x.shape[:-2])
        paddings += [[new_pad_size[0], new_pad_size[1]], [new_pad_size[2], new_pad_size[3]]]

        x_padded = tf.pad(x, paddings, mode="REFLECT")

        # Again, special handling for when padding is the same as image size.
        if pad_size[0] == self.input_size[0]:
            x_padded = tf.concat([tf.expand_dims(x_padded[..., 1, :], axis=-2), x_padded, tf.expand_dims(x_padded[..., x_padded.shape[-2] -2, :], axis=-2)], axis=-2)
        if pad_size[2] == self.input_size[1]:
            x_padded = tf.concat([tf.expand_dims(x_padded[..., :, 1], axis=-1), x_padded, tf.expand_dims(x_padded[..., :,  x_padded.shape[-1]-2], axis=-1)], axis=-1)

        return x_padded

class TensorFlowBackend2D(TensorFlowBackend):
    Pad = Pad

    @staticmethod
    def unpad(in_):
        """
            Slices the input tensor at indices between 1::-1
            Parameters
            ----------
            in_ : tensor_like
                input tensor
            Returns
            -------
            in_[..., 1:-1, 1:-1]
        """
        return in_[..., 1:-1, 1:-1]


    # @classmethod
    # def rfft(cls, x):
    #     cls.real_check(x)
    #     return tf.signal.fft2d(tf.cast(x, tf.complex64), name='rfft2d')

    @classmethod
    def irfft(cls, x):
        cls.complex_check(x)
        return tf.math.real(tf.signal.ifft2d(x, name='irfft2d'))


    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)
        return tf.signal.ifft2d(x, name='ifft2d')

    @classmethod
    def subsample_fourier(cls, x, k):
        """ Subsampling of a 2D image performed in the Fourier domain.

        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor_like
            input tensor with at least three dimensions.
        k : int
            integer such that x is subsampled by k along the spatial variables.

        Returns
        -------
        out : tensor_like
            Tensor such that its Fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k]

        """
        cls.complex_check(x)

        y = tf.reshape(x, (-1, k, x.shape[1] // k, k, x.shape[2] // k))

        out = tf.reduce_mean(y, axis=(1, 3))
        return out
        
    @staticmethod
    def modulus_rotation(x, module):
        """Used for computing rotation invariant scattering transform coefficents.

            Parameters
            ----------
            x : tensor
                Size (batchsize, M, N, O).
            module : tensor
                Tensor that holds the overall sum.

            Returns
            -------
            output : tensor
                Tensor of the same size as input_array. It holds the output of
                the operation::

                $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$

                which is covariant to 3D translations and rotations.

        """
        if module is None:
            module = tf.abs(x) ** 2
        else:
            module = module ** 2 + tf.abs(x) ** 2
        return tf.sqrt(module)

    @staticmethod
    def compute_integrals(input_array, integral_powers):
        """Computes integrals.

            Computes integrals of the input_array to the given powers.

            Parameters
            ----------
            input_array : tensor
                Size (B, M, N, O), where B is batch_size, and M, N, O are spatial
                dims.
            integral_powers : list
                List of P positive floats containing the p values used to
                compute the integrals of the input_array to the power p (l_p
                norms).

            Returns
            -------
            integrals : tensor
                Tensor of size (B, P) containing the integrals of the input_array
                to the powers p (l_p norms).

        """
        integrals = []
        for i_q, q in enumerate(integral_powers):
            integrals.append(tf.reduce_sum(tf.reshape(tf.pow(input_array, q), shape=(tf.shape(input_array)[0], -1)), axis=1))
        return tf.stack(integrals, axis=-1)

    @staticmethod
    def cdgmm2d(A, B):
        """Complex pointwise multiplication.

            Complex pointwise multiplication between (batched) tensor A and tensor B.

            Parameters
            ----------
            A : tensor
                Complex tensor.
            B : tensor
                Complex tensor of the same size as A.

            Returns
            -------
            output : tensor
                Tensor of the same size as A containing the result of the elementwise
                complex multiplication of A with B.

        """
        # if B.ndim != 3:
        #     raise RuntimeError('The dimension of the second input must be 3.')

        Cr = tf.cast(tf.math.real(A) * np.real(B) - tf.math.imag(A) * np.imag(B), tf.complex64)
        Ci = tf.cast(tf.math.real(A) * np.imag(B) + tf.math.imag(A) * np.real(B), tf.complex64)

        return Cr + 1.0j * Ci

    # @staticmethod
    # def stack(arrays, L):
    #     S = tf.stack(arrays, axis=1)
    #     S = tf.reshape(S, ((tf.shape(S)[0], tf.shape(S)[1] // (L + 1), (L + 1))) + (tf.shape(S)[2:]))
        
    #     return S

    @classmethod
    def rfft(cls, x):
        cls.real_check(x)
        return tf.signal.fft2d(tf.cast(x, tf.complex64), name='rfft2d')

    # @classmethod
    # def ifft(cls, x):
    #     cls.complex_check(x)
    #     return tf.signal.ifft2d(x, name='ifft2d')

backend = TensorFlowBackend2D
