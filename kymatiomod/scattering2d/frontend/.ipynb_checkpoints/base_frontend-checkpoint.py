from ...frontend.base_frontend import ScatteringBase

from ..filter_bank import filter_bank, circular_harmonic_filter_bank, gaussian_filter_bank
from ..utils import compute_padding

class HarmonicScatteringBase2D(ScatteringBase):
    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2,
                 rotation_covariant=True, method='integral', points=None,
                 integral_powers=(0.5, 1., 2.), backend=None):
        print(backend)
        # print(
        super(HarmonicScatteringBase2D, self).__init__()
        self.J = J
        self.shape = shape
        self.L = L
        self.sigma_0 = sigma_0

        self.max_order = max_order
        self.rotation_covariant = rotation_covariant
        self.method = method
        self.points = points
        self.integral_powers = integral_powers
        self.backend = backend
        # self.out_type = out_type


    def build(self):
        M, N = self.shape

        if 2 ** self.J > M or 2 ** self.J > N:
            raise RuntimeError('The smallest dimension should be larger than 2^J.')
        self._M_padded, self._N_padded = compute_padding(M, N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        self.pad = self.backend.Pad([(self._M_padded - M) // 2, (self._M_padded - M+1) // 2, (self._N_padded - N) // 2,
                                (self._N_padded - N + 1) // 2], [M, N])

        self.unpad = self.backend.unpad

    @property
    def M(self):
        # warn("The attribute M is deprecated and will be removed in v0.4. "
        # "Replace by shape[0].", DeprecationWarning)
        return int(self.shape[0])

    @property
    def N(self):
        # warn("The attribute N is deprecated and will be removed in v0.4. "
        # "Replace by shape[1].", DeprecationWarning)
        return int(self.shape[1])

    def create_filters(self):
        self.filters = circular_harmonic_filter_bank(
            self.M, self.N, self.J, self.L, self.sigma_0)

        self.gaussian_filters = gaussian_filter_bank(
            self.M, self.N, self.J + 1, self.sigma_0)

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError

    _doc_shape = 'M, N'

    _doc_class = \
    r"""The 2D circular harmonic scattering transform

        This class implements circular harmonic scattering on a 2D input image.

        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 3
            M, N = 32, 32

            # Generate a sample signal.
            x = {sample}

            # Define a HarmonicScattering3D object.
            S = HarmonicScattering3D(J, (M, N))

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}

        Parameters
        ----------
        J: int
            Number of scales.
        shape: tuple of ints
            Shape `(M, N)` of the input signal
        L: int, optional
            Number of `l` values. Defaults to `3`.
        sigma_0: float, optional
            Bandwidth of mother wavelet. Defaults to `1`.
        max_order: int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        rotation_covariant: bool, optional
            If set to `True` the first-order moduli take the form:

            $\sqrt{{\sum_m (x \star \psi_{{j,l,m}})^2)}}$

            if set to `False` the first-order moduli take the form:

            $x \star \psi_{{j,l,m}}$

            The second order moduli change analogously. Defaults to `True`.
        method: string, optional
            Specifies the method for obtaining scattering coefficients.
            Currently, only `'integral'` is available. Defaults to `'integral'`.
        integral_powers: array-like
            List of exponents to the power of which moduli are raised before
            integration.
        """

    _doc_scattering = \
    """Apply the scattering transform

       Parameters
       ----------
       input_array: {array}
           Input of size `(batch_size, M, N)`.

       Returns
       -------
       output: {array}
           If max_order is `1` it returns a{n} `{array}` with the first-order
           scattering coefficients. If max_order is `2` it returns a{n}
           `{array}` with the first- and second- order scattering
           coefficients, concatenated along the feature axis.
    """

    @classmethod
    def _document(cls):
        cls.__doc__ = HarmonicScatteringBase2D._doc_class.format(
            array=cls._doc_array,
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_name=cls._doc_alias_name,
            alias_call=cls._doc_alias_call.format(x="x"),
            sample=cls._doc_sample.format(shape=cls._doc_shape))

        # Sphinx will not show docstrings for inherited methods, so we add a
        # dummy method here that will just call the super.
        if not "scattering" in cls.__dict__:
            def _scattering(self, x):
                return super(cls, self).scattering(x)

            setattr(cls, "scattering", _scattering)

        cls.scattering.__doc__ = HarmonicScatteringBase2D._doc_scattering.format(
            array=cls._doc_array,
            n=cls._doc_array_n)



class ScatteringBase2D(ScatteringBase):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend=None, out_type='array'):
        super(ScatteringBase2D, self).__init__()
        self.pre_pad = pre_pad
        self.L = L
        self.backend = backend
        self.J = J
        self.shape = shape
        self.max_order = max_order
        self.out_type = out_type

    def build(self):
        M, N = self.shape

        if 2 ** self.J > M or 2 ** self.J > N:
            raise RuntimeError('The smallest dimension should be larger than 2^J.')
        self._M_padded, self._N_padded = compute_padding(M, N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        if not self.pre_pad:
            self.pad = self.backend.Pad([(self._M_padded - M) // 2, (self._M_padded - M+1) // 2, (self._N_padded - N) // 2,
                                (self._N_padded - N + 1) // 2], [M, N])
        else:
            self.pad = lambda x: x

        self.unpad = self.backend.unpad

    def create_filters(self):
        filters = filter_bank(self._M_padded, self._N_padded, self.J, self.L)
        self.phi, self.psi = filters['phi'], filters['psi']

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError

    @property
    def M(self):
        # warn("The attribute M is deprecated and will be removed in v0.4. "
        # "Replace by shape[0].", DeprecationWarning)
        return int(self.shape[0])

    @property
    def N(self):
        # warn("The attribute N is deprecated and will be removed in v0.4. "
        # "Replace by shape[1].", DeprecationWarning)
        return int(self.shape[1])

    _doc_shape = 'M, N'

    _doc_instantiation_shape = {True: 'S = Scattering2D(J, (M, N))',
                                False: 'S = Scattering2D(J)'}

    _doc_param_shape = r"""
        shape : tuple of ints
            Spatial support (M, N) of the input."""

    _doc_attrs_shape = r"""
        Psi : dictionary
            Contains the wavelets filters at all resolutions. See
            `filter_bank.filter_bank` for an exact description.
        Phi : dictionary
            Contains the low-pass filters at all resolutions. See
            `filter_bank.filter_bank` for an exact description.
        M_padded, N_padded : int
             Spatial support of the padded input."""

    _doc_param_out_type = r"""
        out_type : str, optional
            The format of the output of a scattering transform. If set to
            `'list'`, then the output is a list containing each individual
            scattering path with meta information. Otherwise, if set to
            `'array'`, the output is a large array containing the
            concatenation of all scattering coefficients. Defaults to
            `'array'`."""

    _doc_attr_out_type = r"""
        out_type : str
            The format of the scattering output. See documentation for
            `out_type` parameter above and the documentation for
            `scattering`."""

    _doc_class = \
    r"""The 2D scattering transform

        The scattering transform computes two wavelet transform
        followed by modulus non-linearity. It can be summarized as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_J^{{(0)}} x = x \star \phi_J$,

            $S_J^{{(1)}} x = [|x \star \psi^{{(1)}}_\lambda| \star \phi_J]_\lambda$, and

            $S_J^{{(2)}} x = [||x \star \psi^{{(1)}}_\lambda| \star
            \psi^{{(2)}}_\mu| \star \phi_J]_{{\lambda, \mu}}$.

        where $\star$ denotes the convolution (in space), $\phi_J$ is a
        lowpass filter, $\psi^{{(1)}}_\lambda$ is a family of bandpass filters
        and $\psi^{{(2)}}_\mu$ is another family of bandpass filters. Only
        Morlet filters are used in this implementation. Convolutions are
        efficiently performed in the Fourier domain.{frontend_paragraph}

        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 3
            M, N = 32, 32

            # Generate a sample signal.
            x = {sample}

            # Define a Scattering2D object.
            {instantiation}

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}

        Parameters
        ----------
        J : int
            Log-2 of the scattering scale.{param_shape}
        L : int, optional
            Number of angles used for the wavelet transform. Defaults to `8`.
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        pre_pad : boolean, optional
            Controls the padding: if set to False, a symmetric padding is
            applied on the signal. If set to True, the software will assume
            the signal was padded externally. Defaults to `False`.
        backend : object, optional
            Controls the backend which is combined with the frontend.{param_out_type}

        Attributes
        ----------
        J : int
            Log-2 of the scattering scale.{param_shape}
        L : int, optional
            Number of angles used for the wavelet transform.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either `1` or `2`.
        pre_pad : boolean
            Controls the padding: if set to False, a symmetric padding is
            applied on the signal. If set to True, the software will assume
            the signal was padded externally.{attrs_shape}{attr_out_type}

        Notes
        -----
        The design of the filters is optimized for the value `L = 8`.

        The `pre_pad` flag is particularly useful when cropping bigger images
        because this does not introduce border effects inherent to padding."""

    _doc_scattering = \
    """Apply the scattering transform

       Parameters
       ----------
       input : {array}
           An input `{array}` of size `(B, M, N)`.

       Raises
       ------
       RuntimeError
           In the event that the input does not have at least two dimensions,
           or the tensor is not contiguous, or the tensor is not of the
           correct spatial size, padded or not.
       TypeError
           In the event that the input is not of type `{array}`.

       Returns
       -------
       S : {array}
           Scattering transform of the input. If `out_type` is set to
           `'array'` (or if it is not availabel for this frontend), this is
           a{n} `{array}` of shape `(B, C, M1, N1)` where `M1 = M // 2 ** J`
           and `N1 = N // 2 ** J`. The `C` is the number of scattering
           channels calculated. If `out_type` is `'list'`, the output is a
           list of dictionaries, with each dictionary corresponding to a
           scattering coefficient and its meta information. The actual
           coefficient is contained in the `'coef'` key, while other keys hold
           additional information, such as `'j'` (the scale of the filter
           used), and `'theta'` (the angle index of the filter used).
    """


    @classmethod
    def _document(cls):
        instantiation = cls._doc_instantiation_shape[cls._doc_has_shape]
        param_shape = cls._doc_param_shape if cls._doc_has_shape else ''
        attrs_shape = cls._doc_attrs_shape if cls._doc_has_shape else ''

        param_out_type = cls._doc_param_out_type if cls._doc_has_out_type else ''
        attr_out_type = cls._doc_attr_out_type if cls._doc_has_out_type else ''

        cls.__doc__ = ScatteringBase2D._doc_class.format(
            array=cls._doc_array,
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_name=cls._doc_alias_name,
            alias_call=cls._doc_alias_call.format(x="x"),
            instantiation=instantiation,
            param_shape=param_shape,
            attrs_shape=attrs_shape,
            param_out_type=param_out_type,
            attr_out_type=attr_out_type,
            sample=cls._doc_sample.format(shape=cls._doc_shape))

        # Sphinx will not show docstrings for inherited methods, so we add a
        # dummy method here that will just call the super.
        if not "scattering" in cls.__dict__:
            def _scattering(self, x):
                return super(cls, self).scattering(x)

            setattr(cls, "scattering", _scattering)


        cls.scattering.__doc__ = ScatteringBase2D._doc_scattering.format(
            array=cls._doc_array,
            n=cls._doc_array_n)


__all__ = ['ScatteringBase2D', 'HarmonicScatteringBase2D']
