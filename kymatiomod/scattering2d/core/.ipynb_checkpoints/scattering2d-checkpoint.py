import warnings
def harmonic_scattering2d(x, filters, rotation_covariant, L, J, max_order, backend, averaging):
    """
    The forward pass of 2D circular harmonic scattering
    Parameters
    ----------
    input_array: torch tensor
        input of size (batchsize, M, N)
    Returns
    -------
    output: tuple | torch tensor
        if max_order is 1 it returns a torch tensor with the
        first order scattering coefficients
        if max_order is 2 it returns a torch tensor with the
        first and second order scattering coefficients,
        stacked along the feature axis
    """
    rfft = backend.rfft
    ifft = backend.ifft
    cdgmm2d = backend.cdgmm2d
    modulus = backend.modulus
    modulus_rotation = backend.modulus_rotation

    # out_S_0, out_S_1, out_S_2 = [], [], []

    stack = backend.stack

    U_0_c = rfft(x) # fourier transform of the signal

    s_order_1, s_order_2 = [], []
    for l in range(L + 1):
        # orders of scattering coefficients
        s_order_1_l, s_order_2_l = [], []
        for j_1 in range(J + 1):
            U_1_m = None
            # if rotation_covariant:
            #     for m in range(len(filters[l][j_1])):
            #         U_1_c = cdgmm2d(U_0_c, filters[l][j_1][m])
            #         U_1_c = ifft(U_1_c)
            #         U_1_m = modulus_rotation(U_1_c, U_1_m)
            # else:
            U_1_c = cdgmm2d(U_0_c, filters[l][j_1]) # compute the product of the filter and signal in fourier space
            U_1_c = ifft(U_1_c) #inverse fouurier transform
            U_1_m = modulus(U_1_c) # modulus non-linearity after scattering, in time domain

            S_1_l = averaging(U_1_m) # average in time domain / low-pass filter to get scattering coeff
            s_order_1_l.append(S_1_l)
            # out_S_1.append({'coef': S_1_l})
            # 2nd order coeffs
            if max_order > 1:
                U_1_c = rfft(U_1_m) # starting point is the modulus of whatever 1st order coeff we've just "processed"
                for j_2 in range(j_1 + 1, J + 1):
                    U_2_m = None # Var to store the coeff
                    # if rotation_covariant:
                    #     for m in range(len(filters[l][j_2])):
                    #         U_2_c = cdgmm2d(U_1_c, filters[l][j_2][m])
                    #         U_2_c = ifft(U_2_c)
                    #         U_2_m = modulus_rotation(U_2_c, U_2_m)
                    # else:
                    U_2_c = cdgmm2d(U_1_c, filters[l][j_2]) # compute product of filter and signal, infourier space as before
                    U_2_c = ifft(U_2_c) # same, inverse fourier to obtain transformed signal in time domain
                    U_2_m = modulus(U_2_c) # modulus of the signal in time-domain
                    S_2_l = averaging(U_2_m) # temporal avg for the actuall coeff
                    # out_S_2.append({'coef': S_2_l})
                    s_order_2_l.append(S_2_l)


        s_order_1.append(s_order_1_l)

        if max_order == 2:
            s_order_2.append(s_order_2_l)

    # Stack the orders (along the j axis) if needed.
    S = s_order_1
    if max_order == 2:
        S = [x + y for x, y in zip(S, s_order_2)]

    # Invert (ell, m × j) ordering to (m × j, ell).
    S = [x for y in zip(*S) for x in y]


    # out_S = []
    # out_S.extend(out_S_0)
    # out_S.extend(out_S_1)
    # out_S.extend(out_S_2)

    
    S = stack(S)

    # warnings.warn(str(S.shape)) 
    # out_S = stack((s_order_1, s_order_2), dim=-1)
    # S = stack(S)

    return S 


def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array'):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft    
    cdgmm = backend.cdgmm
    stack = backend.stack

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []

    U_r = pad(x)

    U_0_c = rfft(U_r)

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi['levels'][0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    S_0 = irfft(U_1_c)
    S_0 = unpad(S_0)

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': ()})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1]['levels'][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)
        U_1_c = modulus(U_1_c)
        U_1_c = rfft(U_1_c)

        # Second low pass filter
        S_1_c = cdgmm(U_1_c, phi['levels'][j1])
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

        S_1_r = irfft(S_1_c)
        S_1_r = unpad(S_1_r)

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'n': (n1,),
                        'theta': (theta1,)})

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1:
                continue

            U_2_c = cdgmm(U_1_c, psi[n2]['levels'][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = ifft(U_2_c)
            U_2_c = modulus(U_2_c)
            U_2_c = rfft(U_2_c)

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi['levels'][j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            S_2_r = irfft(S_2_c)
            S_2_r = unpad(S_2_r)

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'n': (n1, n2),
                            'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        out_S = stack([x['coef'] for x in out_S])

    return out_S


__all__ = ['scattering2d', 'harmonic_scattering2d']
