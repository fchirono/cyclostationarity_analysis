# -*- coding: utf-8 -*-
"""
Python script containing functions to generate and analyse cyclostationary
data.

Contains mostly the same Python codes found at the IPython Notebook at
https://github.com/fchirono/cyclostationarity_analysis . However, this is where
I try new ideas and implementations, so this content is subject to change at
any time.


Author:
    Fabio Casagrande Hirono
    fchirono [at] gmail.com
    November 2022
"""


import numpy as np
rng = np.random.default_rng()

import scipy.signal as ss

import matplotlib.pyplot as plt
plt.close('all')


# %% cyclic periodogram

def calc_xspec_block(x, y, t, alpha=0):
    """
    Calculate cross-spectrum for one block of data over time samples 't',
    optionally including cyclic frequency shift 'alpha'.

    Parameters
    ----------
    x : (N,)-shaped array_like
        Numpy array containing one 'N'-samples-long block of data from signal
        'x'.

    y : (N,)-shaped array_like
        Numpy array containing one 'N'-samples-long block of data from signal
        'y'.

    t : (N,)-shaped array_like
        Numpy array containing one 'N'-samples-long block of time values, in
        seconds.

    alpha : float, optional
        Cyclic frequency 'alpha', in Hz. Default is 0.


    Returns
    -------
    Suu : (N,)-shaped array_like
        Numpy array containing 'N'-long auto-power spectrum of signal 'x',
        frequency-shifted by +'alpha'/2.

    Svv : (N,)-shaped array_like
        Numpy array containing 'N'-long auto-power spectrum of signal 'y',
        frequency-shifted by -'alpha'/2.

    Suv : (N,)-shaped array_like
        Numpy array containing 'N'-long cross-power spectrum of signal 'x'
        frequency-shifted by +'alpha'/2, and signal 'y' frequency-shifted
        by -'alpha'/2.

    """

    # applies frequency shift of +alpha/2
    u_block = x*np.exp(-1j*np.pi*alpha*t)

    # applies frequency shift of -alpha/2
    v_block = y*np.exp(+1j*np.pi*alpha*t)


    # take FFT of data blocks
    u_f = np.fft.fft(u_block)
    v_f = np.fft.fft(v_block)

    # calculates auto- and cross-power spectra
    Suu = (u_f * u_f.conj()).real
    Svv = (v_f * v_f.conj()).real
    Suv = (u_f * v_f.conj())

    return Suu, Svv, Suv


def calc_conj_xspec_block(x, y, t, alpha=0):
    """
    Calculate *conjugated* cross-spectrum for one block of data over time
    samples 't', optionally including cyclic frequency shift 'alpha'.

    Parameters
    ----------
    x : (N,)-shaped array_like
        Numpy array containing one 'N'-samples-long block of data from signal
        'x'.

    y : (N,)-shaped array_like
        Numpy array containing one 'N'-samples-long block of data from signal
        'y'.

    t : (N,)-shaped array_like
        Numpy array containing one 'N'-samples-long block of time values, in
        seconds.

    alpha : float, optional
        Cyclic frequency 'alpha', in Hz. Default is 0.


    Returns
    -------
    Suu : (N,)-shaped array_like
        Numpy array containing auto-power spectrum of signal 'x',
        frequency-shifted by +'alpha'/2.

    Svv : (N,)-shaped array_like
        Numpy array containing auto-power spectrum of signal 'y',
        frequency-reversed and frequency-shifted by +'alpha'/2.

    Suv : (N,)-shaped array_like
        Numpy array containing cross-power spectrum of signal 'x'
        frequency-shifted by +'alpha'/2, and signal 'y' frequency-reversed and
        frequency-shifted by +'alpha'/2.

    """

    # applies frequency shift of +alpha/2
    u_block = x*np.exp(-1j*np.pi*alpha*t)

    # applies frequency shift of -alpha/2
    v_block = y*np.exp(+1j*np.pi*alpha*t)


    # take FFT of data blocks
    u_f = np.fft.fft(u_block)
    v_f = np.fft.fft(v_block)

    # frequency-reverse v_f, so it changes from Y(f-alpha/2) to Y(-f+alpha/2)
    v_f = v_f[::-1]

    # calculates auto- and cross-power spectra
    Suu = (u_f * u_f.conj()).real
    Svv = (v_f * v_f.conj()).real
    Suv = (u_f * v_f.conj())

    return Suu, Svv, Suv


def cyclic_periodogram(x, y, alpha_vec, Ndft, fs, mode='non-conj'):
    """
    Calculates cyclic spectral density and cyclic spectral coherence using
    periodogram (time-smoothing) method.

    Parameters
    ----------
    x : (Nt,)-shaped array_like
        Numpy array containing 'Nt' time-domain samples of signal 'x'.

    y : (Nt,)-shaped array_like
        Numpy array containing 'Nt' time-domain samples of signal 'y'.

    alpha_vec : (N_alpha,)-shaped array_like
        Numpy array containing 'N_alpha' values of cyclic frequency 'alpha',
        in Hz.

    Ndft : int
        Number of points used in calculating the Discrete Fourier Transform.

    fs : float
        Sampling frequency, in Hz.

    mode : {'non-conj', 'conj'}, optional
        String defining whether output is non-conjugate CSD and coherence,
        or conjugate CSD and coherence. Default is 'non-conj'.


    Returns
    -------
    Sxy_avg : (N_alpha, Ndft)-shaped array_like
        Numpy array containing the cyclic spectral density 'Sxy', evaluated at
        'Ndft' frequencies for each cyclic frequency 'alpha'.

    cohere : (N_alpha, Ndft)-shaped array_like
        Numpy array containing the cyclic coherence 'cohere', evaluated at
        'Ndft' frequencies for each cyclic frequency 'alpha'.
    """

    N_alpha = alpha_vec.shape[0]

    Nt = x.shape[0]
    N_blocks = Nt//Ndft


    df = fs/Ndft
    freq = np.linspace(0, fs-df, Ndft)-fs/2

    # Spectral Correlation Density (SCD) function
    Sxx = np.zeros((N_blocks, N_alpha, Ndft))
    Syy = np.zeros((N_blocks, N_alpha, Ndft))
    Sxy = np.zeros((N_blocks, N_alpha, Ndft), dtype='complex')

    # -------------------------------------------------------------------------
    if mode == 'non-conj':
        for n in range(N_blocks):

            n_start = n*Ndft
            t_block = np.linspace(n_start/fs, (n_start+Ndft)/fs, Ndft)

            x_block = x[n_start : n_start+Ndft]
            y_block = y[n_start : n_start+Ndft]

            # calculate non-conjugate spectra for alpha values in 'alpha_vec'
            for a, alpha in enumerate(alpha_vec):
                Sxx[n, a, :], Syy[n, a, :], Sxy[n, a, :] = calc_xspec_block(x_block, y_block,
                                                                            t_block, alpha)

    # -------------------------------------------------------------------------
    elif mode == 'conj':
        for n in range(N_blocks):

            n_start = n*Ndft
            t_block = np.linspace(n_start/fs, (n_start+Ndft)/fs, Ndft)

            x_block = x[n_start : n_start+Ndft]
            y_block = y[n_start : n_start+Ndft]

            # calculate conjugate spectra for alpha values in 'alpha_vec'
            for a, alpha in enumerate(alpha_vec):
                Sxx[n, a, :], Syy[n, a, :], Sxy[n, a, :] = calc_conj_xspec_block(x_block, y_block,
                                                                                 t_block, alpha)
    # -------------------------------------------------------------------------

    Sxx *= 1/(Ndft*fs)
    Syy *= 1/(Ndft*fs)
    Sxy *= 1/(Ndft*fs)

    # apply FFT shift
    Sxx = np.fft.fftshift(Sxx, axes=(-1))
    Syy = np.fft.fftshift(Syy, axes=(-1))
    Sxy = np.fft.fftshift(Sxy, axes=(-1))

    # average over blocks
    Sxx_avg = Sxx.sum(axis=0)/N_blocks
    Syy_avg = Syy.sum(axis=0)/N_blocks
    Sxy_avg = Sxy.sum(axis=0)/N_blocks

    cohere = Sxy_avg/np.sqrt(Sxx_avg*Syy_avg)

    # replace entries outside the principal domain in freq-alpha plane by NaNs
    # (i.e. diamond shape given by |f| > (fs-|alpha|)/2 )
    for a, alpha in enumerate(alpha_vec):
        freqs_outside = (np.abs(freq) > (fs - np.abs(alpha))/2)
        Sxy_avg[a, freqs_outside] = np.nan
        cohere[a, freqs_outside] = np.nan

    return Sxy_avg, cohere


# %% functions to generate test signals

def create_rect_bpsk(T_bits, num_bits, fc, signal_power_dB):
    """
    Generate a rectangular-pulse binary phase-shift keyed signal, as described in
    https://cyclostationary.blog/2015/09/28/creating-a-simple-cs-signal-rectangular-pulse-bpsk/

    Parameters
    ----------
    T_bits : int
        Number of samples per bit. Note that '1/T_bit' is the bit rate.

    num_bits : int
        Desired number of bits in the signal.

    fc : float
        Desired carrier frequency, in normalised units.

    signal_power_dB : float
        Desired signal power, in decibels.

    Returns
    -------
    x_t : (N_t,)-shaped array_like
        Numpy array containing the time-domain, frequency-shifted signal samples.

    """

    N_samples = num_bits*T_bits

    # Create bit sequence - 0s and 1s
    bit_seq = rng.integers(0, 2, num_bits)

    # Create symbol sequence from bit sequence (-1s and +1s)
    sym_seq = 2*bit_seq - 1

    # generate symbol sequence by intercalating each bit with (T_bits-1) zeros
    zero_mat = np.zeros((T_bits-1, num_bits))
    sym_seq = np.concatenate((sym_seq[np.newaxis, :], zero_mat), axis=0)
    sym_seq = np.reshape(sym_seq, (N_samples,), order='F')

    # Create rectangular pulse function
    p_t = np.ones((T_bits,))

    # Convolve bit sequence with pulse function to obtain rectangular-pulse
    # BPSK signal
    s_t = ss.lfilter(p_t, [1], sym_seq)

    # Apply the carrier frequency.
    e_vec = np.exp(1j*2*np.pi*fc*np.arange(N_samples))
    x_t = s_t * e_vec

    # normalise signal power to 'signal_power_dB'
    signal_power = 10**(signal_power_dB/10)
    x_t *= np.sqrt(signal_power/np.var(x_t))

    return x_t


def create_lowpassmod_cos(N_samples, N_filter, fc_filter, f_cos, fs,
                          signal_power_dB):
    """
    Generate bandlimited Gaussian noise, modulating a unitary-amplitude cosine
    wave at 'f_cos' Hz. The Gaussian noise is lowpassed at 'fc_filter' Hz
    using a Butterworth filter of order 'N_filter'.

    Parameters
    ----------
    N_samples : int
        Number of time-domain samples in the output signal.

    N_filter : int
        Filter order used to generate lowpassed Gaussian random samples.

    fc_filter : float
        Cutoff frequency, in Hz, used to generate lowpassed Gaussian random
        samples.

    f_cos : float
        Frequency of the cosine wave, in Hz.

    fs : float
        Sampling frequency, in Hz.

    signal_power_dB : float
        Desired output signal power, in decibels.

    Returns
    -------
    x : (N_samples,)-shaped array_like
        Numpy array containing the time-domain signal samples.

    """

    # define lowpass Butterworth filter
    butter_sos = ss.butter(N_filter, fc_filter, output='sos', fs=fs)

    # create band-limited Gaussian white noise
    xn = rng.normal(loc=0, scale=1, size=N_samples)
    x_lpn = ss.sosfilt(butter_sos, xn)

    # generate cosine wave at 'fc' Hz
    t = np.linspace(0, (N_samples-1)/fs, N_samples)
    xc = np.cos(2*np.pi*f_cos*t)

    x = x_lpn*xc

    # normalise signal power to 'signal_power_dB'
    signal_power = 10**(signal_power_dB/10)
    x *= np.sqrt(signal_power/np.var(x))

    return x


def create_noise(x_t, SNR_dB):
    """
    Returns a real- or complex-valued Gaussian random noise signal, which
    yields a given Signal-to-Noise Ratio (SNR) in decibels when added to the
    input signal 'x_t'.

    Noise samples follow the same data type (real or complex) as input
    signal 'x_t'.

    Parameters
    ----------
    x_t : (N_t,)-shaped array_like
        Numpy array containing the time-domain signal samples.

    SNR_dB : float
        Desired signal-to-noise ratio, in decibels.

    Returns
    -------
    noise_t : (N_t,)-shaped array_like
        Numpy array containing the time-domain noise samples.

    """

    N_t = x_t.shape[0]
    signal_power = np.var(x_t)

    # generate real-valued Gaussian random samples
    noise_t = rng.normal(0, 1, N_t)

    # if 'x' is complex, adds complex-valued Gaussian random samples to
    # 'noise_t'
    if np.iscomplexobj(x_t):
        noise_t = noise_t + 1j*rng.normal(0, 1, N_t)

    noise_power = np.var(noise_t)

    SNR_lin = 10**(SNR_dB/10)

    desired_noise_power = signal_power/SNR_lin

    # renormalise noise power to yield desired SNR
    noise_t *= np.sqrt(desired_noise_power/noise_power)

    return noise_t


# %% comment/uncomment desired option

signal = 'bpsk'
# signal = 'lowpassmod_cos'


# sampling frequency
fs = 1

if signal == 'bpsk':
    # -----------------------------------------------------------------------------
    # rect BPSK signal

    T_bits = 10                 # Number of samples per bit (1/T_bit is the bit rate)
    num_bits = 32768            # Desired number of bits in generated signal
    fc = 0.05                   # Desired carrier frequency (normalized units)

    signal_power_dB = 0.0       # Signal power in decibels
    noise_power_dB = -10.0      # Noise spectral density (average noise power)

    T = (num_bits*T_bits)/fs
    y = create_rect_bpsk(T_bits, num_bits, fc, signal_power_dB)
    # -----------------------------------------------------------------------------

elif signal == 'lowpassmod_cos':
    # -----------------------------------------------------------------------------
    # bandlimited Gaussian white noise modulating a cosine wave
    N_samples = 327680

    # lowpassed white noise parameters
    N_filter = 6                # lowpass filter order (Butterworth)
    fc_filter = 0.05*fs         # lowpass filter cutoff freq
    f_cos = 0.15*fs             # cosine wave frequency

    signal_power_dB = 0.0       # Signal power in decibels
    noise_power_dB = -10.0      # Noise spectral density (average noise power)

    T = N_samples/fs
    y = create_lowpassmod_cos(N_samples, N_filter, fc_filter, f_cos, fs,
                              signal_power_dB)
    # -----------------------------------------------------------------------------



# ********************************************************************************
# Creates vector of noise samples
n = create_noise(y, SNR_dB=signal_power_dB-noise_power_dB)

# Verifies SNR calculation
Py = np.var(y)
Pn = np.var(n)
print('Desired SNR = {:.1f} dB'.format(signal_power_dB-noise_power_dB))
print('Resulting SNR = {:.1f} dB'.format(10*np.log10(Py/Pn)))

# adds noise to time-domain samples
noisy_y = y+n

# -----------------------------------------------------------------------------
# plot original vs. noisy time-domain signal
N_plot = 100
t = np.linspace(0, (N_plot-1), N_plot)

plt.figure(figsize=(9, 6))
plt.subplot(211)
plt.plot(t, y[:N_plot].real, label='Re(y)')
plt.plot(t, y[:N_plot].imag, '--', label='Im(y)')
plt.title("Original vs. noisy time-domain signal", fontsize=18)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(loc='lower right', fontsize=12)

plt.subplot(212)
plt.plot(t, noisy_y[:N_plot].real, label='Re(noisy y)')
plt.plot(t, noisy_y[:N_plot].imag, '--', label='Im(noisy y)')
plt.xlabel('Samples', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(loc='lower right', fontsize=12)


# %% frequency analysis variables

# Number of frequencies to use in PSD estimate
N_psd = 128

df = fs/N_psd
freq_vec = np.linspace(0, fs-df, N_psd) - fs/2

freq_res = fs/(N_psd)       # frequency resolution
TB = T*freq_res             # time-bandwidth product

print('Time-bandwidth product = {:.2f}'.format(TB))

# %% run cyclic periodogram over range of alphas

# range of alphas to test
N_alpha = 11
alpha_vec = np.linspace(0., 1., N_alpha)*fs

Syy, rho_y = cyclic_periodogram(noisy_y, noisy_y, alpha_vec, N_psd, fs)

# %% plot spectral correlation density and spectral coherence in 2D


lines = ['-', '--', '-.', ':']
linewidths = [2.5, 1, 0.5]
N_lines = len(lines)

max_Syy_dB = 10*np.log10(np.nanmax(np.abs(Syy)))

# ****************************************************************************
# plot spectral correlation density

plt.figure(figsize=(9, 6))
for a in range(N_alpha//2+1):
    plt.plot(freq_vec/fs,
             10*np.log10(np.abs(Syy[a, :])),
             linestyle=lines[a%N_lines], linewidth=linewidths[a//N_lines],
             label=r'$\alpha/f_s$={:.2f}'.format(alpha_vec[a]/fs))
plt.legend(loc='upper left', fontsize=12)

plt.xlabel(r'freq/$f_s$', fontsize=12)
plt.xlim([-0.5, 0.5])

plt.ylabel('Magnitude [dB]', fontsize=12)
plt.ylim([max_Syy_dB-25, max_Syy_dB+5])

plt.title('Spectral correlation density', fontsize=15)
plt.grid()
plt.tight_layout()

# ****************************************************************************
# plot spectral coherence
plt.figure(figsize=(9, 6))
for a in range(N_alpha//2+1):
    plt.plot(freq_vec/fs,
             np.abs(rho_y[a, :]),
             linestyle=lines[a%N_lines], linewidth=linewidths[a//N_lines],
             label=r'$\alpha/f_s$={:.2f}'.format(alpha_vec[a]/fs))
plt.legend(loc='upper left', fontsize=12)

plt.xlabel(r'freq/$f_s$', fontsize=12)
plt.xlim([-0.5, 0.5])

plt.ylabel('Magnitude [Linear]', fontsize=12)
plt.ylim([0., 1.2])

plt.title('Spectral coherence function', fontsize=15)
plt.grid()
plt.tight_layout()


# %% Old code with wireframe plots

# plt.figure(figsize=(12, 8))
# ax = plt.subplot(111, projection='3d')
# ax.plot_wireframe(alpha_vec[:, np.newaxis]/fs, freq_vec/fs,
#                   np.abs(Syy),
#                   rstride=1, cstride=0)

# ax.set_title('Spectral correlation density', fontsize=15)
# ax.set_xlabel(r'$\alpha/f_s$', fontsize=12)
# ax.set_ylabel(r'freq/$f_s$', fontsize=12)
# ax.set_zlabel('Magnitude [dB]', fontsize=12)
# plt.tight_layout()


# plt.figure(figsize=(8, 6))
# ax = plt.subplot(111, projection='3d')
# ax.plot_wireframe(alpha_vec[:, np.newaxis]/fs, freq_vec/fs,
#                   np.abs(rho_y),
#                   rstride=1, cstride=0)

# ax.set_title('Spectral coherence function')
# ax.set_xlabel('alpha/fs')
# ax.set_ylabel('freq/fs')



# %% plot spectral correlation density and spectral coherence in 3D

from matplotlib.collections import PolyCollection

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.

    Taken from
    https://matplotlib.org/stable/gallery/mplot3d/polys3d.html#sphx-glr-gallery-mplot3d-polys3d-py
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


# create boolean mask for frequencies inside principal domain in f-alpha plane
freqs_inside = np.zeros((N_alpha, N_psd), dtype='bool')
for a, alpha in enumerate(alpha_vec):
    freqs_inside[a, :] = (np.abs(freq_vec) <= (fs - np.abs(alpha))/2)


# ****************************************************************************
# plot spectral correlation density
fig1 = plt.figure(figsize=(9, 6))
ax1 = fig1.add_subplot(projection='3d')

verts = [polygon_under_graph(freq_vec[freqs_inside[a]]/fs,
                             np.abs(Syy[a, freqs_inside[a]]))
         for a in range(N_alpha)]

# create list of facecolors - incompatible with Matplotlib v3.2.2 currently available
# in Google Colab :(
# facecolors = plt.colormaps['inferno'](np.linspace(0, 1, len(verts)+2))[:len(verts)]

import matplotlib
cmap = matplotlib.cm.get_cmap('inferno')
facecolors = cmap(np.linspace(0, 1, len(verts)+2))[:len(verts)]

transparency = 0.8

poly1 = PolyCollection(verts[::-1], facecolors=facecolors, alpha=transparency,
                       edgecolors='k', linewidths=0.75)
ax1.add_collection3d(poly1, zs=alpha_vec[::-1], zdir='y')

ax1.set(xlim = (-0.5, 0.5),
        ylim = (alpha_vec[-1], alpha_vec[0]),
        zlim = (0., np.nanmax(np.abs(Syy))))

ax1.set_xlabel(r'freq/$f_s$', fontsize=12)
ax1.set_xticks(np.linspace(-0.5, 0.5, 5),
              labels=['{:.2f}'.format(f) for f in np.linspace(-0.5, 0.5, 5)])

ax1.set_ylabel(r'$\alpha/f_s$', fontsize=12)
ax1.set_yticks(alpha_vec[::2])
ax1.set_yticklabels(['{:.2f}'.format(a) for a in alpha_vec[::2]])

ax1.set_zlabel('Magnitude [Linear]', fontsize=12)

# set aspect ratio - incompatible with Matplotlib v3.2.2 currently available
# in Google Colab :(
#ax1.set_box_aspect((1, 1, 0.5))

ax1.set_title('Spectral correlation density', fontsize=15)


# ****************************************************************************
# plot coherence
fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.add_subplot(projection='3d')

verts2 = [polygon_under_graph(freq_vec[freqs_inside[a]]/fs,
                              np.abs(rho_y[a, freqs_inside[a]]))
          for a in range(N_alpha)]


poly2 = PolyCollection(verts2[::-1], facecolors=facecolors, alpha=transparency,
                       edgecolors='k', linewidths=0.75)
ax2.add_collection3d(poly2, zs=alpha_vec[::-1], zdir='y')

ax2.set(xlim = (-0.5, 0.5),
        ylim = (alpha_vec[-1], alpha_vec[0]),
        zlim = (0., 1.))

ax2.set_xlabel(r'freq/$f_s$', fontsize=12)
ax2.set_xticks(np.linspace(-0.5, 0.5, 5),
               labels=['{:.2f}'.format(f) for f in np.linspace(-0.5, 0.5, 5)])

ax2.set_ylabel(r'$\alpha/f_s$', fontsize=12)
ax2.set_yticks(alpha_vec[::2])
ax2.set_yticklabels(['{:.2f}'.format(a) for a in alpha_vec[::2]])

ax2.set_zlabel('Magnitude [Linear]', fontsize=12)

# set aspect ratio - incompatible with Matplotlib v3.2.2 currently available
# in Google Colab :(
#ax1.set_box_aspect((1, 1, 0.5))

ax2.set_title('Spectral coherence', fontsize=15)

# set viewing angle to avoid bug in Matplotlib where polygons are rendered in
# wrong order
ax2.view_init(elev=30, azim=-45)

# ****************************************************************************


# %% plot conjugate spectral correlation density and spectral coherence in 3D

# run conjugate cyclic periodogram over range of alphas
# --> range of alphas is different for conjugate functions!
N_alpha_c = 19
alpha_vec_c = 2*fc + np.linspace(-N_alpha_c//2, N_alpha_c//2-1, N_alpha_c)*(1/T_bits)

Syy_c, rho_y_c = cyclic_periodogram(y, y, alpha_vec_c, N_psd, fs)

# create boolean mask for frequencies inside principal domain in f-alpha plane
freqs_inside_c = np.zeros((N_alpha_c, N_psd), dtype='bool')
for a, alpha in enumerate(alpha_vec_c):
    freqs_inside_c[a, :] = (np.abs(freq_vec) <= (fs - np.abs(alpha))/2)


# ****************************************************************************
# plot conjugate spectral correlation density
fig3 = plt.figure(figsize=(9, 6))
ax3 = fig3.add_subplot(projection='3d')

verts_c = [polygon_under_graph(freq_vec[freqs_inside_c[a]]/fs,
                               np.abs(Syy_c[a, freqs_inside_c[a]]))
            for a in range(N_alpha_c)]

# create list of facecolors - incompatible with Matplotlib v3.2.2 currently available
# in Google Colab :(
# facecolors = plt.colormaps['inferno'](np.linspace(0, 1, len(verts)+2))[:len(verts)]

import matplotlib
cmap = matplotlib.cm.get_cmap('inferno')
facecolors_c = cmap(np.linspace(0, 1, len(verts_c)+2))[:len(verts_c)]

transparency = 0.8

poly3 = PolyCollection(verts_c[::-1],
                        facecolors=facecolors_c, alpha=transparency,
                        edgecolors='k', linewidths=0.75)
ax3.add_collection3d(poly3, zs=alpha_vec_c[::-1], zdir='y')

ax3.set(xlim = (-0.5, 0.5),
        ylim = (alpha_vec[-1], alpha_vec[0]),
        zlim = (0., np.nanmax(np.abs(Syy_c))))

ax3.set_xlabel(r'freq/$f_s$', fontsize=12)
ax3.set_xticks(np.linspace(-0.5, 0.5, 5),
              labels=['{:.2f}'.format(f) for f in np.linspace(-0.5, 0.5, 5)])

ax3.set_ylabel(r'$\alpha/f_s$', fontsize=12)
ax3.set_yticks(alpha_vec_c[::2])
ax3.set_yticklabels(['{:.2f}'.format(a) for a in alpha_vec_c[::2]])

ax3.set_zlabel('Magnitude [Linear]', fontsize=12)

# set aspect ratio - incompatible with Matplotlib v3.2.2 currently available
# in Google Colab :(
#ax1.set_box_aspect((1, 1, 0.5))

ax3.set_title('Conjugate Spectral correlation density', fontsize=15)


# ****************************************************************************
# plot conjugate coherence
fig4 = plt.figure(figsize=(9, 6))
ax4 = fig4.add_subplot(projection='3d')

verts4 = [polygon_under_graph(freq_vec[freqs_inside_c[a]]/fs,
                              np.abs(rho_y_c[a, freqs_inside_c[a]]))
          for a in range(N_alpha_c)]


poly4 = PolyCollection(verts4[::-1],
                        facecolors=facecolors_c, alpha=transparency,
                        edgecolors='k', linewidths=0.75)
ax4.add_collection3d(poly4, zs=alpha_vec_c[::-1], zdir='y')

ax4.set(xlim = (-0.5, 0.5),
        ylim = (alpha_vec_c[-1], alpha_vec_c[0]),
        zlim = (0., 1.))

ax4.set_xlabel(r'freq/$f_s$', fontsize=12)
ax4.set_xticks(np.linspace(-0.5, 0.5, 5),
                labels=['{:.2f}'.format(f) for f in np.linspace(-0.5, 0.5, 5)])

ax4.set_ylabel(r'$\alpha/f_s$', fontsize=12)
ax4.set_yticks(alpha_vec_c[::2])
ax4.set_yticklabels(['{:.2f}'.format(a) for a in alpha_vec_c[::2]])

ax4.set_zlabel('Magnitude [Linear]', fontsize=12)

# set aspect ratio - incompatible with Matplotlib v3.2.2 currently available
# in Google Colab :(
#ax1.set_box_aspect((1, 1, 0.5))

ax2.set_title('Conjugate Spectral coherence', fontsize=15)

# # ****************************************************************************



# %% read Chad's mat file

from scipy.io import loadmat

matfile = loadmat('theory_and_meas_functions_new.mat')

# ****************************************************************************
# plot PSD
psd_fsm = np.squeeze(matfile['psd_fsm'])
freq_psd = np.squeeze(matfile['f_meas_psd'])

psd_theory = np.squeeze(matfile['psd_theory'])
freq_theory_psd = np.squeeze(matfile['f_theory_psd'])


plt.figure()
plt.plot(freq_psd, psd_fsm, label='FSM')
plt.plot(freq_theory_psd, psd_theory, '--', label='Theory')
plt.plot(freq_vec/fs,
          10*np.log10(np.abs(Syy[0, :])), ':', label='Python')
plt.legend()
plt.title('PSD')
plt.grid()
plt.ylim([-30, 15])


# ****************************************************************************
# plot Spectral Correlation Function (non-conjugate)
scf_nc_fsm1 = np.squeeze(matfile['scf_nc_fsm_1'])
freq_meas_nc = np.squeeze(matfile['f_meas_nc'])

scf_nc_theory1 = np.squeeze(matfile['scf_nc_theory_1'])
freq_theory_nc = np.squeeze(matfile['f_theory_nc'])

plt.figure()
plt.plot(freq_meas_nc, scf_nc_fsm1, label='FSM')
plt.plot(freq_theory_nc, scf_nc_theory1, '--', label='Theory')
plt.plot(freq_vec/fs,
          10*np.log10(np.abs(Syy[1, :])), ':', label='Python')
plt.legend()
plt.title('SCF NC alpha=0.1')
plt.grid()
plt.ylim([-30, 15])

# ****************************************************************************
# plot SCF (conjugate)

scf_c_fsm1 = np.squeeze(matfile['scf_c_fsm_p1'])
freq_meas_c = np.squeeze(matfile['f_meas_c'])

scf_c_theory1 = np.squeeze(matfile['scf_c_theory_p1'])
freq_theory_c = np.squeeze(matfile['f_theory_c'])

plt.figure()
plt.plot(freq_meas_c, scf_c_fsm1, label='FSM')
plt.plot(freq_theory_c, scf_c_theory1, '--', label='Theory')
plt.plot(freq_vec/fs,
          10*np.log10(np.abs(Syy_c[11, :])), ':', label='Python')
plt.legend()
plt.title('SCF C alpha=+0.1')
plt.grid()
plt.ylim([-30, 15])
