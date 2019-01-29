import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal

# matplotlib.use('Agg')
# matplotlib.pyplot.ioff()
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange
from scipy.signal import butter, lfilter, freqz


def plot_frequency(data):
    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)
    data = np.sin(2 * np.pi * f * x / Fs)
    plt.plot(x, data)
    signal_spectrum = (np.fft.fft(data))
    time_step = 1 / 30
    freqs = (np.fft.fftfreq(signal_spectrum.size, d=time_step))
    plt.plot(freqs, np.abs(signal_spectrum))


def plot_power(x_k):
    ps = np.abs(x_k) ** 2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(x_k.size, time_step)
    idx = np.argsort(freqs)
    plot(freqs[idx], ps[idx])


def plot_spectrum(y, fs, extitle=None, path_to_save=''):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y)  # length of the signal
    cut = np.floor(n / 2).astype(int)
    Y = abs(fft(y))  # fft computing and normalization
    Y = Y[0:cut] / cut
    if extitle is None:
        extitle = ''

    title('Frequency Spectrum of the signal  %s' % (str(extitle)))
    dk = 1 / n
    freq = linspace(0, 0.5 - dk, cut) * fs
    plt.clf()
    plt.plot(freq, Y, 'r')  # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')
    axes = plt.gca()
    axes.set_ylim([0, Y.max()])
    axes.set_xlim([0, 2000])
    plt.show()
    # plt.savefig('%s/Frequency Spectrum of the signal  %s' % (path_to_save, str(extitle)) + '.png', bbox_inches='tight')


def plot_time(y, fs, extitle=None, path_to_save=''):
    n = len(y)  # length of the signal
    T = n / fs
    t = np.linspace(0, T, n, endpoint=False)
    if extitle is None:
        extitle = ''

    title('Time Domain of the signal  %s' % (str(extitle)))
    plt.clf()
    plt.plot(t, y)
    xlabel('Time')
    ylabel('Amplitude')
    plt.show()
    # plt.savefig('%s/Time Domain of the signal  %s' % (path_to_save, str(extitle)) + '.png', bbox_inches='tight')


def plot_spectrogram(x_t, fs, xtitle=None, path_to_save=None):
    # plt.clf()
    # plt.specgram(x_t, NFFT=len(x_t), noverlap=0, Fs=fs, window=matplotlib.mlab.window_none)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # # plt.title(title)
    # plt.show()

    frequencies, times, spectrogram = signal.spectrogram(x_t, fs)
    plt.clf()
    title('Spectrogram of the signal  %s' % (str(xtitle)))
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # plt.savefig('%s/Spectrogram of the signal  %s' % (path_to_save, str(xtitle)) + '.png', bbox_inches='tight')
    pass


def plot_psd(x_t, fs, xtitle, path_to_save):
    f, Pxx_den = signal.periodogram(x_t, fs)
    plt.clf()
    plt.semilogy(f, Pxx_den)
    title('PSD of the signal  %s' % (str(xtitle)))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    pass


def plot_time_freq(x_t, fs, title='', path_to_save=None):
    return
    x_t = remove_dc(x_t, 100, 1400, fs)
    x_t = normalize(x_t)

    plot_time(x_t, fs, title, path_to_save)
    plot_spectrum(x_t, fs, title, path_to_save)
    plot_spectrogram(x_t, fs, title, path_to_save)
    plot_psd(x_t, fs, title, path_to_save)


def normalize(x_m, min=-1, max=1):
    # x_m -= np.mean(x_m)
    min = np.min(x_m)
    max = np.max(x_m)

    normalized_data = np.array([2 * (x - min) / (max - min) - 1 for i, x in enumerate(x_m)]).reshape(-1, )
    normalized_data -= np.mean(normalized_data)  # Remove DC Component
    return normalized_data


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def remove_dc(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
