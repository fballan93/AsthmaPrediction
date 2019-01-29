import numpy as np
from scipy import signal
from scipy import fftpack

from core_code.Feature_Extraction.helper_functions import plot_time_freq


class FeatureExtractor:
    audio_id = 0
    signal_length: int = 0
    fs_signal: int = 0

    N: int = 256  # frame size
    M: int = 64  # STEP: How much we shift the signal each iteration

    K_min: int = 4
    K_max: int = 38

    L_ase: int = 13
    L_Ti: int = 7

    Beta: int = 7

    input_signal: np.ndarray

    start_time: float = 0
    end_time: float = 0

    signal_index: int = 0  # The current signal index coming from outside

    number_of_samples_per_frame: int = 0
    number_of_frames: int = 0

    final_ASE_val: float = 0
    final_Ti_val: float = 0
    current_iteration = 0
    signal_cache_data = {
        'x_m_arr': [],
        'x_m_k_arr': [],
        'r_m_k_arr': [],
        'power_arr': [],
        'phs_x_m_k_arr': [],
        'energy_arr': [],
        'unpredict_spectrum_arr': [],
        'ASE_arr': [],
        'Ti_arr': [],
    }

    def __init__(self, input_signal, signal_length, fs_signal, start_time, end_time, signal_index, audio_id):
        self.setup_signal_data(end_time, fs_signal, input_signal, signal_index, signal_length, start_time, audio_id)
        self.setup_cache_data()

    # noinspection PyTypeChecker
    def setup_cache_data(self):
        self.signal_cache_data['x_m_arr'] = np.zeros(shape=(self.number_of_frames, self.N))
        number_of_selected_bins = self.K_max - self.K_min + 1
        self.signal_cache_data['x_m_k_arr'] = np.zeros(shape=(self.number_of_frames, self.N), dtype=np.complex_)
        self.signal_cache_data['r_m_k_arr'] = np.zeros(shape=(self.number_of_frames, number_of_selected_bins))
        self.signal_cache_data['power_arr'] = np.zeros(shape=(self.number_of_frames, number_of_selected_bins))
        self.signal_cache_data['phs_x_m_k_arr'] = np.zeros(shape=(self.number_of_frames, number_of_selected_bins))
        self.signal_cache_data['energy_arr'] = np.zeros(shape=(self.number_of_frames, 1))
        self.signal_cache_data['unpredict_spectrum_arr'] = np.zeros(
            shape=(self.number_of_frames, 1))
        self.signal_cache_data['ASE_arr'] = np.zeros(shape=(self.number_of_frames, 1))
        self.signal_cache_data['Ti_arr'] = np.zeros(shape=(self.number_of_frames, 1))

    def setup_signal_data(self, end_time, fs_signal, input_signal, signal_index, signal_length, start_time, audio_id):
        self.fs_signal = fs_signal
        self.start_time = start_time
        self.end_time = end_time
        self.input_signal = input_signal
        self.signal_index = signal_index
        self.signal_length = signal_length
        self.number_of_samples_per_frame = self.M
        self.number_of_frames = np.floor((self.signal_length - self.N) / self.number_of_samples_per_frame).astype(
            int)
        self.audio_id = audio_id

    def calculate_signal_features(self):
        for i in range(self.number_of_frames):
            self.current_iteration = i
            self.__calculate_spectrum_for_one_frame()
            self.__calculate_audio_spectral_envelope()
            self.__calculate_tonality_index()

        return self.get_final_features_index()

    def get_final_features_index(self):
        power = self.signal_cache_data['power_arr']
        energy = self.signal_cache_data['energy_arr']
        phase = self.signal_cache_data['phs_x_m_k_arr']
        ASE = self.signal_cache_data['ASE_arr']
        Ti = self.signal_cache_data['Ti_arr']
        self.final_ASE_val = self.signal_cache_data['ASE_arr'][-1, 0]
        self.final_Ti_val = self.signal_cache_data['Ti_arr'][-1, 0]
        return power, energy, phase, ASE, Ti

    def __calculate_spectrum_for_one_frame(self):
        self.__preproces_frame()
        self.__calculate_spectrum()
        self.__calculate_magnitude()
        self.__calculate_power()

    def __calculate_audio_spectral_envelope(self):
        current_iteration = self.current_iteration
        ASE_m_k_prev = self.__get_previous_ASE(current_iteration)
        ASE_m_k_normalize = self.__normalize_ASE(ASE_m_k_prev)
        fluct_ASE = self.__calculate_fluctuation_in_ASE(ASE_m_k_normalize)
        self.signal_cache_data['ASE_arr'][current_iteration, 0] = fluct_ASE

    def __calculate_fluctuation_in_ASE(self, ASE_m_k):
        fluct_ASE = 0
        k = 1
        for _ in range(self.K_min, self.K_max, 1):
            fluct_ASE += abs(ASE_m_k[k] - ASE_m_k[k - 1])
            k += 1
        fluct_ASE = round(fluct_ASE, 5)
        return fluct_ASE

    def __normalize_ASE(self, ASE_m_k):
        ASE_m_k /= self.L_ase
        min_ase = ASE_m_k.min()
        max_ase = ASE_m_k.max()
        ASE_m_k = (ASE_m_k - min_ase) / (max_ase - min_ase)
        return ASE_m_k

    def __get_previous_ASE(self, current_iteration):
        l = current_iteration - self.L_ase + 1
        if l < 0:
            l = 0
        ASE_m_k = np.zeros(shape=(self.K_max - self.K_min + 1))
        for j in range(l, current_iteration + 1, 1):
            if j == current_iteration + 1:
                break
            ASE_m_k += self.signal_cache_data['power_arr'][j, :]
        return ASE_m_k

    def __calculate_tonality_index(self):
        current_iteration = self.current_iteration
        self.__calculate_phase_of_signal()
        r_m_k_prev = self.__calculate_prev_magnitude(current_iteration)
        phs_x_m_k_prev = self.__calculate_prev_phase(current_iteration)
        self.__calculate_rate_of_change_of_spectrum(r_m_k_prev, phs_x_m_k_prev)
        self.__calculate_energy_of_signal()
        c_m_k_w_mean, e_mean = self.__calculate_prev_changes_energy()
        ratio = c_m_k_w_mean / e_mean
        Ti_m = np.log10(ratio)
        Ti_m = round(Ti_m, 5)  # Round it to most 5 decimates, e.g. 0.000001 => 0.0
        self.signal_cache_data['Ti_arr'][current_iteration, 0] = Ti_m

    def get_ration_between_spectrum_and_energy(self, c_m_k_w_mean, e_mean):
        em_invs = np.linalg.pinv(e_mean.reshape(-1, 1))  # Convert it from 1-D array to  matrix to get the inverse
        em_invs = em_invs.reshape(-1, )  # Return it back to 1-D array, so it is the same type of c_m_k_w_mean
        ratio = np.dot(c_m_k_w_mean,
                       em_invs)  # This is the equivalent of matrix division in matlab that output in scalar
        return ratio

    def __calculate_prev_changes_energy(self):
        current_iteration = self.current_iteration
        e_mean = 0
        c_m_k_w_mean = 0
        l = current_iteration - self.L_Ti + 1
        if (l < 0):
            l = 0
        for j in range(l, current_iteration + 1, 1):
            e_mean += self.signal_cache_data['energy_arr'][j, 0]
            c_m_k_w_mean += self.signal_cache_data['unpredict_spectrum_arr'][j, 0]

        e_mean /= self.L_Ti
        c_m_k_w_mean /= self.L_Ti
        return c_m_k_w_mean, e_mean

    def __calculate_phase_of_signal(self):
        current_iteration = self.current_iteration
        x_m_k = self.signal_cache_data['x_m_k_arr'][current_iteration, :]
        phs_x_m_k = np.angle(x_m_k[self.K_min - 1:self.K_max])
        self.signal_cache_data['phs_x_m_k_arr'][current_iteration, :] = phs_x_m_k

    def __calculate_energy_of_signal(self):
        current_iteration = self.current_iteration
        p_m_k = self.signal_cache_data['power_arr']
        e_m = np.sum(p_m_k)  # Energy of the signal
        self.signal_cache_data['energy_arr'][current_iteration, 0] = e_m

    def __calculate_rate_of_change_of_spectrum(self, r_m_k_prev, phs_x_m_k_prev):
        current_iteration = self.current_iteration
        phs_x_m_k = self.signal_cache_data['phs_x_m_k_arr'][current_iteration, :]
        r_m_k = self.signal_cache_data['r_m_k_arr'][current_iteration, :]
        A_m = np.multiply(r_m_k, np.cos(phs_x_m_k)) - np.multiply(r_m_k_prev, np.cos(phs_x_m_k_prev))
        B_m = np.multiply(r_m_k, np.sin(phs_x_m_k)) - np.multiply(r_m_k_prev, np.sin(phs_x_m_k_prev))
        c_m_k = np.sqrt(A_m ** 2 + B_m ** 2) / (r_m_k + abs(r_m_k_prev))  # Spectrum Un predictability
        p_m_k = self.signal_cache_data['power_arr'][current_iteration, :]
        c_m_k_w = np.sum(np.multiply(c_m_k, p_m_k))  #
        self.signal_cache_data['unpredict_spectrum_arr'][current_iteration, 0] = c_m_k_w

    def __calculate_prev_phase(self, current_iteration):
        phs_x_m_k_prev = 0
        if current_iteration > 0:
            phs_x_m_k_prev = 2 * self.signal_cache_data['phs_x_m_k_arr'][current_iteration - 1, :]
            if current_iteration != 1:
                phs_x_m_k_prev -= self.signal_cache_data['phs_x_m_k_arr'][current_iteration - 2, :]
        return phs_x_m_k_prev

    def __calculate_prev_magnitude(self, current_iteration):
        r_m_k_prev = 0
        if current_iteration > 0:
            r_m_k_prev = 2 * self.signal_cache_data['r_m_k_arr'][current_iteration - 1, :]
            if current_iteration != 1:
                r_m_k_prev -= self.signal_cache_data['r_m_k_arr'][current_iteration - 2, :]
        return r_m_k_prev

    def __preproces_frame(self):  # normalize the signal
        current_iteration = self.current_iteration
        start = current_iteration * self.M
        end = start + self.N
        x_m = self.input_signal[start:end]
        x_m -= x_m.mean()
        max_x_m = np.amax(x_m)
        if max_x_m != 0:
            x_m /= max_x_m

        self.signal_cache_data['x_m_arr'][current_iteration, :] = x_m

    def __calculate_spectrum(self):
        current_iteration = self.current_iteration
        x_m = self.__windowing_signal(current_iteration)
        # plot_time_freq(x_m, self.fs_signal,
        #                'Segment : %d' % self.current_iteration, )
        x_m_k = fftpack.fft(x_m, x_m.size)
        self.signal_cache_data['x_m_k_arr'][current_iteration, :] = x_m_k

    def __windowing_signal(self, current_iteration):
        x_m = self.signal_cache_data['x_m_arr'][current_iteration, :]
        window = signal.kaiser(x_m.size, self.Beta, False)
        x_m = np.multiply(x_m, window)
        return x_m

    def __calculate_power(self):
        current_iteration = self.current_iteration
        r_m_k = self.signal_cache_data['r_m_k_arr'][current_iteration, :]
        p_m_k = r_m_k ** 2
        self.signal_cache_data['power_arr'][current_iteration, :] = p_m_k

    def __calculate_magnitude(self):
        current_iteration = self.current_iteration
        x_m_k = self.signal_cache_data['x_m_k_arr'][current_iteration, :]
        r_m_k = np.abs(
            x_m_k[
            self.K_min - 1:self.K_max])  # -1 for each cut edge because indices start from zero not one,
        # K_max without -1 to include to K_max
        self.signal_cache_data['r_m_k_arr'][current_iteration, :] = r_m_k

    def get_cache_obj(self):
        return {
            'number_of_segments': self.number_of_frames,
            'frame_index': self.signal_index,
            'frame_length': self.input_signal.size,
            'k_min': self.K_min,
            'k_max': self.K_max,
            'selected_spectrum_frequency': self.signal_cache_data['x_m_k_arr'][self.K_min - 1:self.K_max],
            'ASE_values': self.signal_cache_data['ASE_arr'],
            'TI_values': self.signal_cache_data['Ti_arr'],
            'energy_values': self.signal_cache_data['energy_arr'],
            'spectral_unpredictability': self.signal_cache_data['unpredict_spectrum_arr'],
            'final_ASE': self.final_ASE_val,
            'final_TI': self.final_Ti_val,
            'audio_fk_id': self.audio_id,
        }
