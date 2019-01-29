import string

import numpy as np

from pydub import AudioSegment
from pydub.utils import which
import os
import soundfile as sf
import resampy

current_file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(current_file_path + '/../../../')

from core_code.Feature_Extraction.FeatureExtractor import FeatureExtractor
from core_code.Feature_Extraction.helper_functions import plot_time_freq, remove_dc

AudioSegment.converter = which("ffmpeg")


class AsthmaAnalyzer(object):
    audio_file_path: string = ''
    fs_constant_signal: int = 8 * 10 ** 3
    audio_id = 0
    audio_duration: float = 0
    fs_input_signal: int = 0
    signal_length: int = 1024
    number_of_signals_in_file: np.int64 = 0
    input_signal: np.ndarray
    resampled_input_signal: np.ndarray
    extracted_features = []
    file_name: string = ''  # Saving paths variables
    general_path_to_save = os.path.abspath(root_dir + '/experiments')
    result_path = ''
    screen_shot_path = ''
    signal_cache_data = []
    final_ASE: float
    final_Ti: float

    def __init__(self, audio_file_path: string, file_name: string, audio_id=0):
        self.cut_time = 5.12
        self.file_name = file_name
        self.audio_file_path = audio_file_path
        self.audio_id = audio_id
        self.__extract_data()

    def __extract_data(self):
        # fs, data = wavfile.read(self.audio_file_path)
        data, fs = sf.read(self.audio_file_path)
        # data = np.sin(data)
        if data.ndim > 1:
            data = data[:, 0]  # Taking the first channel only
        # data = data.astype(np.float64)
        # data = (1 - (-1)) * (data - data.min(0)) / data.ptp(0) - 1  # normalize the data between -1 and
        # # data = np.random.rand(301) - 0.5

        self.audio_duration = data[:].size / fs
        self.fs_input_signal = fs
        self.input_signal = data[:]
        # self.input_signal = remove_dc(self.input_signal, 100, 1400, self.fs_input_signal)

        self.create_required_directories()
        plot_time_freq(self.input_signal, self.fs_input_signal, 'Pure Signal', self.screen_shot_path)

    def get_signal_features(self):
        # 1. Pre process the signal
        self.__pre_process_signal()
        # 2. Loop through all signals
        # sd.play(self.input_signal, self.fs_input_signal)
        # plot_time_freq(self.resampled_input_signal, self.fs_resampled_signal, ' Resembled Signal', self.screen_shot_path)
        file_obj = open(self.result_path + "/data.txt", "w+")
        file_obj.write("Signal Name : " + self.file_name + "\n")
        current_time: float = 0
        max_ase = 0
        max_Ti = -10000
        power, energy, phase, ASE, Ti, number_small_frames_in_one_segment, total_number_small_frames, selected_bins = self.initiate_output_variables()
        for i in range(self.number_of_signals_in_file):
            current_signal_obj = self.input_signal[i * self.signal_length:(i + 1) * self.signal_length]
            file_obj.write('------------- New Frame : %d----------------\n' % (i + 1))

            end_time, start_time = self.calculate_current_frame_time(current_signal_obj, current_time)
            current_time = end_time

            # plot_time_freq(current_signal_obj, self.fs_input_signal, 'Frame ', self.screen_shot_path)

            current_feature_extractor_obj = FeatureExtractor(current_signal_obj, self.signal_length,
                                                             self.fs_input_signal, start_time,
                                                             end_time, i, self.audio_id)
            power_cur, energy_cur, phase_cur, ASE_cur, Ti_cur = current_feature_extractor_obj.calculate_signal_features()
            power[i * number_small_frames_in_one_segment:(i + 1) * number_small_frames_in_one_segment, :] = power_cur
            energy[i * number_small_frames_in_one_segment:(i + 1) * number_small_frames_in_one_segment, :] = energy_cur
            phase[i * number_small_frames_in_one_segment:(i + 1) * number_small_frames_in_one_segment, :] = phase_cur
            ASE[i * number_small_frames_in_one_segment:(i + 1) * number_small_frames_in_one_segment, :] = ASE_cur
            Ti[i * number_small_frames_in_one_segment:(i + 1) * number_small_frames_in_one_segment, :] = Ti_cur
            self.signal_cache_data.append(current_feature_extractor_obj.get_cache_obj())
            ASE_val = ASE_cur[-1, 0]
            Ti_val = Ti_cur[-1, 0]
            self.final_ASE = ASE_val
            self.final_Ti = Ti_val
            if ASE_val > 3:
                plot_time_freq(current_signal_obj, self.fs_input_signal,
                               'Segment : %d ; start =  %.3f ; End = %.3f' % (i, start_time, end_time),
                               self.screen_shot_path)
            file_obj.write(
                'start Time=%f \nEnd time = %f \nASE = %f \nTi= %f \n' % (start_time, end_time, ASE_val, Ti_val))
            max_ase = max(max_ase, ASE_val)
            max_Ti = max(max_Ti, Ti_val)

        file_obj.write('Max Values  :  \n ASE = %f  \n Ti = %f \n' % (max_ase, max_Ti))
        file_obj.close()
        return power, energy, phase, ASE, Ti, total_number_small_frames, selected_bins

    def calculate_current_frame_time(self, current_signal_obj, current_time):
        start_time = current_time
        current_time += current_signal_obj.size / self.fs_constant_signal
        end_time = current_time
        return end_time, start_time

    def __pre_process_signal(self):
        # resampled_length = np.floor(self.fs_constant_signal * self.audio_duration)
        # resampled_length = resampled_length.astype(np.int64)
        # self.resampled_input_signal = resampy.resample(self.input_signal, self.fs_input_signal,
        #                                                self.fs_constant_signal)
        # self.resampled_input_signal_length = resampled_length
        max_number_of_samples_in_audio = int(self.fs_constant_signal * self.cut_time)
        if self.input_signal.size > max_number_of_samples_in_audio:
            self.input_signal = self.input_signal[0:max_number_of_samples_in_audio]
        self.number_of_signals_in_file = np.floor(self.input_signal.size / self.signal_length).astype(
            np.int64)

    def create_required_directories(self):
        self.general_path_to_save += '/' + self.file_name
        if not os.path.exists(self.general_path_to_save):
            os.makedirs(self.general_path_to_save)
        self.screen_shot_path = self.general_path_to_save + '/screenshot/'
        if not os.path.exists(self.screen_shot_path):
            os.makedirs(self.screen_shot_path)
        self.result_path = self.general_path_to_save + '/result/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def initiate_output_variables(self):
        cut_time = self.cut_time
        window_size = 256
        shift = 64

        total_number_small_frames = int((self.fs_constant_signal * cut_time) / self.signal_length)
        number_small_frames_in_one_segment = int((self.signal_length - window_size) / shift)
        total_number_small_frames *= number_small_frames_in_one_segment

        K_min: int = 4
        K_max: int = 38
        selected_bins = K_max - K_min + 1

        power = np.zeros(shape=(total_number_small_frames, selected_bins))
        energy = np.zeros(shape=(total_number_small_frames, selected_bins))
        phase = np.zeros(shape=(total_number_small_frames, selected_bins))
        ASE = np.zeros(shape=(total_number_small_frames, 1))
        Ti = np.zeros(shape=(total_number_small_frames, 1))
        return power, energy, phase, ASE, Ti, number_small_frames_in_one_segment, total_number_small_frames, selected_bins

    # for i in *.wav; do ffmpeg -i $i -ar 8000 ../8k_fs/$i; done

    def map_data_to_model(self):
        frequency_audio_file = self.fs_input_signal
        number_of_frames = self.number_of_signals_in_file
        duration_audio_file = self.audio_duration
        final_ASE = self.final_ASE
        final_TI = self.final_Ti
        return {
            "frequency_audio_file": frequency_audio_file,
            "number_of_frames": number_of_frames,
            "duration_audio_file": duration_audio_file,
            "final_ASE": final_ASE,
            "final_TI": final_TI,
        }
