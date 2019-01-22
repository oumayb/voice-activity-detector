import os

import wave

import json

import numpy as np

import matplotlib.pyplot as plt

from shutil import copyfile


def create_speakers_dirs(data_path):
    """
    Creates directories containing the files ordered by speakers
    Params
    ______
    data_path: `String`
        path to the directory
    """
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            speaker_id = filename.split("-")[0]
            path_speaker = os.path.join(root.lstrip('./'), speaker_id)
            path_file = os.path.join(root.lstrip('./'), filename)
            if not os.path.exists(path_speaker):
                os.makedirs(path_speaker)
                new_path_file = os.path.join(os.path.join(root.lstrip('./'), speaker_id), filename)
                copyfile(path_file, new_path_file)
            else:
                new_path_file = os.path.join(os.path.join(root.lstrip('./'), speaker_id), filename)
                copyfile(path_file, new_path_file)


def get_signals_labels(path):
    """
    Access the data from the given path and return diverse information about it

    Params
    ------
    path: path to the data

    Returns
    -------
    signals: `list` of `numpy arrays`objects
        list of the read waveforms
    wav_paths: `list` of String` objects`
        list of the wav file paths
    audio_durations: `list` of `float` objects
        list of durations for each wav file
    label_dicts: `list` of dictionaries
        list of label dictionaries
    fs: `int`
        sampling rate of the audios
    """
    signals = []  # numpy arrays
    wav_paths = []  # paths
    label_paths = []
    num_speech_seg = []  # Number of speech segments
    label_dicts = []  # Labels dictionaries
    audio_durations = []  # Audios durations

    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.wav'):
                wav_path = os.path.join(root.lstrip('./'), filename)
                wav_paths.append(wav_path)

                f = wave.open(wav_path)
                signal = f.readframes(-1)
                signal = np.fromstring(signal, 'Int16')
                fs = f.getframerate()
                n_frames = f.getnframes()
                duration = n_frames / fs  # duration of the recording in seconds
                audio_durations.append(duration)
                signals.append(signal)
                f.close()

                # Labels
                label = filename[:-3] + "json"
                label_path = os.path.join(root.lstrip('./'), label)
                label_paths.append(label_path)
                with open(label_path) as f:
                    label = json.load(f)
                    label_dicts.append(label)
                    num_speech_seg.append(len(label["speech_segments"]))
                    f.close()

    return signals, wav_paths, audio_durations, label_dicts, fs


def get_signals_labels_speakers(speakers_dirs):
    speakers_signals = []
    speakers_wav_paths = []
    speakers_audio_durations = []
    speakers_label_dicts = []
    speakers_fs = []

    for speaker_path in speakers_dirs:
        signals, wav_paths, audio_durations, label_dicts, fs = get_signals_labels(path=speaker_path)
        speakers_signals.append(signals)
        speakers_wav_paths.append(wav_paths),
        speakers_audio_durations.append(audio_durations)
        speakers_label_dicts.append(label_dicts)
        speakers_fs.append(fs)

    return speakers_signals, speakers_wav_paths, speakers_audio_durations, speakers_label_dicts, speakers_fs


def get_spectrograms(signals, fs):
    """
    Returns list of spectrograms of input waforms

    Params
    ------
    signals: `list` of `numpy.array`
        list of the waveforms

    fs: `int`
        sampling rate

    Returns
    -------
    specs: `list` of `numpy.array`
        List of spectrograms
    """
    specs = []
    for signal in signals:
        spec, freq, t, im = plt.specgram(signal, Fs=fs)
        specs.append(spec)

    return specs


def find_segment(start, end, time_step):
    """

    :param start:
    :param end:
    :param time_step:
    :return:
    """
    a = int(start / time_step) + 1
    b = int(end / time_step) + 1

    return a, b


def describe_sample(label_dicts, sample_idx):
    """
    Returns information about a given sample
    Params
    ------
    label_dicts: `list` of dictionaries
        List of the label dictionaries
    sample_idx: `int`
        Index of the sample whose information we want

    Returns
    -------
    speech_durations: `list of `float`
        durations of the speech segments in sample sample_idx
    silence_durations: `list` of `float`
        durations of the silence segments in sample_idx
    start_times: `list` of `float`
        list of start times of the speech segments
    end_times: `list` of `float`
        list of end times of the speech segments

    """
    speech_durations = [(segment["end_time"] - segment["start_time"]) for segment in
                        label_dicts[sample_idx]["speech_segments"]]
    start_times = [segment["start_time"] for segment in label_dicts[sample_idx]["speech_segments"]]
    end_times = [segment["end_time"] for segment in label_dicts[sample_idx]["speech_segments"]]
    silence_durations = [s - e for (s, e) in zip(start_times[1:], end_times[:-1])]

    return speech_durations, silence_durations, start_times, end_times


def pad_signals(signals):
    """
    Pads signals with 0s to make them all the same size
    Params
    ______
    signals: `list` of `numpy.arrays`
        List of signals

    Returns
    -------
    padded_signals: `list` of `numpy.arrays`
        List of signals, all the same size (the biggest), padded with 0s
    """
    max_len = 0
    padded_signals =[]
    for signal in signals:
        max_len = max(max_len, signal.shape[0])
    for signal in signals:
        pad_size = max_len - signal.shape[0]
        left_pad = int(pad_size / 2)
        right_pad = pad_size - left_pad
        padded_signals.append(np.pad(signal, (left_pad, right_pad), 'constant', constant_values=(0, 0)))

    return padded_signals


def pad_signals_with_noise(signals):
    """
    Pads signals with random floats from -500 to 500 to make them all the same size
    Params
    ______
    signals: `list` of `numpy.arrays`
        List of signals

    Returns
    -------
    padded_signals: `list` of `numpy.arrays`
        List of signals, all the same size (the biggest), padded with random numbers from -500 to 500
    """
    max_len = 0
    padded_signals =[]
    for signal in signals:
        max_len = max(max_len, signal.shape[0])
    for signal in signals:
        pad_size = max_len - signal.shape[0]
        left_pad = int(pad_size / 2)
        right_pad = pad_size - left_pad
        padded_sig = np.zeros(max_len)
        padded_sig[:left_pad] = np.random.normal(0, 200, size=left_pad)
        padded_sig[max_len - right_pad:] = np.random.normal(0, 200, size=right_pad)
        padded_sig[left_pad:max_len - right_pad] = signal
        padded_signals.append(padded_sig)

    return padded_signals


def build_datasets(X, y, mode):
    """
    Params
    ------
    X: `numpy.array`
        inputs
    y: `numpy.array`
        labels
    mode: `String`
        can be either "train", "valid", or "test". Specifies which phase we want X, y for

    Returns
    -------
    X: `numpy.array`
    y: `numpy.array`
    """
    n_samples = len(X)
    split_1 = int(0.7 * n_samples)
    split_2 = int(0.9 * n_samples)

    # Create a random seed to always produce the same splits
    np.random.seed(2)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Re order spectrograms and labels based on the shuffle
    X_ = np.array([X[i] for i in indices])
    y_ = np.array([y[i] for i in indices])

    if mode == 'train':
        X = np.array(X_[:split_1])
        y = np.array(y_[:split_1])

    elif mode == 'valid':
        X = np.array(X_[split_1:split_2])
        y = np.array(y_[split_1:split_2])

    elif mode == 'test':
        X = np.array(X_[split_2:])
        y = np.array(y_[split_2:])

    return X, y


def binarize(predictions):
    """
    Transforms the output of the network into predictions in the format of our labels
    Params
    ------
    predictions: `np.array`
        array of continous predictions (outputs of the sigmoid in the last fcl)

    Returns
    -------
    predictions_list: `list` of arrays
        Each element of the list is an array of shape T, of 0s(for speech) and 1s(for silence).
    """
    predictions_list = list(predictions)
    for i in range(len(predictions_list)):
        for j, val in enumerate(predictions_list[i]):
            if val >= 0.5:
                predictions_list[i][j] = 1
            else:
                predictions_list[i][j] = 0
    return predictions_list


def build_datasets(X, y, mode):
    """
    Splits the dataset in training, validation, and test sets

    Params
    ------
    X:
    y:
    mode: `string`
        Can be 'train', 'valid', or 'test'

    Returns
    -------
    X:
    y:
    """
    n_samples = len(X)
    split_1 = int(0.7 * n_samples)
    split_2 = int(0.9 * n_samples)

    np.random.seed(2)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Re order spectrograms and labels based on the shuffle
    X_ = np.array([X[i] for i in indices])
    y_ = np.array([y[i] for i in indices])

    if mode == 'train':
        X = np.array(X_[:split_1])
        y = np.array(y_[:split_1])

    elif mode == 'valid':
        X = np.array(X_[split_1:split_2])
        y = np.array(y_[split_1:split_2])

    elif mode == 'test':
        X = np.array(X_[split_2:])
        y = np.array(y_[split_2:])

    return X, y


def labels_to_rectangular(label_dicts, T, max_duration, durations, min_silence):
    """
    Builds a representation of the labels that will be used for the learning algorithm. Each label dictionary will be
    represented as a rectangular signal, with 0s during speech time, and 1s during silences.

    Params
    ------
    label_dicts: `list
        List of label dictionaries. label_dicts[i] corresponds to the labeled speech segments of sample i
    T: `int`
        number of time steps to discretize to. It's the output dimension of the Conv layer(s)
    max_duration: `float`
        length in seconds of the longest audio sample
    durations: `list` of floats
        durations[i] is the length in seconds of sample i
    min_silence: `float`
        minimum silence time for a silence to truly be one

    Returns
    -------
    labels_rectangular: `list` of arrays
        Each element of the list is an array of shape T, of 0s(for speech) and 1s(for silence).
    """
    labels_rectangular = []

    time_step = max_duration / T

    n_samples = len(label_dicts)

    for i in range(n_samples):

        y = np.zeros(T)
        pad_size_temp = max_duration - durations[i]
        pad_size = int(pad_size_temp * T / max_duration)
        left_pad = int(pad_size / 2)
        right_pad = pad_size - left_pad

        speech_durations, silence_durations, start_times, end_times = describe_sample(label_dicts=label_dicts,
                                                                                      sample_idx=i)
        silence_start_times = end_times[:-1]
        silence_end_times = start_times[1:]

        for (j, silence_duration) in enumerate(silence_durations):
            if silence_duration >= min_silence:
                s, e = find_segment(silence_start_times[j], silence_end_times[j], time_step)
                y[s + left_pad:e + left_pad] = 1

        y[:left_pad] = 1
        y[T - right_pad:] = 1

        labels_rectangular.append(y)

    return labels_rectangular


def smooth_predictions(predictions):
    """
    Smooths the output predicted probabilities of the network
    Params
    ------
    predictions: `list` of `numpy.array`
        each array of the list is (538, 1), and contains the probabilities of each of the 538 time steps to be 1

    Returns
    -------
    smoothed_predictions: `list of smoothed probabilities`
    """
    n = len(predictions)
    pred_size = len(predictions[0])

    smoothed_predictions = []
    for i in range(n):
        smoothed_prediction = np.zeros(pred_size)
        for j in range(3):
            smoothed_prediction[j] = predictions[i][j]
            smoothed_prediction[pred_size - j - 1] = predictions[i][pred_size - j - 1]

        for h in range(3, pred_size - 3):
            smoothed_prediction[h] = (predictions[i][h - 3] + predictions[i][h - 2] + predictions[i][h - 1] +
                                      predictions[i][h] + predictions[i][h + 1] + predictions[i][h + 2] +
                                      predictions[i][h + 3]) / 7

        smoothed_predictions.append(smoothed_prediction)

    return smoothed_predictions
