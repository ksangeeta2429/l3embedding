
# coding: utf-8

# In[76]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import soundfile as sf
import resampy
import numpy as np
import minispec
import pyfftw
import time

import argparse

minispec.set_fftlib(pyfftw.interfaces.numpy_fft)


# In[2]:


def pcm2float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def amplitude_to_db(x, amin=1e-10, dynamic_range=80.0):
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)

    ref_value = power.max()

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= log_spec.max()

    log_spec = np.maximum(log_spec, -dynamic_range)

    return log_spec

audio_path = "UrbanSound8K/audio/fold5/100032-3-0-0.wav"

audio_data, sr = sf.read(audio_path)
audio_data = audio_data.flatten()

# Target sampling rate
target_sr = int(input("Enter target sampling rate (Hz): ")) #16000

# n_mels
n_mels = int(input("Enter number of mel banks: ")) #64

# Hop length
hop_length = int(input("Enter hop length (samples): ")) #968

# n_fft
n_fft = int(input("Enter FFT size (samples): ")) #1024

if sr != target_sr:
    audio_data = resampy.resample(audio_data, sr, target_sr)

# Frame length
frame_length = target_sr

# Padding logic
audio_length = len(audio_data)
if audio_length < frame_length:
    # Make sure we can have at least one frame of audio
    pad_length = frame_length - audio_length
else:
    # Zero pad so we compute embedding on all samples
    pad_length = int(np.ceil(audio_length - frame_length)/hop_length) * hop_length \
                 - (audio_length - frame_length)

if pad_length > 0:
    # Use (roughly) symmetric padding
    left_pad = pad_length // 2
    right_pad= pad_length - left_pad
    audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')

frames = minispec.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length).T

start_ts = time.time()

frame_specs = []

for frame in frames:
    # Compute spectrogram
    S = np.abs(minispec.core.stft(frame, n_fft=n_fft, hop_length=hop_length,
                                  window='hann', center=True,
                                  pad_mode='constant'))
    S = minispec.feature.melspectrogram(sr=target_sr, S=S,
                                        n_mels=n_mels, power=1.0,
                                        htk=True)
    S = amplitude_to_db(np.array(S))
    frame_specs.append(S)

# Convert amplitude to dB
spec_data = np.array(frame_specs)[:,:,:,np.newaxis]

end_ts = time.time()

print("Took {} seconds".format(end_ts - start_ts))