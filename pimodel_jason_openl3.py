
# coding: utf-8

# In[76]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import soundfile as sf
import resampy
import numpy as np
import minispec
import keras
import pyfftw
import time
import random
from sklearn.externals import joblib

minispec.set_fftlib(pyfftw.interfaces.numpy_fft)

US8K_CLASSES = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}


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


# In[3]:


POOLINGS = {
    'linear': {
        6144: (8, 8),
        512: (32, 24),
    },
    'mel128': {
        6144: (4, 8),
        512: (16, 24),
    },
    'mel256': {
        6144: (8, 8),
        512: (32, 24),
    },
    '16k_64_50': {
        512: (8,6)
    }
}


# In[4]:




def construct_mlp_model(input_shape, weight_decay=1e-5, num_classes=10):
    """
    Constructs a multi-layer perceptron model
    Args:
        input_shape: Shape of input data
                     (Type: tuple[int])
        weight_decay: L2 regularization factor
                      (Type: float)
    Returns:
        model: L3 CNN model
               (Type: keras.models.Model)
        input: Model input
               (Type: list[keras.layers.Input])
        output:Model output
                (Type: keras.layers.Layer)
    """
    l2_weight_decay = keras.regularizers.l2(weight_decay)
    inp = keras.layers.Input(shape=input_shape, dtype='float32')
    y = keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_weight_decay)(inp)
    y = keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_weight_decay)(y)
    y = keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2_weight_decay)(y)
    m = keras.models.Model(inputs=inp, outputs=y)
    m.name = 'urban_sound_classifier'

    return m, inp, y


# In[30]:


def amplitude_to_db(x, amin=1e-10, dynamic_range=80.0):
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)

    ref_value = power.max()

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= log_spec.max()

    log_spec = np.maximum(log_spec, -dynamic_range)

    return log_spec


# In[94]:

#audio_dir = "UrbanSound8K/audio/fold1"
#audio_path = os.path.join(audio_dir, random.choice(os.listdir(audio_dir)))
audio_path = "UrbanSound8K/audio/fold5/100032-3-0-0.wav"
model_path = "models/16k_64_50/16k_model.h5" #"models/sonyc_el3_models/openl3_audio_mel256_music.h5"
#classifier_path = "models/us8k-music-melspec2-512emb-model/model.h5"
classifier_basepath = "models/16k_64_50/downstream_fold5_0.71/"            ####### Only edit this line for alternate downstream models ########
classifier_path = classifier_basepath + "model.h5"


# In[65]:


#ls /scratch/jtc440/us8k-music-melspec2-512emb-model/


# In[95]:


with open(classifier_basepath + 'stdizer.pkl', 'rb') as f:
    stdizer = joblib.load(f)


# In[96]:


#Audio(audio_path)


# In[48]:


model = keras.models.load_model(model_path)
#model_type = os.path.basename(model_path).split('_')[2]
model_type = '16k_64_50'
embedding_size = 512

pool_size = POOLINGS[model_type][embedding_size]
y_a = keras.layers.MaxPooling2D(pool_size=pool_size, padding='same')(model.output)
y_a = keras.layers.Flatten()(y_a)
model = keras.models.Model(inputs=model.input, outputs=y_a)


# In[49]:


m_class, inp, out = construct_mlp_model((512,))
m_class.load_weights(classifier_path)


# Load and preprocess audio
audio_data, sr = sf.read(audio_path)
audio_data = audio_data.flatten()


if sr != 16000:
    audio_data = resampy.resample(audio_data, sr, 16000)

# Frame length
frame_length = 16000

# Hop length
hop_length = 320

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

frames = minispec.util.frame(audio_data, frame_length=frame_length, hop_length=frame_length).T

for i in range(10):
    start_ts = time.time()

    frame_specs = []

    for frame in frames:
        # Compute spectrogram
        if model_type == 'mel256':
            S = np.abs(minispec.core.stft(frame, n_fft=2048, hop_length=hop_length,
                                          window='hann', center=True,
                                          pad_mode='constant'))
            S = minispec.feature.melspectrogram(sr=48000, S=S,
                                                         n_mels=256, power=1.0,
                                                         htk=True)
        elif model_type == 'mel128':
            S = np.abs(minispec.core.stft(frame, n_fft=2048, hop_length=hop_length,
                                          window='hann', center=True,
                                          pad_mode='constant'))
            S = minispec.feature.melspectrogram(sr=48000, S=S,
                                                         n_mels=128, power=1.0,
                                                         htk=True)
        elif model_type == '16k_64_50':
            S = np.abs(minispec.core.stft(frame, n_fft=1024, hop_length=hop_length,
                                          window='hann', center=True,
                                          pad_mode='constant'))
            S = minispec.feature.melspectrogram(sr=16000, S=S,
                                                n_mels=64, power=1.0,
                                                htk=True)
        else:

            S = np.abs(minispec.core.stft(frame, n_fft=512, hop_length=hop_length,
                                                   window='hann', center=True,
                                                   pad_mode='constant'))
        S = amplitude_to_db(np.array(S))
        frame_specs.append(S)

    # Convert amplitude to dB
    spec_data = np.array(frame_specs)[:,:,:,np.newaxis]
    emb_data = model.predict(spec_data)
    emb_data = stdizer.transform(emb_data)
    output = m_class.predict(emb_data)
    label_idx = output.mean(axis=0).argmax()
    label = US8K_CLASSES[label_idx]

    end_ts = time.time()

    print("Iteration {}:\tTook {} seconds".format(i, end_ts - start_ts))

print('Output label: ', label)

#TODO:
# Print output
#if y == y_hat:
#    qualifier = 'correctly'
#else:
#    qualifier = 'incorrectly'

#print('-----------------------------------')
#print('The audio clip with true label ' + US8K_CLASSES[y] + ' has been ' + qualifier + ' classified as ' + US8K_CLASSES[y_hat])
#print('-----------------------------------')
# In[99]:


#(output * 100).astype(int)


# In[98]:


#label

