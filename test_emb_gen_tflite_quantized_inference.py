import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import csv
import json
import glob
import numpy as np
import resampy
from log import *
import tensorflow as tf
import soundfile as sf
import librosa

LOGGER = logging.getLogger('quantized_inference')
LOGGER.setLevel(logging.DEBUG)

def load_audio(path, sr):
    """
    Load audio file
    """
    data, sr_orig = sf.read(path, dtype='float32', always_2d=True)
    data = data.mean(axis=-1)

    if sr_orig != sr:
        data = resampy.resample(data, sr_orig, sr)

    return data

def amplitude_to_db(S, amin=1e-10, dynamic_range=80.0):
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)
    ref_value = power.max()

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= log_spec.max()

    log_spec = np.maximum(log_spec, -dynamic_range)
    return log_spec

def get_melspectrogram(frame, n_fft=2048, mel_hop_length=242, samp_rate=48000, n_mels=256, fmax=None):
    S = np.abs(librosa.core.stft(frame, n_fft=n_fft, hop_length=mel_hop_length, window='hann', center=True, pad_mode='constant'))
    S = librosa.feature.melspectrogram(sr=samp_rate, S=S, n_fft=n_fft, n_mels=n_mels, fmax=fmax, power=1.0, htk=True)
    S = amplitude_to_db(np.array(S))
    return S
    
def initialize_uninitialized_variables(sess):
    if hasattr(tf, 'global_variables'):
        variables = tf.global_variables()
    else:
        variables = tf.all_variables()

    #print(variables)
    uninitialized_variables = []
    for v in variables:
        if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
            uninitialized_variables.append(v)
            v._keras_initialized = True
    
    #print(uninitialized_variables)
    if uninitialized_variables:
        if hasattr(tf, 'variables_initializer'):
            sess.run(tf.variables_initializer(uninitialized_variables))
        else:
            sess.run(tf.initialize_variables(uninitialized_variables)) 
            
def get_l3model(model_path, saved_model_type='tflite'):
    l3embedding_model = tf.lite.Interpreter(model_path=model_path)  
    return l3embedding_model
    
def load_us8k_metadata(path):
    """
    Load UrbanSound8K metadata
    Args:
        path: Path to metadata csv file
              (Type: str)
    Returns:
        metadata: List of metadata dictionaries
                  (Type: list[dict[str, *]])
    """
    metadata = [{} for _ in range(10)]
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fname = row['slice_file_name']
            row['start'] = float(row['start'])
            row['end'] = float(row['end'])
            row['salience'] = float(row['salience'])
            fold_num = row['fold'] = int(row['fold'])
            row['classID'] = int(row['classID'])
            metadata[fold_num-1][fname] = row

    return metadata

def get_l3_frames_uniform_tflite(audio, interpreter=None, n_fft=2048, n_mels=256,\
                                 mel_hop_length=242, hop_size=0.1, sr=48000, fmax=None, **kwargs):
    """
    Get L3 embedding from tflite model for each frame in the given audio file

    Args:
        audio: Audio data or path to audio file
               (Type: np.ndarray or str)

        l3embedding_model:  Audio embedding model
                            (keras.engine.training.Model)

    Keyword Args:
        hop_size: Hop size in seconds
                  (Type: float)

    Returns:
        features:  Array of embedding vectors
                   (Type: np.ndarray)
    """

    if type(audio) == str:
        audio = load_audio(audio, sr)

    hop_size = hop_size
    hop_length = int(hop_size * sr)
    frame_length = sr * 1

    audio_length = len(audio)
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
        audio = np.pad(audio, (left_pad, right_pad), mode='constant')
   
    frames = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T
    X = []
    for frame in frames:
        S = np.abs(librosa.core.stft(frame, n_fft=n_fft, hop_length=mel_hop_length,\
                                     window='hann', center=True,\
                                     pad_mode='constant'))
        S = librosa.feature.melspectrogram(sr=sr, S=S, n_mels=n_mels, fmax=fmax,
                                           power=1.0, htk=True)
        S = amplitude_to_db(np.array(S))
        X.append(S)

    #X = np.array(X)[:, :, :, np.newaxis].astype(np.float32)

    # Get the L3 embedding for each frame
    batch_size = len(X)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape'][1:]
    output_shape = output_details[0]['shape'][1:]
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    embedding_length = output_shape[-1]
    
    #interpreter.resize_tensor_input(input_index, ((batch_size, ) + tuple(input_shape)))
    #interpreter.resize_tensor_input(output_index, ((batch_size, ) + tuple(output_shape)))
     
    print("== Input details ==")
    print(interpreter.get_input_details()[0])
    print("type:", input_details[0]['dtype'])
    print("\n== Output details ==")
    print(interpreter.get_output_details()[0])
    
    predictions = np.zeros((batch_size, embedding_length), dtype=np.float32)
    for idx in range(len(X)):
        #predictions per batch
        #print(np.array(X[idx]).shape)
        x = np.array(X[idx])[np.newaxis, :, :, np.newaxis].astype(np.float32)
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        #print('Interpreter Invoked!')
        output = interpreter.get_tensor(output_index)
        predictions[idx] = np.reshape(output, (output.shape[0], output.shape[-1]))

    print('One file done!')
    return predictions


def compute_file_features(path, feature_type, l3embedding_model=None, model_type='keras', **feature_args):
           
    if feature_type == 'l3':
        if not l3embedding_model:
            err_msg = 'Must provide L3 embedding model to use {} features'
            raise ValueError(err_msg.format(feature_type))

        if model_type == 'tflite':
            file_features = get_l3_frames_uniform_tflite(path, interpreter=l3embedding_model, **feature_args)
        else:
            raise ValueError('Only tflite models supported!')
            
    else:
        raise ValueError('Invalid feature type: {}'.format(feature_type))

    return file_features

def generate_us8k_fold_data(metadata, data_dir, fold_idx, output_dir, l3embedding_model=None, model_type='keras',
                            features='l3', random_state=12345678, **feature_args):
    """
    Generate all of the data for a specific fold

    Args:
        metadata: List of metadata dictionaries, or a path to a metadata file to be loaded
                  (Type: list[dict[str,*]] or str)

        data_dir: Path to data directory
                  (Type: str)

        fold_idx: Index of fold to load
                  (Type: int)

        output_dir: Path to output directory where fold data will be stored
                    (Type: str)

    Keyword Args:
        l3embedding_model: L3 embedding model, used if L3 features are used
                           (Type: keras.engine.training.Model or None)

        features: Type of features to be computed
                  (Type: str)

    """

    if type(metadata) == str:
        metadata = load_us8k_metadata(metadata)

    # Set random seed
    random_state = random_state + fold_idx
    random.seed(random_state)
    np.random.seed(random_state)

    audio_fold_dir = os.path.join(data_dir, "fold{}".format(fold_idx+1))

    # Create fold directory if it does not exist
    output_fold_dir = os.path.join(output_dir, "fold{}".format(fold_idx+1))
    if not os.path.isdir(output_fold_dir):
        os.makedirs(output_fold_dir)

    LOGGER.info('Generating fold {} in {}'.format(fold_idx+1, output_fold_dir))
    print('Generating fold {} in {}'.format(fold_idx+1, output_fold_dir))

    num_files = len(metadata[fold_idx])

    for idx, (fname, example_metadata) in enumerate(metadata[fold_idx].items()):
        desc = '({}/{}) Processed {} -'.format(idx+1, num_files, fname)
        with LogTimer(LOGGER, desc, log_level=logging.DEBUG):
            # TODO: Make sure glob doesn't catch things with numbers afterwards
            variants = [x for x in glob.glob(os.path.join(audio_fold_dir,
                '**', os.path.splitext(fname)[0] + '[!0-9]*[wm][ap][v3]'), recursive=True)
                if os.path.isfile(x) and not x.endswith('.jams')]
            num_variants = len(variants)
            for var_idx, var_path in enumerate(variants):
                audio_dir = os.path.dirname(var_path)
                var_fname = os.path.basename(var_path)
                desc = '\t({}/{}) Variants {} -'.format(var_idx+1, num_variants, var_fname)
                with LogTimer(LOGGER, desc, log_level=logging.DEBUG):
                    generate_us8k_file_data(var_fname, example_metadata, audio_dir,
                                            output_fold_dir, features,
                                            l3embedding_model, model_type, **feature_args)


def generate_us8k_file_data(fname, example_metadata, audio_fold_dir,
                            output_fold_dir, features,
                            l3embedding_model, model_type, **feature_args):
    audio_path = os.path.join(audio_fold_dir, fname)

    basename, _ = os.path.splitext(fname)
    output_path = os.path.join(output_fold_dir, basename + '.npz')

    if os.path.exists(output_path):
        LOGGER.info('File {} already exists'.format(output_path))
        return

    print('Filename: ', fname)
    X = compute_file_features(audio_path, features, l3embedding_model=l3embedding_model,\
                                           model_type=model_type, **feature_args)

    # If we were not able to compute the features, skip this file
    if X is None:
        LOGGER.error('Could not generate data for {}'.format(audio_path))
        return

    class_label = example_metadata['classID']
    y = class_label

    np.savez_compressed(output_path, X=X, y=y)
    return output_path, 'success'
    
if __name__=='__main__':
    #Faster inference with model whose activations are not quantized
    #Full Quantized Model: quantized_model_default.tflite
    #Only weights quantized model: quantized_model_size.tflite
    
    #Change the two paths below as per your env
    model_path = '../models/quantized_model_size.tflite' 
    dataset_output_dir = '/scratch/sk7898/test_quant_tflite'
    
    fold_num = 1
    metadata_path = '/beegfs/jtc440/UrbanSound8K/metadata/UrbanSound8K.csv'
    data_dir = '/beegfs/jtc440/UrbanSound8K/audio'
    random_state = 20180302
    samp_rate = 48000
    n_mels = 256
    n_hop = 242
    n_dft = 2048 
    fmax=None
    with_melSpec = False
    
    saved_model_type = 'tflite' 
    l3embedding_model = get_l3model(model_path, saved_model_type=saved_model_type)

    # Generate a single fold if a fold was specified
    generate_us8k_fold_data(metadata_path, data_dir, fold_num-1, dataset_output_dir,
                            l3embedding_model=l3embedding_model, model_type=saved_model_type, 
                            features='l3', random_state=random_state,
                            mel_hop_length=n_hop, n_mels=n_mels,\
                            n_fft=n_dft, fmax=fmax, sr=samp_rate, with_melSpec=with_melSpec)