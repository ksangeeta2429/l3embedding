import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import random
import librosa
import h5py
import tensorflow as tf
import keras
from keras.models import Model
import keras.regularizers as regularizers
from keras.optimizers import Adam
from l3embedding.audio import pcm2float
from resampy import resample
import pescador
from skimage import img_as_float

def tflite_io_details(quant_output_path):
    interpreter = tf.lite.Interpreter(model_path=str(quant_output_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape'][1:]
    output_shape = output_details[0]['shape'][1:]
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    interpreter.allocate_tensors()

    print("== Input details ==")
    print(interpreter.get_input_details()[0])
    print("type:", input_details[0]['dtype'])
    print("\n== Output details ==")
    print(interpreter.get_output_details()[0])
    
def shuffle_files(iterable):
    lst = list(iterable)
    random.shuffle(lst)
    return iter(lst)

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

def quant_data_generator_classifier(data_dir, random_state=None):

    if random_state:
        random.seed(random_state)
    
    for fname in shuffle_files(os.listdir(data_dir)):
        print(fname)
        X = np.load(os.path.join(data_dir, fname))
        X = X['embedding']

        x = random.choice(X).reshape(1, -1).astype(np.float32)
        #print(x.shape) #(1, 256)
        
        return x
    
def quant_data_generator(data_dir, batch_size=512, samp_rate=48000, n_fft=2048, \
                         n_mels=256, mel_hop_length=242, hop_size=0.1, fmax=None,\
                         random_state=None, start_batch_idx=None):

    if random_state:
        random.seed(random_state)
        
    frame_length = samp_rate * 1

    batch = None
    curr_batch_size = 0
    batch_idx = 0
       
    for fname in shuffle_files(os.listdir(data_dir)):
        #print(fname)
        data_batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        data_blob = h5py.File(data_batch_path, 'r')
        blob_size = len(data_blob['audio'])

        while blob_start_idx < blob_size:
            blob_end_idx = min(blob_start_idx + batch_size - curr_batch_size, blob_size)

            # If we are starting from a particular batch, skip computing all of
            # the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                if batch is None:
                    batch = data_blob['audio'][blob_start_idx:blob_end_idx]
                else:
                    batch = np.concatenate([batch, data_blob['audio'][blob_start_idx:blob_end_idx]])

            curr_batch_size += blob_end_idx - blob_start_idx
            blob_start_idx = blob_end_idx

            if blob_end_idx == blob_size:
                data_blob.close()

            if curr_batch_size == batch_size:
                X = []
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Convert audio to float
                    if(samp_rate==48000):
                        batch = pcm2float(batch, dtype='float32')
                    else:
                        batch = resample(pcm2float(batch, dtype='float32'),
                                         sr_orig=48000,
                                         sr_new=samp_rate)

                    X = [get_melspectrogram(batch[i].flatten(),
                                            n_fft=n_fft,
                                            mel_hop_length=mel_hop_length,
                                            samp_rate=samp_rate, 
                                            n_mels=n_mels, 
                                            fmax=fmax) for i in range(batch_size)]

                    batch = np.array(X)[:, :, :, np.newaxis]
                    #print(np.shape(batch)) #(64, 256, 191, 1)
                    return batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None

def single_epoch_test_data_generator(data_dir, epoch_size, **kwargs):
    for _ in range(epoch_size):
        x = quant_data_generator(data_dir, **kwargs)
        yield x

def quantize_classifier(classifier_path, calibrate_data_dir, quant_mode='default',
                        output_dir=None, quant_type='int8', calibration_steps=256):

    def representative_dataset_gen():      
        print('Calibrating.........')
        for _ in range(calibration_steps):
            x = quant_data_generator_classifier(calibrate_data_dir)
            yield [x]
            
    classifier = tf.keras.models.load_model(classifier_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(classifier)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  
    converter.representative_dataset = representative_dataset_gen

    tflite_model_file = os.path.join(output_dir, 'quant_mlp_ust_default_int8.h5')
    tflite_model = converter.convert()
    with open(tflite_model_file, "wb") as f:
        f.write(tflite_model)    
        
    return tflite_model_file
        
def quantize_l3(tflite_model_file, keras_model, quant_mode='default',\
                n_mels=256, n_hop=242, n_dft=2048, asr=48000, halved_convs=False,\
                quant_type='int8', calibrate_data_dir=None, calibration_steps=1024):

    def representative_dataset_gen():
        print('Calibrating.........')
        for _ in range(calibration_steps):
            x = quant_data_generator(calibrate_data_dir,
                                     batch_size=1,
                                     samp_rate=asr,
                                     n_fft=n_dft,
                                     n_mels=n_mels,
                                     mel_hop_length=n_hop)
            yield [np.array(x).astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    #converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
 
    if quant_mode == 'default' or quant_mode == 'latency':
        if calibrate_data_dir is None:
            raise ValueError('Quantized activation calibration needs data directory!')

        if quant_mode == 'default':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            
        if quant_type == 'int8':
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]        
        if quant_type == 'float16':
            converter.target_spec.supported_types = [tf.float16]
            
        converter.representative_dataset = representative_dataset_gen
                
    elif quant_mode == 'size':
        if quant_type == 'float16':
            converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.representative_dataset = representative_dataset_gen
    else:
        raise ValueError('Unrecognized Quantization mode!')

    tflite_model = converter.convert()
    with open(tflite_model_file, "wb") as f:
        f.write(tflite_model)
        
def post_training_l3_quantization(model_path, calibrate_data_dir, quant_mode='default',\
                                  n_mels=256, n_hop=242, n_dft=2048, asr=48000, halved_convs=False,\
                                  quant_type='int8', calibration_steps=1024):
    
    dir_prefix = '/scratch/sk7898/quantization/' + os.path.basename(model_path).strip('.h5')
    
    if not os.path.isdir(dir_prefix):
        os.makedirs(dir_prefix)
    
    keras_model = tf.keras.models.load_model(model_path)
    print(keras_model.summary())
    
    print('Quantizing keras model and saving as tflite')
    tflite_model_file = os.path.join(dir_prefix, 
                                     'quantized_'+ quant_mode + '_'+ quant_type + '.tflite')
    
    quantize_l3(tflite_model_file, 
                keras_model, 
                quant_mode=quant_mode,
                quant_type=quant_type,
                asr=asr, n_mels=n_mels,
                n_hop=n_hop,
                n_dft=n_dft,
                halved_convs=halved_convs,
                calibrate_data_dir=calibrate_data_dir,
                calibration_steps=calibration_steps)
    
    return tflite_model_file

if __name__ == '__main__':
    
    model_type = 'downstream'
    
    if model_type == 'upstream':
        model_path = '/scratch/sk7898/models/reduced_input/embedding/environmental/audio_models/'\
                     'l3_audio_20200304152812_8000_64_160_1024_half.h5'
        calibrate_data_dir = '/beegfs/work/AudioSetSamples_environmental/environmental_train'
        calibration_steps = 5

        quant_mode = 'size' #Options: {'size', 'default', 'latency'}
        n_mels = 64
        n_hop = 160
        n_dft = 1024
        asr = 8000
        halved_convs=True if 'half' in model_path else False
        quant_type = 'int8'

        quant_output_path = post_training_l3_quantization(model_path, 
                                                          calibrate_data_dir,
                                                          quant_mode=quant_mode,
                                                          n_mels=n_mels, 
                                                          n_hop=n_hop,
                                                          n_dft=n_dft, 
                                                          asr=asr,
                                                          halved_convs=halved_convs,
                                                          quant_type=quant_type,
                                                          calibration_steps=calibration_steps)
    else:
        calibrate_data_dir = '/scratch/sk7898/embeddings/db_mel/sonyc_ust/quantized_l3/'\
                             'default_int8_toco/l3_audio_20200304152812_8000_64_160_1024_half'
        classifier_dir = '/scratch/sk7898/embeddings/classifier/sonyc_ust/mlp/db_mel/default_int8_toco/0_0/results'
        classifier_path = os.path.join(classifier_dir, 'mlp_ust.h5')
        calibration_steps = 512
        quant_output_path = quantize_classifier(classifier_path, calibrate_data_dir, 
                                                output_dir=classifier_dir,
                                                calibration_steps=calibration_steps)
