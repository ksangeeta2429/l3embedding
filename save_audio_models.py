import argparse
import keras
import os
import glob
from l3embedding.model import load_model
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation, Lambda
from kapre.time_frequency import Spectrogram, Melspectrogram
import tensorflow as tf
import keras.regularizers as regularizers


def construct_cnn_L3_melspec2_spec_model(n_mels=256, n_hop = 242, n_dft = 2048, asr = 48000, audio_window_dur = 1):
    """
    Constructs a model that replicates the audio subnetwork  used in Look,
    Listen and Learn
    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .
    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    weight_decay = 1e-5

    #n_frames = 1 + int((asr * audio_window_dur) / float(n_hop))
    #x_a = Input(shape=(n_mels, n_frames, 1), dtype='float32')
    #y_a = BatchNormalization()(x_a)

    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # MELSPECTROGRAM PREPROCESSING
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
                      return_decibel_melgram=True, padding='same')(x_a)
    y_a = BatchNormalization()(y_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    #pool_size_a_4 = (32, 24)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    
    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a


if __name__ == '__main__':

    samp_rate = 48000
    n_mels = 256
    n_hop = 242
    n_dft = 2048
    mt = 'cnn_L3_melspec2'
    
    weight_dir_ft = '/scratch/sk7898/l3pruning/embedding/fixed/finetune/full_model'
    output_dir_ft = '/scratch/sk7898/l3pruning/embedding/fixed/finetune/audio_model'

    weight_dir_kd = '/scratch/sk7898/l3pruning/embedding/fixed/kd/full_model'
    output_dir_kd = '/scratch/sk7898/l3pruning/embedding/fixed/kd/audio_model'

    files_ft = glob.glob(os.path.join(weight_dir_ft, '*best_valid_accuracy_sparsity*'))
    files_kd = glob.glob(os.path.join(weight_dir_kd, '*best_valid_loss_sparsity*'))

    for weight_file in files_ft:
        sparsity = os.path.splitext(weight_file)[0].split('_')[-1]
        print(sparsity)
        
        # Load and convert model back to 1 gpu
        print("Loading fine-tuned model.......................")
        m, inputs, outputs = load_model(weight_file, mt, src_num_gpus=1, tgt_num_gpus=1, \
                                        return_io=True, n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, asr=samp_rate)
        _, x_a = inputs
        audio_model_output = m.get_layer('audio_model').get_layer('audio_embedding_layer').output
        audio_embed_model = Model(inputs=x_a, outputs=audio_model_output)
        audio_output_path = os.path.join(output_dir_ft, 'edgel3_ft_audio_sparsity_{}.h5'.format(sparsity))

        weights = audio_embed_model.get_weights()

        # Save converted model back to disk
        audio_spec_embed_model, _, _ = construct_cnn_L3_melspec2_spec_model(n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, asr=samp_rate)
        audio_spec_embed_model.set_weights(weights)
        audio_spec_embed_model.save(audio_output_path)

        print('Finetuned audio model saved: ', audio_output_path)

    for weight_file in files_kd:
        sparsity = os.path.splitext(weight_file)[0].split('_')[-1]
        print(sparsity)
        
        # Load and convert model back to 1 gpu
        print("Loading knowledge distilled model.......................")
        m, inputs, outputs = load_model(weight_file, mt, src_num_gpus=1, tgt_num_gpus=1, \
                                        return_io=True, n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, asr=samp_rate)
        _, x_a = inputs
        audio_model_output = m.get_layer('audio_model').get_layer('audio_embedding_layer').output
        audio_embed_model = Model(inputs=x_a, outputs=audio_model_output)
        audio_output_path = os.path.join(output_dir_kd, 'edgel3_kd_audio_sparsity_{}.h5'.format(sparsity))

        weights = audio_embed_model.get_weights()

        # Save converted model back to disk
        audio_spec_embed_model, _, _ = construct_cnn_L3_melspec2_spec_model(n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, asr=samp_rate)
        audio_spec_embed_model.set_weights(weights)
        audio_spec_embed_model.save(audio_output_path)

        print('Distilled audio model saved: ', audio_output_path)

