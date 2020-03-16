import argparse
import keras
import os
import glob
from l3embedding.model import load_model
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation, Lambda
import tensorflow as tf
import keras.regularizers as regularizers
from kapre.time_frequency import Melspectrogram

def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Convert multi-GPU model to single-GPU model and save the model instead of just the weights')

    parser.add_argument('-wdir',
                        '--multiGPU-weight-dir',
                        dest='multiGPU_weight_dir',
                        action='store',
                        type=str,
                        help='Path to L3 embedding multi-GPU model weights file directory')

    parser.add_argument('--gpus',
                        dest='gpus',
                        type=int,
                        default=4,
                        help='Number of gpus used to train model.')

    parser.add_argument('-srate',
                        '--samp-rate',
                        dest='samp_rate',
                        action='store',
                        type=int,
                        default=48000,
                        help='Sampling rate')

    parser.add_argument('-audio',
                        '--only-audio',
                        dest='only_audio',
                        action='store_true',
                        default=False,
                        help='Save only audio model?')

    parser.add_argument('-nmels',
                        '--num-mels',
                        dest='n_mels',
                        action='store',
                        type=int,
                        default=256,
                        help='Number of mel filters')

    parser.add_argument('-lhop',
                        '--hop-length',
                        dest='n_hop',
                        action='store',
                        type=int,
                        default=242,
                        help='Melspec hop length in samples')

    parser.add_argument('-ndft',
                        '--num-dft',
                        dest='n_dft',
                        action='store',
                        type=int,
                        default=2048,
                        help='DFT size')

    parser.add_argument('-fmax',
                        '--freq-max',
                        dest='fmax',
                        action='store',
                        type=int,
                        default=None,
                        help='Max. freq in DFT')

    parser.add_argument('-half',
                        '--halved-filters',
                        dest='halved_convs',
                        action='store_true',
                        default=False,
                        help='Use half the number of conv. filters as in the original audio model?')

    parser.add_argument('-melSpec',
                        '--melSpec',
                        dest='melSpec',
                        action='store_true',
                        default=False,
                        help='Set to True is Melspec is included in the model')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where single-GPU models will be stored')

    return vars(parser.parse_args())


def construct_cnn_L3_melspec2_spec_model(n_mels=256, n_hop = 242, n_dft = 2048,
                                         fmax=None, asr = 48000, halved_convs=False, audio_window_dur = 1):
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

    n_frames = 1 + int((asr * audio_window_dur) / float(n_hop))
    x_a = Input(shape=(n_mels, n_frames, 1), dtype='float32')
    y_a = BatchNormalization()(x_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    if halved_convs:
        n_filter_a_1 //= 2

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
    if halved_convs:
        n_filter_a_2 //= 2

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
    if halved_convs:
        n_filter_a_3 //= 2

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
    if halved_convs:
        n_filter_a_4 //= 2

    filt_size_a_4 = (3, 3)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)

    pool_size_a_4 = tuple(y_a.get_shape().as_list()[1:3])
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    y_a = Flatten()(y_a)
    
    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a


if __name__ == '__main__':
    args = parse_arguments()

    samp_rate = args['samp_rate']
    n_mels = args['n_mels']
    n_hop = args['n_hop']
    n_dft = args['n_dft']
    fmax = args['fmax']
    src_gpus = args['gpus']
    weight_dir = args['multiGPU_weight_dir']
    output_dir = args['output_dir']
    halved_convs = args['halved_convs']
    melSpec = args['melSpec']
    
    model_id = weight_dir.split('/')[-1]
    mt = os.path.basename(os.path.dirname(weight_dir))
    
    if halved_convs:
        input_repr = str(samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)+'_half'
    else:
        input_repr = str(samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)

    # Add suffix for fmax
    if fmax:
        input_repr = input_repr + '_fmax_' + str(fmax)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    weight_file = glob.glob(os.path.join(weight_dir, '*best_valid_loss*'))[0]

    # Load and convert model back to 1 gpu
    print("Loading model.......................")
    if src_gpus > 1:
        m, inputs, outputs = load_model(weight_file, mt, src_num_gpus=src_gpus, tgt_num_gpus=1, return_io=True, \
                                        n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, halved_convs=halved_convs, fmax=fmax, asr=samp_rate)
        _, x_a = inputs
    else:
        m = keras.models.load_model(weight_file, custom_objects={'Melspectrogram': Melspectrogram})
        model_output_path = os.path.join(output_dir, 'l3_audio_{}_{}.h5'.format(model_id, input_repr))
        audio_spec_embed_model, _, _ = construct_cnn_L3_melspec2_spec_model(n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, \
                                                                            halved_convs=halved_convs, asr=samp_rate)
        audio_spec_embed_model.set_weights(m.get_weights()[3:])
        audio_spec_embed_model.save(model_output_path)
        print(model_output_path)

    if args['only_audio'] and not melSpec:
        model_output_path = os.path.join(output_dir, 'l3_audio_{}_{}.h5'.format(model_id, input_repr))
        audio_model_output = m.get_layer('audio_model').get_layer('audio_embedding_layer').output
        audio_embed_model = Model(inputs=x_a, outputs=audio_model_output)
        
        weights = audio_embed_model.get_weights()[3:]

        # Save converted model back to disk
        audio_spec_embed_model, _, _ = construct_cnn_L3_melspec2_spec_model(n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, \
                                                                            halved_convs=halved_convs, asr=samp_rate)
        audio_spec_embed_model.set_weights(weights)
        audio_spec_embed_model.save(model_output_path)
    
    elif args['only_audio'] and melSpec:
        model_output_path = os.path.join(output_dir, 'l3_audio_melSpec_{}_{}.h5'.format(model_id, input_repr))
        audio_model = m.get_layer('audio_model')
        audio_model.save(model_output_path)

    else:
        model_output_path = os.path.join(output_dir, 'l3_full_{}_{}.h5'.format(model_id, input_repr))
        m.save(model_output_path)

    print('Single GPU Model saved: ', model_output_path)
