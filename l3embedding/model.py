import h5py
from keras.layers import concatenate, Dense
from .vision_model import *
from .audio_model import *
from .training_utils import multi_gpu_model, conv_keyval_lists_to_dict

global_thresholds = None

def L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, model_name, layer_size=128):
    """
    Merges the audio and vision subnetworks and adds additional fully connected
    layers in the fashion of the model used in Look, Listen and Learn

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
    # Merge the subnetworks
    weight_decay = 1e-5
    y = concatenate([vision_model(x_i), audio_model(x_a)])
    y = Dense(layer_size, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Dense(2, activation='softmax',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    m = Model(inputs=[x_i, x_a], outputs=y)
    m.name = model_name

    return m, [x_i, x_a], y


def convert_num_gpus(model, inputs, outputs, model_type, src_num_gpus, tgt_num_gpus):
    """
    Converts a multi-GPU model to a model that uses a different number of GPUs

    If the model is single-GPU/CPU, the given model is returned

    Args:
        model:  Keras model
                (Type: keras.models.Model)

        inputs: Input Tensor.
                (Type: keras.layers.Input)

        outputs: Embedding output Tensor/Layer.
                 (Type: keras.layers.Layer)

        model_type: Name of model type
                    (Type: str)

        src_num_gpus: Number of GPUs the source model uses
                      (Type: int)

        tgt_num_gpus: Number of GPUs the converted model will use
                      (Type: int)

    Returns:
        model_cvt:  Embedding model object
                    (Type: keras.engine.training.Model)

        inputs_cvt: Input Tensor. Not returned if return_io is False.
                    (Type: keras.layers.Input)

        ouputs_cvt: Embedding output Tensor/Layer. Not returned if return_io is False.
                    (Type: keras.layers.Layer)
    """
    if src_num_gpus <= 1 and tgt_num_gpus <= 1:
        return model, inputs, outputs

    m_new, inputs_new, output_new = MODELS[model_type]()
    m_new.set_weights(model.layers[-2].get_weights())

    if tgt_num_gpus > 1:
        m_new = multi_gpu_model(m_new, gpus=tgt_num_gpus)

    return m_new, inputs_new, output_new


def convert_num_gpus_new(model, inputs, outputs, model_type, src_num_gpus, tgt_num_gpus, thresholds=None, old_model=None):
    """
    Converts a multi-GPU model to a model that uses a different number of GPUs

    If the model is single-GPU/CPU, the given model is returned

    Args:
        model:  Keras model
                (Type: keras.models.Model)

        inputs: Input Tensor.
                (Type: keras.layers.Input)

        outputs: Embedding output Tensor/Layer.
                 (Type: keras.layers.Layer)

        model_type: Name of model type
                    (Type: str)

        src_num_gpus: Number of GPUs the source model uses
                      (Type: int)

        tgt_num_gpus: Number of GPUs the converted model will use
                      (Type: int)
        
        thresholds:  Threshold list for the pruned model mask creation
                    (Type: float list)
        old_model:  If the new model is reduced, the old model architecture is loaded from old_model

    Returns:
        model_cvt:  Embedding model object
                    (Type: keras.engine.training.Model)

        inputs_cvt: Input Tensor. Not returned if return_io is False.
                    (Type: keras.layers.Input)

        ouputs_cvt: Embedding output Tensor/Layer. Not returned if return_io is False.
                    (Type: keras.layers.Layer)
    """
    if src_num_gpus <= 1 and tgt_num_gpus <= 1:
        return model, inputs, outputs

    if thresholds is not None:
        m_new, inputs_new, output_new = PRUNING_MODELS[model_type](thresholds)
    else:
        m_new, inputs_new, output_new = old_model, inputs, outputs
    m_new.set_weights(model.layers[-2].get_weights())

    if tgt_num_gpus > 1:
        m_new = multi_gpu_model(m_new, gpus=tgt_num_gpus)

    return m_new, inputs_new, output_new


def load_new_model(weights_path, model_type, src_num_gpus=0, tgt_num_gpus=None, thresholds=None, return_io=False, inputs=None, outputs=None):
    """
    Loads an audio-visual correspondence model

    Args:
        weights_path:  Path to Keras weights file
                       (Type: str)
        model_type:    Name of model type if thresholds
                       (Type: str)
                       Model id thresholds=None
                       (Type: Model)

    Keyword Args:
        src_num_gpus:   Number of GPUs the saved model uses
                        (Type: int)

        tgt_num_gpus:   Number of GPUs the loaded model will use
                        (Type: int)

        return_io:  If True, return input and output tensors
                    (Type: bool)
    
        inputs, output:     Is passed if model_type is of type Model

    Returns:
        model:  Loaded model object
                (Type: keras.engine.training.Model)
        x_i:    Input Tensor. Not returned if return_io is False.
                (Type: keras.layers.Input)
        y_i:    Embedding output Tensor/Layer. Not returned if return_io is False.
                (Type: keras.layers.Layer)
    """
    if thresholds is not None and model_type not in PRUNING_MODELS:
        raise ValueError('Invalid model type: "{}"'.format(model_type))

    if thresholds is not None:
        global global_thresholds
        global_thresholds = thresholds
        m, inputs, output = PRUNING_MODELS[model_type](thresholds)
        #m, inputs, output = construct_cnn_L3_melspec2_masked(thresholds)
        print("Loaded new model")
    else:
        m, inputs, output = model_type, inputs, outputs
    
    old_m = m

    if src_num_gpus > 1:
        m = multi_gpu_model(m, gpus=4)

    print(m.summary())
    m.load_weights(weights_path)
    print("Loaded weights")

    if tgt_num_gpus is not None and src_num_gpus != tgt_num_gpus:
        m, inputs, output = convert_num_gpus_new(m, inputs, output, model_type,
                                                 src_num_gpus, tgt_num_gpus, thresholds=thresholds, old_model=old_m)

    if return_io:
        return m, inputs, output
    else:
        return m


def load_model(weights_path, model_type, src_num_gpus=0, tgt_num_gpus=None, return_io=False):
    """
    Loads an audio-visual correspondence model

    Args:
        weights_path:  Path to Keras weights file
                       (Type: str)
        model_type:    Name of model type
                       (Type: str)

    Keyword Args:
        src_num_gpus:   Number of GPUs the saved model uses
                        (Type: int)

        tgt_num_gpus:   Number of GPUs the loaded model will use
                        (Type: int)

        return_io:  If True, return input and output tensors
                    (Type: bool)

    Returns:
        model:  Loaded model object
                (Type: keras.engine.training.Model)
        x_i:    Input Tensor. Not returned if return_io is False.
                (Type: keras.layers.Input)
        y_i:    Embedding output Tensor/Layer. Not returned if return_io is False.
                (Type: keras.layers.Layer)
    """
    if model_type not in MODELS:
        raise ValueError('Invalid model type: "{}"'.format(model_type))

    m, inputs, output = MODELS[model_type]()
    
    if src_num_gpus > 1:
        m = multi_gpu_model(m, gpus=src_num_gpus)
    m.load_weights(weights_path)

    if tgt_num_gpus is not None and src_num_gpus != tgt_num_gpus:
        m, inputs, output = convert_num_gpus(m, inputs, output, model_type,
                                             src_num_gpus, tgt_num_gpus)
    if return_io:
        return m, inputs, output
    else:
        return m


def load_embedding(weights_path, model_type, embedding_type, pooling_type,
                   kd_model=False, src_num_gpus=0, tgt_num_gpus=None, thresholds=None, return_io=False,
                   from_convlayer=8):
    """
    Loads an embedding model

    Args:
        weights_path:    Path to Keras weights file
                         (Type: str)
        model_type:      Name of model type
                         (Type: str)
        embedding_type:  Type of embedding to load ('audio' or 'vision')
                         (Type: str)
        pooling_type:    Type of pooling applied to final convolutional layer
                         (Type: str)
        from_convlayer:  Get embedding from convlayer# (default is 8)

    Keyword Args:
        src_num_gpus:   Number of GPUs the saved model uses
                        (Type: int)

        tgt_num_gpus:   Number of GPUs the loaded model will use
                        (Type: int)

        return_io:  If True, return input and output tensors
                    (Type: bool)

    Returns:
        model:  Embedding model object
                (Type: keras.engine.training.Model)
        x_i:    Input Tensor. Not returned if return_io is False.
                (Type: keras.layers.Input)
        y_i:    Embedding output Tensor/Layer. Not returned if return_io is False.
                (Type: keras.layers.Layer)
    """
    '''
    def print_attrs(name, obj):
        print(name)
    '''

    def relabel_embedding_layer(audio_model, embedding_layer_num):
        count = 1

        for layer in audio_model.layers:
            layer_name = layer.name

            if (layer_name[0:6] == 'conv2d' or layer_name == 'audio_embedding_layer'):
                # Rename the conv layers as conv_1, conv_2 .... conv_8, and relabel audio embedding layer
                if count == embedding_layer_num:
                    layer.name = 'audio_embedding_layer'
                else:
                    layer.name = 'conv_' + str(count)

                count += 1
                # print (layer.name)
        return audio_model

    if 'masked' in model_type:
        # Convert thresholds list to dictionary
        conv_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8']
        thresholds = conv_keyval_lists_to_dict(conv_layers, thresholds)
        
        f = h5py.File(weights_path, 'r')
        #f = f['cnn_L3_masked']
        #f.visititems(print_attrs)
        
        #for layer in model:
        #    print(layer)

        #print(list(f['cnn_L3_masked'].keys()))
        #exit(0)
        
        m, inputs, output = load_new_model(weights_path, model_type, src_num_gpus=src_num_gpus,
                                           tgt_num_gpus=tgt_num_gpus, thresholds=thresholds, return_io=True)
    else:
        m, inputs, output = load_model(weights_path, model_type, src_num_gpus=src_num_gpus,
                                       tgt_num_gpus=tgt_num_gpus, return_io=True)
    if 'audio' in model_type:
        x_a = inputs
    else:
        x_i, x_a = inputs
    if embedding_type == 'vision':
        m_embed_model = m.get_layer('vision_model')
        m_embed, x_embed, y_embed = VISION_EMBEDDING_MODELS[model_type](m_embed_model, x_i)

    elif embedding_type == 'audio':
        if not 'audio' in model_type:
            m_embed_model = m.get_layer('audio_model')
        else:
            m_embed_model = m

        # m_embed, x_embed, y_embed = AUDIO_EMBEDDING_MODELS[model_type](m_embed_model, x_a)
        if from_convlayer==8:
            m_embed, x_embed, y_embed = convert_audio_model_to_embedding(m_embed_model, x_a, model_type, pooling_type, kd_model)
        else:
            m_embed, x_embed, y_embed = convert_audio_model_to_embedding(relabel_embedding_layer(m_embed_model, from_convlayer),\
                                                                        x_a, model_type, pooling_type)
    else:
        raise ValueError('Invalid embedding type: "{}"'.format(embedding_type))

    if return_io:
        return m_embed, x_embed, y_embed
    else:
        return m_embed


def gpu_wrapper(model_f):
    """
    Decorator for creating multi-gpu models
    """
    def wrapped(num_gpus=0, *arwargs):
        if global_thresholds is None:
            m, inp, out = model_f(*args, **kwargs)
        else:
            m, inp, out = model_f(global_thresholds)
        if num_gpus > 1:
            m = multi_gpu_model(m, gpus=num_gpus)

        return m, inp, out

    return wrapped


@gpu_wrapper
def construct_cnn_L3_orig():
    """
    Constructs a model that replicates that used in Look, Listen and Learn

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
    vision_model, x_i, y_i = construct_cnn_L3_orig_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_orig_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_orig')
    return m

@gpu_wrapper
def construct_cnn_L3_kapredbinputbn():
    """
    Constructs a model that replicates that used in Look, Listen and Learn

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
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_kapredbinputbn_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_kapredbinputbn')
    return m

@gpu_wrapper
def construct_cnn_L3_melspec1():
    """
    Constructs a model that replicates that used in Look, Listen and Learn

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
    vision_model, x_i, y_i = construct_cnn_L3_orig_vision_model() #construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_melspec1_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_kapredbinputbn')
    return m

@gpu_wrapper
def construct_cnn_L3_melspec2():
    """
    Constructs a model that replicates that used in Look, Listen and Learn

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
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_melspec2_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_kapredbinputbn')
    return m


@gpu_wrapper
def construct_cnn_L3_melspec2_masked(thresholds):
    """
    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
        
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_melspec2_masked_audio_model(thresholds)

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_masked')
    return m


@gpu_wrapper
def construct_cnn_L3_melspec2_kd(masks):
    """
    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_melspec2_kd_audio_model(masks)

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_kapredbinputbn_kd')
    return m


@gpu_wrapper
def construct_tiny_L3():
    """
    Constructs a model that implements a small L3 model for validation purposes

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    vision_model, x_i, y_i = construct_tiny_L3_vision_model()
    audio_model, x_a, y_a = construct_tiny_L3_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'tiny_L3', layer_size=64)
    return m


MODELS = {
    'cnn_L3_orig': construct_cnn_L3_orig,
    'tiny_L3': construct_tiny_L3,
    'cnn_L3_kapredbinputbn': construct_cnn_L3_kapredbinputbn,
    'cnn_L3_melspec1': construct_cnn_L3_melspec1,
    'cnn_L3_melspec2': construct_cnn_L3_melspec2,
    'cnn_L3_melspec2_audioonly': construct_cnn_L3_melspec2_audio_model
}

PRUNING_MODELS = {
    'cnn_L3_melspec2_masked': construct_cnn_L3_melspec2_masked,
    'cnn_L3_melspec2_masked_audio': construct_cnn_L3_melspec2_masked_audio_model
}


if __name__=='__main__':
    # model_path = '../models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
    model_path = '../pruned_model/pruned_audio_0.71586.h5'
    model_type = 'cnn_L3_melspec2_audioonly'
    pooling_type = 'original'
    num_gpus = 0

    l3embedding_model = load_embedding(model_path,
                                       model_type,
                                       'audio', pooling_type,
                                       tgt_num_gpus=num_gpus, from_convlayer=8)
