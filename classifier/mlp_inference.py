import keras.regularizers as regularizers
from keras.layers import Input, Dense
from keras.models import Model

DATASET_NUM_CLASSES = {
    'us8k': 10,
    'esc50': 50,
    'dcase2013': 10,
}


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
    l2_weight_decay = regularizers.l2(weight_decay)
    print('Input shape:', str(input_shape))
    inp = Input(shape=input_shape, dtype='float32')
    y = Dense(512, activation='relu', kernel_regularizer=l2_weight_decay)(inp)
    y = Dense(128, activation='relu', kernel_regularizer=l2_weight_decay)(y)
    y = Dense(num_classes, activation='softmax', kernel_regularizer=l2_weight_decay)(y)
    m = Model(inputs=inp, outputs=y)
    m.name = 'urban_sound_classifier'

    return m, inp, y

def run_mlp(X, model, weights_file):
    model.load_weights(weights_file)
    out_vec=model.predict(X)

    #Predict class index
    #pred =
    return out_vec.argmax()

