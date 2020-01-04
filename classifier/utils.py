from keras.models import Model, load_model
import tensorflow as tf
import os

def create_sonyc_pipeline_model(l3_model_path, mlp_model_path):
    l3 = load_model(l3_model_path)
    mlp = load_model(mlp_model_path)

    # Pop MLP's input layer
    mlp.layers.pop(0)

    # Connect l3 with MLP


def convert_keras_classifier_to_TFLite(keras_model_path, out_path_name):
    """
    Converts Keras classifier model to TensorFlow Lite model (no quantization).
    Does not support custom layers too well. >=TensorFlow 1.14 only

    :param keras_model_path: Keras model path
    :param out_path_name: Output filename (Must have .tflite extension)
    """
    # Assert TensorFlow version is >= 1.14
    assert tf.__version__ >= '1.14.0', 'Must use TensorFlow 1.14.0 or greater for TFLite conversion.'

    # Assert output file is a .tflite file
    assert os.path.splitext(out_path_name)[-1]=='.tflite', 'Output path must be a .tflite file'

    converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
    tflite_model = converter.convert()

    # Create output path if it doesn't exist
    os.makedirs(os.path.dirname(out_path_name), exist_ok=True)

    open(out_path_name, 'wb').write(tflite_model)
