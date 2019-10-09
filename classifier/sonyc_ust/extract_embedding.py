import argparse
import gzip
import os
import numpy as np
import pandas as pd
import librosa
import resampy
import soundfile as sf
import warnings
from tqdm import tqdm

import tensorflow as tf


from vggish import vggish_input
from vggish import vggish_postprocess
from vggish import vggish_slim


def make_extract_vggish_embedding(frame_duration, hop_duration, input_op_name='vggish/input_features',
                                  output_op_name='vggish/embedding', embedding_size=128, resources_dir=None):
    """
    Creates a coroutine generator for extracting and saving VGGish embeddings

    Parameters
    ----------
    frame_duration
    hop_duration
    input_op_name
    output_op_name
    embedding_size
    resources_dir

    Returns
    -------
    coroutine

    """
    if frame_duration is None:
        frame_duration = 0.96

    if hop_duration is None:
        hop_duration = 0.96

    params = {
        'frame_win_sec': frame_duration,
        'frame_hop_sec': hop_duration,
        'embedding_size': embedding_size
    }

    if not resources_dir:
        resources_dir = os.path.join(os.path.dirname(__file__), 'vggish/resources')

    pca_params_path = os.path.join(resources_dir, 'vggish_pca_params.npz')
    model_path = os.path.join(resources_dir, 'vggish_model.ckpt')

    try:
        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False, **params)
            vggish_slim.load_vggish_slim_checkpoint(sess, model_path, **params)

            while True:
                # We use a coroutine to more easily keep open the Tensorflow contexts
                # without having to constantly reload the model
                audio_path, output_path = (yield)

                if os.path.exists(output_path):
                    continue

                try:
                    examples_batch = vggish_input.wavfile_to_examples(audio_path, **params)
                except ValueError:
                    print("Error opening {}. Skipping...".format(audio_path))
                    continue

                # Prepare a postprocessor to munge the model embeddings.
                pproc = vggish_postprocess.Postprocessor(pca_params_path, **params)

                input_tensor_name = input_op_name + ':0'
                output_tensor_name = output_op_name + ':0'

                features_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
                embedding_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

                # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})

                emb = pproc.postprocess(embedding_batch, **params).astype(np.float32)

                with gzip.open(output_path, 'wb') as f:
                    emb.dump(f)

    except GeneratorExit:
        pass


def extract_embeddings_vggish(annotation_path, dataset_dir, output_dir,
                              vggish_resource_dir, frame_duration=None,
                              hop_duration=None, progress=True,
                              vggish_embedding_size=128):
    """
    Extract embeddings for files annotated in the SONYC annotation file and save them to disk.

    Parameters
    ----------
    annotation_path
    dataset_dir
    output_dir
    vggish_resource_dir
    frame_duration
    hop_duration
    progress
    vggish_embedding_size

    Returns
    -------

    """

    print("* Loading annotations.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

    extract_vggish_embedding = make_extract_vggish_embedding(frame_duration, hop_duration,
        input_op_name='vggish/input_features', output_op_name='vggish/embedding',
        resources_dir=vggish_resource_dir, embedding_size=vggish_embedding_size)
    # Start coroutine
    next(extract_vggish_embedding)

    out_dir = os.path.join(output_dir, 'vggish')
    os.makedirs(out_dir, exist_ok=True)

    df = annotation_data[['split', 'audio_filename']].drop_duplicates()
    row_iter = df.iterrows()

    if progress:
        row_iter = tqdm(row_iter, total=len(df))

    print("* Extracting embeddings.")
    for _, row in row_iter:
        filename = row['audio_filename']
        split_str = row['split']
        audio_path = os.path.join(dataset_dir, split_str, filename)
        emb_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.npy.gz')
        extract_vggish_embedding.send((audio_path, emb_path))

    extract_vggish_embedding.close()


def _save_l3_embedding(filepath, model, output_dir, center=True, hop_size=0.1):
    """
    Computes and returns L3 embedding for given audio data

    Parameters
    ----------
    filepath : str
        Path to audio file
    model : keras.models.Model
        Embedding model
    output_dir : str
        Output directory
    center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.
    hop_size : float
        Hop size in seconds.

    Returns
    -------
        embedding : np.ndarray [shape=(T, D)]
            Array of embeddings for each window.
        timestamps : np.ndarray [shape=(T,)]
            Array of timestamps corresponding to each embedding in the output.

    """
    import openl3

    audio, sr = sf.read(filepath)
    output_path = openl3.core.get_output_path(filepath, ".npz", output_dir=output_dir)

    if audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    # Resample if necessary
    if sr != openl3.core.TARGET_SR:
        audio = resampy.resample(audio, sr_orig=sr, sr_new=openl3.core.TARGET_SR, filter='kaiser_best')

    audio_len = audio.size
    frame_len = openl3.core.TARGET_SR
    hop_len = int(hop_size * openl3.core.TARGET_SR)

    if center:
        # Center audio
        audio = openl3.core._center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = openl3.core._pad_audio(audio, frame_len, hop_len)

    # Split audio into frames, copied from librosa.util.frame
    n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
    x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
        strides=(audio.itemsize, hop_len * audio.itemsize)).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))

    # Get embedding and timestamps
    embedding = model.predict(x, verbose=0)

    ts = np.arange(embedding.shape[0]) * hop_size

    np.savez(output_path, embedding=embedding, timestamps=ts)


def get_l3_embedding_model(input_repr, content_type, embedding_size, load_weights=True):
    """
    Returns a model with the given characteristics. Loads the model
    if the model has not been loaded yet.

    Parameters
    ----------
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.
    embedding_size : 6144 or 512
        Embedding dimensionality.
    load_weights : bool
        If True, load trained model weights.

    Returns
    -------
    model : keras.models.Model
        Model object.
    """
    from openl3.models import MODELS, POOLINGS, get_embedding_model_path
    from keras.layers import (
        Input, Conv2D, BatchNormalization, MaxPooling2D,
        Flatten, Activation, Lambda
    )
    from keras.models import Model


    # Construct embedding model and load model weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MODELS[input_repr]()

    if load_weights:
        m.load_weights(get_embedding_model_path(input_repr, content_type))

    # Pooling for final output embedding size
    pool_size = POOLINGS[input_repr][embedding_size]
    y_a = MaxPooling2D(pool_size=pool_size, padding='same')(m.output)
    y_a = Flatten()(y_a)
    m = Model(inputs=m.input, outputs=y_a)
    return m


def extract_embeddings_l3(annotation_path, dataset_dir, output_dir, hop_duration=None, progress=True,
                          input_repr='mel256', content_type='music', embedding_size=512,
                          load_l3_weights=True):
    """
    Extract embeddings for files annotated in the SONYC annotation file and save them to disk.

    Parameters
    ----------
    annotation_path
    dataset_dir
    output_dir
    vggish_resource_dir
    frame_duration
    hop_duration
    progress

    Returns
    -------

    """
    import openl3

    if hop_duration is None:
        hop_duration = 1.0

    print("* Loading annotations.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

    out_dir = os.path.join(output_dir, 'l3-{}-{}-{}'.format(input_repr, content_type, embedding_size))
    os.makedirs(out_dir, exist_ok=True)

    df = annotation_data[['split', 'audio_filename']].drop_duplicates()
    row_iter = df.iterrows()

    if progress:
        row_iter = tqdm(row_iter, total=len(df))

    # Load model
    model = get_l3_embedding_model(input_repr, content_type, embedding_size,
                                   load_weights=load_l3_weights)

    print("* Extracting embeddings.")
    for _, row in row_iter:
        filename = row['audio_filename']
        split_str = row['split']
        audio_path = os.path.join(dataset_dir, split_str, filename)
        out_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.npz')
        if not os.path.exists(out_path):
            _save_l3_embedding(audio_path, model, out_dir, center=True,
                               hop_size=hop_duration)


def extract_embeddings_mfcc(annotation_path, dataset_dir, output_dir, hop_duration=None, progress=True):
    """
    Extract embeddings for files annotated in the SONYC annotation file and save them to disk.

    Parameters
    ----------
    annotation_path
    dataset_dir
    output_dir
    vggish_resource_dir
    frame_duration
    hop_duration
    progress
    vggish_embedding_size

    Returns
    -------

    """
    if hop_duration is None:
        # To make more comparable to deep embeddings, make window size close to 1 second
        hop_length = 16384
    else:
        hop_length = int(22050 * hop_duration)

    print("* Loading annotations.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

    out_dir = os.path.join(output_dir, 'mfcc')
    os.makedirs(out_dir, exist_ok=True)

    df = annotation_data[['split', 'audio_filename']].drop_duplicates()
    row_iter = df.iterrows()

    if progress:
        row_iter = tqdm(row_iter, total=len(df))

    print("* Extracting embeddings.")
    for _, row in row_iter:
        filename = row['audio_filename']
        split_str = row['split']
        audio_path = os.path.join(dataset_dir, split_str, filename)
        audio, sr = librosa.load(audio_path)
        emb_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.npy.gz')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=hop_length, hop_size=hop_length, n_mels=15)
        # Get rid of DC component
        mfcc = mfcc.T[:, 1:]

        with gzip.open(emb_path, 'wb') as f:
            np.save(f, mfcc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("dataset_dir")
    parser.add_argument("output_dir")
    parser.add_argument("embedding_type", choices=["vggish", "l3", "mfcc"])

    parser.add_argument("--vggish_resource_dir")
    parser.add_argument("--vggish_embedding_size", type=int, default=128)

    parser.add_argument("--l3_random", action="store_true")
    parser.add_argument("--l3_input_repr", type=str,
                        choices=["linear", "mel128", "mel256"], default="mel256")
    parser.add_argument("--l3_content_type", type=str,
                        choices=["music", "env"], default="music")
    parser.add_argument("--l3_embedding_size", type=int,
                        choices=[512, 6144], default=6144)

    parser.add_argument("--frame_duration", type=float)
    parser.add_argument("--hop_duration", type=float)
    parser.add_argument("--progress", action="store_const", const=True, default=False)

    args = parser.parse_args()

    if args.embedding_type == "vggish":
        extract_embeddings_vggish(annotation_path=args.annotation_path,
                                  dataset_dir=args.dataset_dir,
                                  output_dir=args.output_dir,
                                  vggish_resource_dir=args.vggish_resource_dir,
                                  vggish_embedding_size=args.vggish_embedding_size,
                                  frame_duration=args.frame_duration,
                                  hop_duration=args.hop_duration,
                                  progress=args.progress)
    elif args.embedding_type == "l3":
        extract_embeddings_l3(annotation_path=args.annotation_path,
                              dataset_dir=args.dataset_dir,
                              output_dir=args.output_dir,
                              hop_duration=args.hop_duration,
                              progress=args.progress,
                              load_l3_weights=(not args.l3_random),
                              input_repr=args.l3_input_repr,
                              content_type=args.l3_content_type,
                              embedding_size=args.l3_embedding_size)
    elif args.embedding_type == "mfcc":
        extract_embeddings_mfcc(annotation_path=args.annotation_path,
                               dataset_dir=args.dataset_dir,
                               output_dir=args.output_dir,
                               hop_duration=args.hop_duration,
                               progress=args.progress)
