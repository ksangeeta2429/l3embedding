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
    import edgel3

    audio, sr = sf.read(filepath)
    output_path = edgel3.core.get_output_path(filepath, ".npz", output_dir=output_dir)

    if audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    # Resample if necessary
    if sr != edgel3.core.TARGET_SR:
        audio = resampy.resample(audio, sr_orig=sr, sr_new=edgel3.core.TARGET_SR, filter='kaiser_best')

    audio_len = audio.size
    frame_len = edgel3.core.TARGET_SR
    hop_len = int(hop_size * edgel3.core.TARGET_SR)

    if center:
        # Center audio
        audio = edgel3.core._center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = edgel3.core._pad_audio(audio, frame_len, hop_len)

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

def extract_embeddings_l3(annotation_path, dataset_dir, output_dir, l3embedding_model, hop_duration=None, progress=True):
    """
    Extract embeddings for files annotated in the SONYC annotation file and save them to disk.
    Parameters
    ----------
    annotation_path: str
        Path to SONYC_UST annotation csv file
    dataset_dir: str
        Path to SONYC_UST dataset
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    l3embedding_model : keras.models.Model
        Model Object
    hop_duration : float
        Hop size in seconds.
    progress : bool

    Returns
    -------
    """

    import edgel3

    if hop_duration is None:
        hop_duration = 1.0

    print("* Loading annotations.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

    #out_dir = os.path.join(output_dir, 'l3-{}-{}-{}'.format(input_repr, content_type, embedding_size))
    os.makedirs(out_dir, exist_ok=True)

    df = annotation_data[['split', 'audio_filename']].drop_duplicates()
    row_iter = df.iterrows()

    if progress:
        row_iter = tqdm(row_iter, total=len(df))

    # Load model
    #model = get_l3_embedding_model(input_repr, content_type, embedding_size,
                                   #load_weights=load_l3_weights)

    print("* Extracting embeddings.")
    for _, row in row_iter:
        filename = row['audio_filename']
        split_str = row['split']
        audio_path = os.path.join(dataset_dir, split_str, filename)
        out_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.npz')
        if not os.path.exists(out_path):
            _save_l3_embedding(audio_path, model, out_dir, center=True,
                               hop_size=hop_duration)
