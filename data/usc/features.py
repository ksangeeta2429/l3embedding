import logging
import os
import librosa
import numpy as np
import scipy as sp
import soundfile as sf
import resampy
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from .vggish import vggish_input
#from .vggish import vggish_postprocess
#from .vggish import vggish_slim

LOGGER = logging.getLogger('cls-data-generation')
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


def one_hot(idx, n_classes=10):
    """
    Creates a one hot encoding vector

    Args:
        idx:  Class index
              (Type: int)

    Keyword Args:
        n_classes: Number of classes
                   (Type: int)

    Returns:
        one_hot_vector: One hot encoded vector
                        (Type: np.ndarray)
    """
    y = np.zeros((n_classes,))
    y[idx] = 1
    return y


def sample_non_overlap_file(X, chunk_size=10):
    def _chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    return np.array([chunk[0] for chunk in _chunks(X, chunk_size)])


def remove_data_overlap(data, chunk_size=10):
    X = []
    file_idxs = []
    new_start_idx = 0
    for start_idx, end_idx in data['file_idxs']:
        features = data['features'][start_idx:end_idx]
        features = sample_non_overlap_file(features, chunk_size=chunk_size)
        X.append(features)

        new_end_idx = new_start_idx + features.shape[0]
        file_idxs.append([new_start_idx, new_end_idx])
        new_start_idx = new_end_idx

    data['features'] = np.vstack(X)
    data['file_idxs'] = np.array(file_idxs)


def framewise_to_stats(data):
    X = []
    for start_idx, end_idx in data['file_idxs']:
        features = data['features'][start_idx:end_idx]
        X.append(compute_stats_features(features))

    data['features'] = np.vstack(X)

    idxs = np.arange(data['features'].shape[0])
    data['file_idxs'] = np.column_stack((idxs, idxs + 1))


def expand_framewise_labels(data):
    labels = []
    for y, (start_idx, end_idx) in zip(data['labels'], data['file_idxs']):
        num_frames = end_idx - start_idx
        labels.append(np.tile(y, num_frames))

    data['labels'] = np.concatenate(labels)


def preprocess_split_data(train_data, valid_data, test_data,
                          feature_mode='framewise', non_overlap=False,
                          non_overlap_chunk_size=10, use_min_max=False):
    # NOTE: This function mutates data so there aren't extra copies

    # Remove overlapping frames if no overlap
    if non_overlap:
        remove_data_overlap(train_data, chunk_size=non_overlap_chunk_size)
        if valid_data:
            remove_data_overlap(valid_data, chunk_size=non_overlap_chunk_size)
        remove_data_overlap(test_data, chunk_size=non_overlap_chunk_size)

    # Apply min max scaling to data
    min_max_scaler = MinMaxScaler()
    if use_min_max:
        train_data['features'] = min_max_scaler.fit_transform(
            train_data['features'])
        if valid_data:
            valid_data['features'] = min_max_scaler.transform(valid_data['features'])
        test_data['features'] = min_max_scaler.transform(test_data['features'])

    if feature_mode == 'framewise':
        # Expand training and validation labels to apply to each frame
        expand_framewise_labels(train_data)
        if valid_data:
            expand_framewise_labels(valid_data)
    elif feature_mode == 'stats':
        # Summarize frames in each file using summary statistics
        framewise_to_stats(train_data)
        if valid_data:
            framewise_to_stats(valid_data)
        framewise_to_stats(test_data)
    else:
        raise ValueError('Invalid feature mode: {}'.format(feature_mode))

    # Standardize features
    stdizer = StandardScaler()
    train_data['features'] = stdizer.fit_transform(train_data['features'])
    if valid_data:
        valid_data['features'] = stdizer.transform(valid_data['features'])
    test_data['features'] = stdizer.transform(test_data['features'])

    # Shuffle training data
    num_train_examples = len(train_data['labels'])
    shuffle_idxs = np.random.permutation(num_train_examples)
    reverse_shuffle_idxs = np.argsort(shuffle_idxs)
    train_data['features'] = train_data['features'][shuffle_idxs]
    train_data['labels'] = train_data['labels'][shuffle_idxs]
    train_data['file_idxs'] = [reverse_shuffle_idxs[slice(*pair)]
                               for pair in train_data['file_idxs']]

    return min_max_scaler, stdizer


def preprocess_features(data, min_max_scaler, stdizer,
                        feature_mode='framewise'):
    data['features'] = min_max_scaler.fit_transform(data['features'])
    if feature_mode == 'framewise':
        # Expand training and validation labels to apply to each frame
        expand_framewise_labels(data)
    elif feature_mode == 'stats':
        # Summarize frames in each file using summary statistics
        framewise_to_stats(data)
    else:
        raise ValueError('Invalid feature mode: {}'.format(feature_mode))
    data['features'] = stdizer.transform(data['features'])


def extract_vggish_embedding(audio_path, input_op_name='vggish/input_features',
                             output_op_name='vggish/embedding',
                             resources_dir=None, **params):
    # TODO: Make more efficient so we're not loading model every time we extract features
    fs = params.get('target_sample_rate', 16000)
    frame_win_sec = params.get('frame_win_sec', 0.96)
    audio_data = load_audio(audio_path, fs)

    # For some reason 0.96 doesn't work, padding to 0.975 empirically works
    frame_samples = int(np.ceil(fs * max(frame_win_sec, 0.975)))
    if audio_data.shape[0] < frame_samples:
        pad_length = frame_samples - audio_data.shape[0]
        # Use (roughly) symmetric padding
        left_pad = pad_length // 2
        right_pad= pad_length - left_pad
        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')

    if not resources_dir:
        resources_dir = os.path.join(os.path.dirname(__file__), '../../resources/vggish')

    pca_params_path = os.path.join(resources_dir, 'vggish_pca_params.npz')
    model_path = os.path.join(resources_dir, 'vggish_model.ckpt')


    examples_batch = vggish_input.waveform_to_examples(audio_data, fs, **params)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(pca_params_path, **params)

    # If needed, prepare a record writer to store the postprocessed embeddings.
    #writer = tf.python_io.TFRecordWriter(
    #    FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

    input_tensor_name = input_op_name + ':0'
    output_tensor_name = output_op_name + ':0'

    with tf.Graph().as_default(), tf.Session() as sess:
      # Define the model in inference mode, load the checkpoint, and
      # locate input and output tensors.
      vggish_slim.define_vggish_slim(training=False, **params)
      vggish_slim.load_vggish_slim_checkpoint(sess, model_path, **params)
      features_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
      embedding_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

      # Run inference and postprocessing.
      [embedding_batch] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: examples_batch})
      postprocessed_batch = pproc.postprocess(embedding_batch, **params)

      # Write the postprocessed embeddings as a SequenceExample, in a similar
      # format as the features released in AudioSet. Each row of the batch of
      # embeddings corresponds to roughly a second of audio (96 10ms frames), and
      # the rows are written as a sequence of bytes-valued features, where each
      # feature value contains the 128 bytes of the whitened quantized embedding.

    return postprocessed_batch.astype(np.float32)


def get_vggish_frames_uniform(audio_path, hop_size=0.1):
    """
    Get vggish embedding features for each frame in the given audio file

    Args:
        audio: Audio data or path to audio file
               (Type: np.ndarray or str)

    Keyword Args:
        hop_size: Hop size in seconds
                  (Type: float)

    Returns:
        features:  Array of embedding vectors
                   (Type: np.ndarray)
    """
    return extract_vggish_embedding(audio_path, frame_hop_sec=hop_size)


def compute_stats_features(embeddings):
    # Compute statistics on the time series of embeddings
    minimum = np.min(embeddings, axis=0)
    maximum = np.max(embeddings, axis=0)
    median = np.median(embeddings, axis=0)
    mean = np.mean(embeddings, axis=0)
    var = np.var(embeddings, axis=0)
    skewness = sp.stats.skew(embeddings, axis=0)
    kurtosis = sp.stats.kurtosis(embeddings, axis=0)

    return np.concatenate((minimum, maximum, median, mean, var, skewness, kurtosis))


def amplitude_to_db(S, amin=1e-10, dynamic_range=80.0):
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)
    ref_value = power.max()

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= log_spec.max()

    log_spec = np.maximum(log_spec, -dynamic_range)

    return log_spec


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
     
    #print("== Input details ==")
    #print(interpreter.get_input_details()[0])
    #print("type:", input_details[0]['dtype'])
    #print("\n== Output details ==")
    #print(interpreter.get_output_details()[0])
    
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

    return predictions

def get_l3_frames_uniform(audio, l3embedding_model, n_fft=2048, n_mels=256,
                          mel_hop_length=242, hop_size=0.1, sr=48000, with_melSpec=None, fmax=None, **kwargs):
    """
    Get L3 embedding for each frame in the given audio file

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

    # Divide into overlapping 1 second frames
    frames = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T

    if with_melSpec:
        #print("Melspectrogram is part of the weight file")
        # Add a channel dimension
        X = frames.reshape((frames.shape[0], 1, frames.shape[-1]))
    
    else:
        X = []
        for frame in frames:
            S = np.abs(librosa.core.stft(frame, n_fft=n_fft, hop_length=mel_hop_length,
                                         window='hann', center=True,
                                         pad_mode='constant'))
            S = librosa.feature.melspectrogram(sr=sr, S=S, n_mels=n_mels, fmax=fmax,
                                           power=1.0, htk=True)
            S = amplitude_to_db(np.array(S))
            X.append(S)

        X = np.array(X)[:, :, :, np.newaxis]

    # Get the L3 embedding for each frame
    l3embedding = l3embedding_model.predict(X)

    return l3embedding, frames


def compute_file_features(path, feature_type, l3embedding_model=None, model_type='keras', **feature_args):
    if feature_type == 'l3':
        if not l3embedding_model:
            err_msg = 'Must provide L3 embedding model to use {} features'
            raise ValueError(err_msg.format(feature_type))
        #hop_size = feature_args.get('hop_size', 0.1)
        #samp_rate = feature_args.get('samp_rate', 48000)
        audio = None
        if model_type == 'keras':
            # Handling CMSIS mel inputs
            if os.path.splitext(path)[-1]=='.npz':
                x = np.load(path)
                file_features = l3embedding_model.predict(x['db_mels'][:,:,:,np.newaxis])
            else:
                file_features, audio = get_l3_frames_uniform(path, l3embedding_model, **feature_args)
        elif model_type == 'tflite':
            #TODO: Add logic for '.npz' (CMSIS mel input) handling
            file_features = get_l3_frames_uniform_tflite(path, interpreter=l3embedding_model, **feature_args)
        else:
            raise ValueError('Model type not supported!')
            
    else:
        raise ValueError('Invalid feature type: {}'.format(feature_type))
        #hop_size=hop_size, sr=samp_rate)
    #elif feature_type == 'vggish':
    #    hop_size = feature_args.get('hop_size', 0.1)
    #    file_features = get_vggish_frames_uniform(path, hop_size=hop_size)

    return file_features, audio


def flatten_file_frames(X, y):
    """
    For data organized by file, flattens the frame features for training data
    and duplicates the file labels for each frame

    Args:
        X: Framewise feature data, which consists of lists of frame features
           for each file
                 (Type: np.ndarray[list[np.ndarray]])
        y: Label data for each file
                 (Type: np.ndarray)

    Returns:
        X_flattened: Flatten matrix of frame features
                     (Type: np.ndarray)
        y_flattened: Label data for each frame
                     (Type: np.ndarray)

    """
    if X.ndim == 1:
        # In this case the number of frames per file varies, which makes the
        # numpy array store lists for each file
        num_frames_per_file = []
        X_flatten = []
        for X_file in X:
            num_frames_per_file.append(len(X_file))
            X_flatten += X_file
        X_flatten = np.array(X_flatten)
    else:
        # In this case the number of frames per file is the same, which means
        # all of the data is in a unified numpy array
        X_shape = X.shape
        num_files, num_frames_per_file = X_shape[0], X_shape[1]
        new_shape = (num_files * num_frames_per_file,) + X_shape[2:]
        X_flatten = X.reshape(new_shape)

    # Repeat the labels for each frame
    y_flatten = np.repeat(y, num_frames_per_file)

    return X_flatten, y_flatten
