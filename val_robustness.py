import os 
#import umap
import io
import random
import h5py
import numpy as np
from keras import models
from decrypt import read_encrypted_tar_audio_file
from kapre.time_frequency import Melspectrogram

def get_raw_windows_from_encrypted_audio(audio_path, tar_data, sample_rate=8000, clip_duration=10,
                                         decrypt_url='https://decrypt-sonyc.engineering.nyu.edu/decrypt',
                                         cacert_path='/home/jtc440/sonyc/decrypt/CA.pem',
                                         cert_path='/home/jtc440/sonyc/decrypt/jason_data.pem',
                                         key_path='/home/jtc440/sonyc/decrypt/sonyc_key.pem'):
    
    audio = read_encrypted_tar_audio_file(audio_path,
                                          enc_tar_filebuf=tar_data,
                                          sample_rate=sample_rate,
                                          url=decrypt_url,
                                          cacert=cacert_path,
                                          cert=cert_path,
                                          key=key_path)[0]
    if audio is None:
        return None

    audio_len = int(sample_rate * clip_duration)

    # Make sure audio is all consistent length (10 seconds)
    if len(audio) > audio_len:
        audio = audio[:audio_len]
    elif len(audio) < audio_len:
        pad_len = audio_len - len(audio)
        audio = np.pad(audio, (0, pad_len), mode='constant')

    # Return raw windows
    return get_audio_windows(audio, sr=sample_rate)


def get_audio_windows(audio, sr=8000, center=True, hop_size=0.5):
    """
    Similar to openl3.get_embedding(...)
    """

    def _center_audio(audio, frame_len):
        """Center audio so that first sample will occur in the middle of the first frame"""
        return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)

    def _pad_audio(audio, frame_len, hop_len):
        """Pad audio if necessary so that all samples are processed"""
        audio_len = audio.size
        if audio_len < frame_len:
            pad_length = frame_len - audio_len
        else:
            pad_length = int(np.ceil((audio_len - frame_len) / float(hop_len))) * hop_len \
                         - (audio_len - frame_len)

        if pad_length > 0:
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

        return audio

    # Check audio array dimension
    if audio.ndim > 2:
        raise AssertionError('Audio array can only be be 1D or 2D')
    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    audio_len = audio.size
    frame_len = sr
    hop_len = int(hop_size * sr)

    if audio_len < frame_len:
        warnings.warn('Duration of provided audio is shorter than window size (1 second). Audio will be padded.')

    if center:
        # Center audio
        audio = _center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = _pad_audio(audio, frame_len, hop_len)

    # Split audio into frames, copied from librosa.util.frame
    n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
    x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
                                        strides=(audio.itemsize, hop_len * audio.itemsize)).T

    # Add a channel dimension
    # x = x.reshape((x.shape[0], 1, x.shape[-1]))

    return x

def get_audio_feats(audio_dir, feats_dir, indices_dir, path, n_feats=100):
    feats_list = []
    audio_list = [] 
    h5_path = os.path.join(feats_dir, path)
    f = h5py.File(h5_path, 'r')
    num_datasets = f[list(f.keys())[0]].shape[0]
    
    for i in range(n_feats):
        dataset_index = np.random.randint(0, num_datasets)
        num_features = f[list(f.keys())[0]][dataset_index]['openl3'].shape[0]
        index = h5py.File(
            os.path.join(
                indices_dir, 
                os.path.basename(h5_path).split('.')[0]+'.sonyc_recording_index.h5'), 'r'
                )
        audio_file_name = os.path.join(audio_dir,
                                       index[list(index.keys())[0]][dataset_index]['day_hdf5_path'].decode()
                                       )
        row = index[list(index.keys())[0]][dataset_index]['day_h5_index']
        audio_file = h5py.File(audio_file_name, 'r')
        tar_data = io.BytesIO(audio_file['recordings'][row]['data'])
        raw_audio = get_raw_windows_from_encrypted_audio(audio_file_name, tar_data, sample_rate=8000)

        if raw_audio is None:  
            continue
        feature_index = random.sample(range(num_features), n_feats)
        feats_list.append(f[list(f.keys())[0]][dataset_index]['openl3'][feature_index])
        audio_list.append(raw_audio[feature_index])

    return np.array(audio_list).reshape(-1, 8000), np.array(feats_list).reshape(-1, 512)  


if __name__ == '__main__':
    model_dir = '/scratch/sk7898/embedding_approx_mse/models/sonyc/mse_original/8000_64_160_1024_fmax_None/20200909145902'
    weight_path = os.path.join(model_dir, 'model_best_valid_loss.h5')
    model = models.load_model(weight_path, custom_objects={'Melspectrogram': Melspectrogram})

    trfiles_dict = {}
    n_sensors = 7
    data_dir = '/scratch/sk7898/sonyc_30mil/train'
    parts = random.sample(range(15), n_sensors)

    for part in parts:
        splits = random.sample(range(2000), 100)
        trfiles_dict[part] = []
        for split in splits:
            fname = 'sonyc_ndata=2500000_part={}_split={}.h5'.format(part, split)
            if os.path.exists(os.path.join(data_dir, fname)):
                trfiles_dict[part].append(fname)


    for k, v in trfiles_dict.items():
        print('Part: {} Files: {}'.format(k, len(v)))

    mse_dict = {}
    embs_dict = {}
    n_feats = 100

    for sensor in trfiles_dict.keys():
        mse_error = 0
        embs_dict[sensor] = []
        train_files = trfiles_dict[sensor]
        for fname in train_files:
            idxs = sorted(random.sample(range(1024), n_feats))
            data_batch_path = os.path.join(data_dir, fname)
            data_blob = h5py.File(data_batch_path, 'r')
            audio_batch = np.array(data_blob['audio'][idxs])[:, np.newaxis, :]
            ref_embs = data_blob['l3_embedding'][idxs]
            pred_embs = model.predict(audio_batch)
            embs_dict[sensor].append(pred_embs)
            mse_error += np.mean((ref_embs - pred_embs)**2)
        mse_dict[sensor] = mse_error/len(train_files)

    for k, v in mse_dict.items():
        print('Sensor: {} Mean MSE: {}'.format(k, v))

    mse_test = {}
    n_feats = 10
    audio_dir = '/scratch/work/sonyc'
    indices_dir = '/scratch/work/sonyc/indices/2017'
    feats_dir = '/scratch/work/sonyc/features/openl3/2017'
    test_sensors = [
        'sonycnode-b827ebc6dcc6.sonyc_features_openl3.h5',
        'sonycnode-b827ebba613d.sonyc_features_openl3.h5',
        'sonycnode-b827ebad073b.sonyc_features_openl3.h5',
        'sonycnode-b827eb0fedda.sonyc_features_openl3.h5',
        'sonycnode-b827eb44506f.sonyc_features_openl3.h5',
        'sonycnode-b827eb122f0f.sonyc_features_openl3.h5',
        'sonycnode-b827eb0d8af7.sonyc_features_openl3.h5',
        'sonycnode-b827eb29eb77.sonyc_features_openl3.h5'
    ]
    for i, path in enumerate(test_sensors):
        embs_dict[30+i] = []
        audio_list, feats_list = get_audio_feats(audio_dir, feats_dir, indices_dir, path, n_feats=n_feats)
        test_error = 0
        for audio, ref_embs in zip(audio_list, feats_list):
            ref_embs = ref_embs.reshape(-1, 512)
            audio_batch = audio.reshape((1, 1, audio.shape[-1]))
            pred_embs = model.predict(audio_batch)
            embs_dict[30+i].append(pred_embs)
            test_error += np.mean((ref_embs - pred_embs)**2)
        mse_test[path] = test_error/audio_list.shape[0]

    for k, v in mse_test.items():
        print('Sensor: {} Mean MSE: {}'.format(k, v))

    embeddings = None
    clsses = []
    for k, v in embs_dict.items():
        e = np.array(embs_dict[k]).reshape(-1, 512)
        clsses += [k for i in range(e.shape[0])]
        if embeddings is None:
            embeddings = e
        else:
            embeddings = np.concatenate((embeddings, e), axis=0)

    with open('test_embs.npy', 'wb') as f:
        np.save(f, embeddings)
