from l3embedding.model import load_embedding
from data.usc.us8k import generate_us8k_file_data, load_us8k_metadata
from classifier.mlp_inference import run_mlp, construct_mlp_model
import random
import numpy as np
import os
import glob
import time

US8K_CLASSES = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

# Load embedding
weights_path = 'models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
model_type = 'cnn_L3_melspec2'
embedding_type = 'audio'
pooling_type = 'short'

print('Loading embedding...')
l3embedding_model = load_embedding(weights_path, model_type, embedding_type, pooling_type)

# Set up model
print('Loading MLP...')
m_class, inp, out = construct_mlp_model((512,))

# Featurizer and classifier params
fold_idx = 5
features = 'l3'
metadata_path = 'UrbanSound8K/metadata/UrbanSound8K.csv'
data_dir = 'UrbanSound8K/audio'
dataset_output_dir = 'embeddings'
hop_size = 0.1
classifier_weights_path = 'models/us8k-music-melspec2-512emb-model/model.h5'

# Start
metadata = load_us8k_metadata(metadata_path)

# 0-indexed fold id
fold_idx = fold_idx - 1

# Set random seed
random_state = 12345678
random_state = random_state + fold_idx
random.seed(random_state)
np.random.seed(random_state)

# Audio fold directory
audio_fold_dir = os.path.join(data_dir, "fold{}".format(fold_idx+1))

for idx, (fname, example_metadata) in enumerate(metadata[fold_idx].items()):
    variants = [x for x in glob.glob(os.path.join(audio_fold_dir,
                                                  '**', os.path.splitext(fname)[0] + '[!0-9]*[wm][ap][v3]'),
                                     recursive=True) if os.path.isfile(x) and not x.endswith('.jams')]

    for var_idx, var_path in enumerate(variants):
        audio_dir = os.path.dirname(var_path)
        var_fname = os.path.basename(var_path)

        print('Profiling featurization + classification...\n')
        # Keep iterating
        for i in range(10):
            start = time.time()
            X, y = generate_us8k_file_data(var_fname, example_metadata, audio_dir, features=features,
                                l3embedding_model=l3embedding_model, hop_size=hop_size)
            y_hat = run_mlp(X, m_class, classifier_weights_path)
            done = time.time()
            elapsed = done - start
            print('Time: '+ str(elapsed) + ' s')

        # Print output
        if y == y_hat:
            qualifier = 'correctly'
        else:
            qualifier = 'incorrectly'

        print('-----------------------------------')
        print('The audio clip with true label ' + US8K_CLASSES[y] + ' has been ' + qualifier + ' classified as ' + US8K_CLASSES[y_hat])
        print('-----------------------------------')
