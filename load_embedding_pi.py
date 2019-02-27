from l3embedding.model import load_embedding
from data.usc.us8k import generate_us8k_fold_data

# Load embedding
weights_path = 'cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
model_type = 'cnn_L3_melspec2'
embedding_type = 'audio'
pooling_type = 'short'

print('Loading embedding...')
l3embedding_model = load_embedding(weights_path, model_type, embedding_type, pooling_type)

# Featurization
fold_num = 5
features = 'l3'
metadata_path = 'UrbanSound8K/metadata/UrbanSound8K.csv'
data_dir = 'UrbanSound8K/audio'
dataset_output_dir = 'embeddings'
hop_size = 0.1

print('Computing features...')
for i in range(100):
    generate_us8k_fold_data(metadata_path, data_dir, fold_num-1, dataset_output_dir,
                                    l3embedding_model=l3embedding_model,
                                    features=features, hop_size=hop_size)

