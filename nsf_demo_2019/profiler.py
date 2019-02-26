from execution_engine.load_embedding_pi import load_embedding
import time

def run_cnn_l3_melspec2_embedding():
    weights_path='cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
    model_type='cnn_L3_melspec2'
    embedding_type='audio'
    pooling_type='short'

    embedding=load_embedding(weights_path, model_type, embedding_type, pooling_type)
    return embedding

start = time.time()
emb = run_cnn_l3_melspec2_embedding()
done = time.time()
elapsed = done - start
print(elapsed)