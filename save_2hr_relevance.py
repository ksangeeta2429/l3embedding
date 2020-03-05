import os
import h5py
import numpy as np
from sklearn import preprocessing
from sampling_utils import *

def write_to_h5(scores, output_path):
    
    if os.path.isfile(output_path):
        print("{} already exists. Deleting.".format(output_path))
        os.remove(output_path)
        
    num_embeddings = 20
    with h5py.File(output_path) as h5:
        d = h5.create_dataset('recording_index',
                              (len(scores),),
                              dtype=[('relevance_2hr_vector', 'f4', (num_embeddings,))],)
            
        for idx, score_vec in enumerate(scores):
            d[idx, 'relevance_2hr_vector'] = tuple(score_vec)
            
        h5.close()
        
    print('Completed writing to {}'.format(output_path))
    
def get_relevance_scores_files(feats_dir, indices_dir, output_dir):
    
    for file in os.listdir(feats_dir):
        feats_path = os.path.join(feats_dir, file)
        indices_path = os.path.join(indices_dir, os.path.basename(file).replace('features_openl3', 'recording_index'))
        output_path = os.path.join(output_dir, os.path.basename(file).replace('features_openl3', 'relevance_2hr'))
        
        indices = h5py.File(indices_path)
        spl_vecs = indices['recording_index']['spl_vector']

        blob = h5py.File(feats_path)
        ts = blob['openl3']['timestamp']
        
        if ts.shape[0] > 0:
            scores = get_relevance_scores(ts, spl_vecs)
            write_to_h5(scores, output_path)
            
if __name__=='__main__':
    feats_dir = '/beegfs/work/sonyc/features/openl3/2017/'
    indices_dir = '/beegfs/work/sonyc/indices/2017/'
    output_dir = '/beegfs/sk7898/sonyc/relevance/2017'
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    get_relevance_scores_files(feats_dir, indices_dir, output_dir)
    