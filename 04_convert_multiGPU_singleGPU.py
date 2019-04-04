import argparse
import keras
from keras.models import Model
import os
import glob
from l3embedding.model import load_model

def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Convert multi-GPU model to single-GPU model and save the model instead of just the weights')

    parser.add_argument('-wdir',
                        '--multiGPU-weight-dir',
                        dest='multiGPU_weight_dir',
                        action='store',
                        type=str,
                        help='Path to L3 embedding multi-GPU model weights file directory')

    parser.add_argument('-srate',
                        '--samp-rate',
                        dest='samp_rate',
                        action='store',
                        type=int,
                        default=48000,
                        help='Sampling rate')

    parser.add_argument('-nmels',
                        '--num-mels',
                        dest='n_mels',
                        action='store',
                        type=int,
                        default=256,
                        help='Number of mel filters')


    parser.add_argument('-lhop',
                        '--hop-length',
                        dest='n_hop',
                        action='store',
                        type=int,
                        default=242,
                        help='Melspec hop length in samples')

    parser.add_argument('-ndft',
                        '--num-dft',
                        dest='n_dft',
                        action='store',
                        type=int,
                        default=2048,
                        help='DFT size')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where single-GPU models will be stored')

    return vars(parser.parse_args())



if __name__ == '__main__':
    args = parse_arguments()

    weight_dir = args['multiGPU_weight_dir']
    output_dir = args['output_dir']
    samp_rate = args['samp_rate']
    n_mels = args['n_mels']
    n_hop = args['n_hop']
    n_dft = args['n_dft']

    '''
    model_dirs  = ['/scratch/sk7898/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190303194756',
                   '/scratch/sk7898/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190303210054',
                   '/scratch/dr2915/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190304024942',
                   '/scratch/mc6591/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190304160730',
                   '/scratch/mc6591/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190304160731',
                   '/scratch/dr2915/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190304165251',
                   '/scratch/cm3580/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190305211344',
                   '/scratch/cm3580/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190306112819',
                   '/scratch/sk7898/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190310204758',
                   '/scratch/mc6591/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190311170637'
                   ]
               

    kwargs = ['n_mels=256, n_hop=484,  n_dft=2048, samp_rate=48000',
              'n_mels=256, n_hop=968,  n_dft=2048, samp_rate=48000',
              'n_mels=64,  n_hop=968,  n_dft=2048, samp_rate=48000',
              'n_mels=128, n_hop=968,  n_dft=2048, samp_rate=48000',
              'n_mels=64,  n_hop=242,  n_dft=2048, samp_rate=48000',
              'n_mels=64,  n_hop=1936, n_dft=2048, samp_rate=48000',
              'n_mels=64,  n_hop=484,  n_dft=1024, samp_rate=16000',
              'n_mels=32,  n_hop=484,  n_dft=1024, samp_rate=16000',
              'n_mels=256, n_hop=1936, n_dft=2048, samp_rate=48000',
              'n_mels=128, n_hop=1936, n_dft=2048, samp_rate=48000'
              ]
    '''

    #output_dir = '/scratch/sk7898/l3pruning/embedding/fixed/'

    model_id = weight_dir.split('/')[-1]
    mt = os.path.basename(os.path.dirname(weight_dir))
    input_repr = str(samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)

    #model_id = md.split('/', 4)[-1]
    #out_model_dir = os.path.join(output_dir, model_id)
    #if not os.path.isdir(out_model_dir):
    #    os.makedirs(out_model_dir)
        
    weight_file = glob.glob(os.path.join(weight_dir, '*best_valid_accuracy*'))[0]
    #print('Model_id: ', model_id)
    #print('Model Type: ',mt)
    #print('Weight File: ', weight_file)

    #output_weight_file = os.path.join(output_dir, os.path.basename(wf))

    # Load and convert model back to 1 gpu
    print("Loading model.......................")
    m, inputs, outputs = load_model(weight_file, mt, src_num_gpus=4, tgt_num_gpus=1, return_io=True, n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, asr=samp_rate)
    _, x_a = inputs
    audio_model_output = m.get_layer('audio_model').get_layer('audio_embedding_layer').output
    audio_embed_model = Model(inputs=x_a, outputs=audio_model_output)
    audio_output_path = os.path.join(output_dir, 'l3_audio_{}_{}.h5'.format(model_id, input_repr))

    # Save converted model back to disk
    audio_embed_model.save(audio_output_path)
    
    print('Audio model saved: ', audio_output_path)
