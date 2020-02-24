import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import csv
import json
import glob
import numpy as np
import resampy
import tensorflow as tf
import soundfile as sf
import librosa
from classifier.sonyc_ust.metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc
import oyaml as yaml

def get_output_from_tflite(cmsis_mels, tflite_model, input_index, output_index):
    
    predictions = []
    #predictions per frame   
    for idx in range(cmsis_mels.shape[0]):   #Ex of shape: (91, 64, 51)
        x = np.array(cmsis_mels[idx])[np.newaxis, :, :, np.newaxis].astype(np.float32)
        tflite_model.set_tensor(input_index, x)
        tflite_model.invoke()
        output = tflite_model.get_tensor(output_index)
        predictions.append(output)
    
    return predictions

def write_to_output(output_path, test_file_list, y_pred, taxonomy):
    
    coarse_fine_labels = [["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                           for fine_id, fine_label in fine_dict.items()]
                          for coarse_id, fine_dict in taxonomy['fine'].items()]
        
    full_fine_target_labels = [fine_label for fine_list in coarse_fine_labels
                               for fine_label in fine_list]
        
    coarse_target_labels = ["_".join([str(k), v])
                            for k, v in taxonomy['coarse'].items()]
        
    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + full_fine_target_labels + coarse_target_labels
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(test_file_list, y_pred):
            filename = os.path.basename(filename.replace('npz', 'wav'))
            row = [filename]

            # Add placeholder values for fine level
            row += [0.0 for _ in range(len(full_fine_target_labels))]
            # Add coarse level labels
            row += list(y)

            csvwriter.writerow(row)

def convert_pb_to_tflite(pb_model_dir=None, tflite_out_path=None):
    # A list of the names of the model's input tensors
    input_arrays = ['input_1']
    # A list of the names of the model's output tensors
    output_arrays = ['urban_sound_classifier/output/Sigmoid']

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(os.path.join(pb_model_dir, 'frozen_pipeline_cmsis_mels_quant.pb'),
                                                                    input_arrays = input_arrays,\
                                                                    output_arrays = output_arrays)

    #converter.inference_type = tf.int8
    #converter.inference_input_type = tf.int8
    input_arrays = converter.get_input_arrays()

    flatbuffer = converter.convert()

    with open(tflite_out_path, 'wb') as outfile:
        outfile.write(flatbuffer)
            
def process_cmsis_mels(file_list, taxonomy, output_path, model_path):
    
    y_pred_mean = []
    interpreter = tf.lite.Interpreter(model_path=model_path) 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape'][1:]
    output_shape = output_details[0]['shape'][1:]
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

#     print("== Input details ==")
#     print(interpreter.get_input_details()[0])
#     print("type:", input_details[0]['dtype'])
#     print("\n== Output details ==")
#     print(interpreter.get_output_details()[0])

    interpreter.allocate_tensors()
    
    for file in file_list:
        cmsis_mels = np.load(file)['db_mels']
        output = get_output_from_tflite(cmsis_mels, interpreter, input_index, output_index)

        #coarse classes of sonyc = 8
        pred_frame = np.array(output).reshape(-1, 8)
        y_pred_mean.append(pred_frame.mean(axis=0).tolist())
    
    write_to_output(output_path, file_list, y_pred_mean, taxonomy)
    assert os.path.exists(output_path)
    
def evaluate_all(prediction_path, annotation_path, yaml_path, mode='coarse'):
    
    metrics = {
        'coarse': {}
    }

    df_dict = evaluate(prediction_path,
                       annotation_path,
                       yaml_path,
                       mode)

    micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
    macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)    

     # Get index of first threshold that is at least 0.5
    thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).nonzero()[0][0]

    metrics[mode]["micro_auprc"] = micro_auprc
    metrics[mode]["micro_f1"] = eval_df["F"][thresh_0pt5_idx]
    metrics[mode]["macro_auprc"] = macro_auprc

    print("{} level evaluation:".format(mode.capitalize()))
    print("======================")
    print(" * Micro AUPRC:           {}".format(metrics[mode]["micro_auprc"]))
    print(" * Micro F1-score (@0.5): {}".format(metrics[mode]["micro_f1"]))
    print(" * Macro AUPRC:           {}".format(metrics[mode]["macro_auprc"]))
    print(" * Coarse Tag AUPRC:")

    metrics[mode]["class_auprc"] = {}
    for coarse_id, auprc in class_auprc.items():
        coarse_name = taxonomy['coarse'][int(coarse_id)]
        metrics[mode]["class_auprc"][coarse_name] = auprc
        print("      - {}: {}".format(coarse_name, auprc))       
        
if __name__=='__main__':
    
    model_dir = '/scratch/sk7898/quantization/pipeline_cmsis/'
    model_ts = '8k_size_quant'
    #model_ts = '20200224110114'
    MODEL_DIR = os.path.join(model_dir, model_ts)
    OUTPUT_DIR = os.path.join(MODEL_DIR, 'output/sonyc_ust/cmsis_val')
    tflite_path = '/scratch/sk7898/quantization/pipeline_cmsis_mels/tf_2_full_quantized_size_float32.tflite'
    #tflite_path = os.path.join(MODEL_DIR, 'qa_cmsis_model.tflite')
 
    DATA_DIR = '/beegfs/dr2915/sonyc_ust'
    annotation_path = os.path.join(DATA_DIR, 'annotations.csv')
    yaml_path = os.path.join(DATA_DIR, 'dcase-ust-taxonomy.yaml')
    test_data = glob.glob(os.path.join(DATA_DIR, 'db_mels/validate/*.npz'))
    prediction_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    output_path = os.path.join(OUTPUT_DIR, 'output_mean.csv')
    
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(yaml_path) as f:
        taxonomy = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(tflite_path):
        convert_pb_to_tflite(pb_model_dir=MODEL_DIR, tflite_out_path=tflite_path)
        
    process_cmsis_mels(test_data, taxonomy, output_path, model_path=tflite_path)
