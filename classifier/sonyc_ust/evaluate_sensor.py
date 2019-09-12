import argparse
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Evaluation script for Urban Sound Tagging task for the DCASE 2019 Challenge.

        See `metrics.py` for more information about the metrics.
        """)

    parser.add_argument('output_dir', type=str,
                        help='Path output directory.')
    parser.add_argument('aggregation_mode', type=str,
                        help='Aggregation method.')
    parser.add_argument('splits_path', type=str,
                        help='Path to splits / dataset annotation CSV file.')

    args = parser.parse_args()

    metrics = {
        'accuracy': [],
        'micro_precision': [],
        'micro_recall': [],
        'micro_f1': [],
        'macro_precision': [],
        'macro_recall': [],
        'macro_f1': [],
    }

    for valid_split_id in os.listdir(args.output_dir):
        prediction_path = os.path.join(args.output_dir, valid_split_id, 'output_{}.csv'.format(args.aggregation_mode))
        if not os.path.exists(prediction_path):
            continue

        gt_df = pd.read_csv(args.splits_path)
        sensor_ids = list(np.sort(gt_df['sensor_id'].unique()))
        gt_df = gt_df[gt_df['split'] == int(valid_split_id)]
        pred_df = pd.read_csv(prediction_path)

        np.testing.assert_array_equal(gt_df['filename'].values, pred_df['audio_filename'].values)

        target_list = gt_df['sensor_id'].values
        _target_list = np.zeros([len(target_list), len(sensor_ids)])
        for i in range(target_list.shape[0]):
            _target_list[i, sensor_ids.index(target_list[i])] = 1
        y = np.argmax(_target_list, axis=1)
        pred_y = np.argmax(pred_df.loc[:,'class0':].values, axis=1)

        metrics['accuracy'].append(accuracy_score(y, pred_y))
        metrics['micro_precision'].append(precision_score(y, pred_y, average='micro'))
        metrics['micro_recall'].append(recall_score(y, pred_y, average='micro'))
        metrics['micro_f1'].append(f1_score(y, pred_y, average='micro'))
        metrics['macro_precision'].append(precision_score(y, pred_y, average='macro'))
        metrics['macro_recall'].append(recall_score(y, pred_y, average='macro'))
        metrics['macro_f1'].append(f1_score(y, pred_y, average='macro'))

    accuracy = metrics['accuracy']
    micro_precision = metrics['micro_precision']
    micro_recall = metrics['micro_recall']
    micro_f1 = metrics['micro_f1']
    macro_precision = metrics['macro_precision']
    macro_recall = metrics['macro_recall']
    macro_f1 = metrics['macro_f1']

    average_metrics = {
        'accuracy': {
            'mean': float(np.mean(accuracy)),
            'stdev': float(np.std(accuracy)) },
        'micro_f1': {
            'mean': float(np.mean(micro_f1)),
            'stdev': float(np.std(micro_f1)) },
        'macro_f1': {
            'mean': float(np.mean(macro_f1)),
            'stdev': float(np.std(macro_f1)) },
    }

    print("======================")
    print(" * Accuracy:           {} (stdev {})".format(average_metrics['accuracy']['mean'],
                                                        average_metrics['accuracy']['stdev']))
    print(" * Micro F1-score (@0.5): {} (stdev {})".format(average_metrics['micro_f1']['mean'],
                                                           average_metrics['micro_f1']['stdev']))
    print(" * Macro F1-score:           {} (stdev {})".format(average_metrics['macro_f1']['mean'],
                                                              average_metrics['macro_f1']['stdev']))

    total_metrics = {
        'folds': metrics,
        'average': average_metrics
    }

    eval_path = os.path.join(args.output_dir, 'evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(total_metrics, f)
