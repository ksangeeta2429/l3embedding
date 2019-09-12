import argparse
import json
import os
import oyaml as yaml
import numpy as np
from collections import defaultdict
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Evaluation script for Urban Sound Tagging task for the DCASE 2019 Challenge.

        See `metrics.py` for more information about the metrics.
        """)

    parser.add_argument('output_dir', type=str,
                        help='Path output directory.')
    parser.add_argument('annotation_path', type=str,
                        help='Path to dataset annotation CSV file.')
    parser.add_argument('yaml_path', type=str,
                        help='Path to dataset taxonomy YAML file.')

    args = parser.parse_args()

    with open(args.yaml_path) as f:
        taxonomy = yaml.load(f)

    metrics = {
        'fine': {
            'micro_auprc': [],
            'micro_f1': [],
            'macro_auprc': [],
            'class_auprc': defaultdict(list),
        },
        'coarse': {
            'micro_auprc': [],
            'micro_f1': [],
            'macro_auprc': [],
            'class_auprc': {},
        },
    }

    for valid_sensor_id in os.listdir(args.output_dir):
        prediction_path = os.path.join(args.output_dir, valid_sensor_id, 'output.csv')
        if not os.path.exists(prediction_path):
            continue

        for mode in ("fine", "coarse"):

            df_dict = evaluate(prediction_path,
                               args.annotation_path,
                               args.yaml_path,
                               mode,
                               valid_sensor_id=valid_sensor_id)

            micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
            macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)

            # Get index of first threshold that is at least 0.5
            thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).to_numpy().nonzero()[0][0]

            metrics[mode]['micro_auprc'].append(float(micro_auprc))
            metrics[mode]['micro_f1'].append(float(eval_df["F"][thresh_0pt5_idx]))
            metrics[mode]['macro_auprc'].append(float(macro_auprc))
            for coarse_id, auprc in class_auprc.items():
                if coarse_id not in metrics[mode]['class_auprc']:
                    metrics[mode]['class_auprc'][coarse_id] = []
                metrics[mode]['class_auprc'][coarse_id].append(float(auprc))

    average_metrics = {}
    for mode, mode_metrics in metrics.items():
        micro_auprc = mode_metrics['micro_auprc']
        micro_f1 = mode_metrics['micro_f1']
        macro_auprc = mode_metrics['macro_auprc']

        average_metrics[mode] = {
            'micro_auprc': {
                'mean': float(np.mean(micro_auprc)),
                'stdev': float(np.std(micro_auprc)) },
            'micro_f1': {
                'mean': float(np.mean(micro_f1)),
                'stdev': float(np.std(micro_f1)) },
            'macro_auprc': {
                'mean': float(np.mean(macro_auprc)),
                'stdev': float(np.std(macro_auprc)) },
            'class_auprc': {}
        }

        print("{} level evaluation:".format(mode.capitalize()))
        print("======================")
        print(" * Micro AUPRC:           {} (stdev {})".format(average_metrics[mode]['micro_auprc']['mean'],
                                                               average_metrics[mode]['micro_auprc']['stdev']))
        print(" * Micro F1-score (@0.5): {} (stdev {})".format(average_metrics[mode]['micro_f1']['mean'],
                                                               average_metrics[mode]['micro_f1']['stdev']))
        print(" * Macro AUPRC:           {} (stdev {})".format(average_metrics[mode]['macro_auprc']['mean'],
                                                               average_metrics[mode]['macro_auprc']['stdev']))
        print(" * Coarse Tag AUPRC:")

        for coarse_id, auprc_list in mode_metrics['class_auprc'].items():
            coarse_name = taxonomy['coarse'][int(coarse_id)]
            average_metrics[mode]['class_auprc'][coarse_name] = {
                'mean': float(np.mean(auprc_list)),
                'stdev': float(np.std(auprc_list))
            }
            print("      - {}: {} (stdev {})".format(coarse_name, average_metrics[mode]['class_auprc'][coarse_name]['mean'],
                                                                average_metrics[mode]['class_auprc'][coarse_name]['stdev']))

        total_metrics = {
            'folds': metrics,
            'average': average_metrics
        }

        eval_path = os.path.join(args.output_dir, 'evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(total_metrics, f)
