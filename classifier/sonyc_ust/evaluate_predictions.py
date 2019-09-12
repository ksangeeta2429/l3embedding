import argparse
import json
import os
import oyaml as yaml
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Evaluation script for Urban Sound Tagging task for the DCASE 2019 Challenge.

        See `metrics.py` for more information about the metrics.
        """)

    parser.add_argument('prediction_path', type=str,
                        help='Path to prediction CSV file.')
    parser.add_argument('annotation_path', type=str,
                        help='Path to dataset annotation CSV file.')
    parser.add_argument('yaml_path', type=str,
                        help='Path to dataset taxonomy YAML file.')
    parser.add_argument('output_dir', type=str,
                        help='Output directory.')
    parser.add_argument('--split_path', type=str,
                        help='Optional path to split CSV file.')

    args = parser.parse_args()

    with open(args.yaml_path) as f:
        taxonomy = yaml.load(f)

    metrics = {
        'fine': {},
        'coarse': {}
    }

    for mode in ("fine", "coarse"):

        df_dict = evaluate(args.prediction_path,
                           args.annotation_path,
                           args.yaml_path,
                           mode,
                           split_path=args.split_path)

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


    eval_path = os.path.join(args.output_dir, 'evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(metrics, f)


