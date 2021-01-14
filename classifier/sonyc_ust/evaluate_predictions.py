import argparse
import json
import os
print('Cur. path:', os.getcwd())
import oyaml as yaml
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc
import git
import copy
import getpass
from googleapiclient import discovery
from gsheets import get_credentials, append_row

class GSheetLogger():
    """
    Update Google Sheets Spreadsheet
    """

    def __init__(self, google_dev_app_name, spreadsheet_id, param_dict):
        self.google_dev_app_name = google_dev_app_name
        self.spreadsheet_id = spreadsheet_id
        self.credentials = get_credentials(google_dev_app_name)
        self.service = discovery.build('sheets', 'v4', credentials=self.credentials)
        self.param_dict = copy.deepcopy(param_dict)

        append_row(self.service, self.spreadsheet_id, self.param_dict, 'sonyc_ust')


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
    parser.add_argument('gsheet_id', type=str,
                        help='Google Spreadsheet ID for centralized logging of experiments')
    parser.add_argument('google_dev_app_name', type=str,
                        help='Google Developer Application Name for using API')
    parser.add_argument('--valid_sensor_id', type=int,
                        help='Optional validate sensor id to be evaluated.')
    parser.add_argument('--split_path', type=str,
                        help='Optional path to split CSV file.')

    args = parser.parse_args()

    with open(args.yaml_path) as f:
        taxonomy = yaml.load(f, Loader=yaml.FullLoader)

    metrics = {
        'fine': {},
        'coarse': {}
    }

    for mode in ("fine", "coarse"):

        df_dict = evaluate(args.prediction_path,
                           args.annotation_path,
                           args.yaml_path,
                           mode,
                           valid_sensor_ids=args.valid_sensor_id,
                           split_path=args.split_path)

        micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
        macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)

        # Get index of first threshold that is at least 0.5
        thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).to_numpy().nonzero()[0][0]

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


    # Get hyperparameters from .json
    with open(os.path.join(args.output_dir, 'hyper_params.json')) as hyp:
        hyp = json.load(hyp)

    if 'mae' in hyp['emb_dir']:
        model_type = 'MAE'
    elif 'edgel3' in hyp['emb_dir']:
        model_type = 'EDGEL3'
    elif 'umap' in hyp['emb_dir']:
        model_type = 'UMAP'
    else:
        model_type = 'BASELINE'

    if '/music/' in hyp['emb_dir'] or model_type in ['BASELINE', 'EDGEL3']:
        upstream_data = 'music'
    elif '/sonyc/' in hyp['emb_dir']:
        upstream_data = 'sonyc'
    else:
        upstream_data = 'sonyc_ust'

    # Update Google spreadsheet
    param_dict = {
        'username': getpass.getuser(),
        'model_type': model_type,
        'uptsream_data': upstream_data,
        'git_commit': git.Repo(os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True).head.object.hexsha,
        'annotation_path': hyp['annotation_path'],
        'taxonomy_path': hyp['taxonomy_path'],
        'emb_dir': hyp['emb_dir'],
        'output_dir': hyp['output_dir'],
        'exp_id': hyp['exp_id'],
        'hidden_layer_size': hyp['hidden_layer_size'],
        'num_hidden_layers': hyp['num_hidden_layers'],
        'learning_rate': hyp['learning_rate'],
        'l2_reg': hyp['l2_reg'],
        'batch_size': hyp['batch_size'],
        'num_epochs': hyp['num_epochs'],
        'patience': hyp['patience'],
        'sensor_factor': hyp['sensor_factor'],
        'proximity_factor': hyp['proximity_factor'],
        'no_standardize': hyp['no_standardize'],
        'cooccurrence_loss': hyp['cooccurrence_loss'],
        'cooccurrence_loss_factor': hyp['cooccurrence_loss_factor'],
        'pca': hyp['pca'],
        'pca_components': hyp['pca_components'],
        'label_mode': hyp['label_mode'],
        'oversample': hyp['oversample'],
        'oversample_iters': hyp['oversample_iters'],
        'thresh_type': hyp['thresh_type'],
        'target_mode': hyp['target_mode'],
        'no_timestamp': hyp['no_timestamp'],
        'split_path': hyp['split_path'],
        'optimizer': hyp['optimizer'],
        'micro_auprc': metrics[hyp['label_mode']]["micro_auprc"],
        'micro_f1': metrics[hyp['label_mode']]["micro_f1"],
        'macro_auprc': metrics[hyp['label_mode']]["macro_auprc"],
        'class_auprc': metrics[hyp['label_mode']]["class_auprc"]
    }

    GSheetLogger(args.google_dev_app_name, args.gsheet_id, param_dict)

