from tqdm import tqdm, trange
import torch

import sys
import os
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Transformer.dataset.utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)
from Transformer.dataset.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad, plot_pr_curve
import numpy as np 

"""
use this file for the test bench of SQAD
"""
def official_eval_squad_test(examples, features, all_results, EP, output_dir):
    # Compute predictions
    lr = str(EP.LEARNING_RATE).replace(".", ",")
    output_prediction_file = output_dir
    output_nbest_file = f"{EP.preds_dir}nbest_predictions_dp{int(EP.privacy_num)}_lr{lr}_BL{EP.tuning.training_bert}_{EP.seed}.json"
    output_null_log_odds_file = f"{EP.preds_dir}null_odds_dp{int(EP.privacy_num)}_lr{lr}_BL{EP.tuning.training_bert}_{EP.seed}.json"

    write_predictions(examples, features, all_results, 10,
                    30, True, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file, False,
                    True, 0.0)