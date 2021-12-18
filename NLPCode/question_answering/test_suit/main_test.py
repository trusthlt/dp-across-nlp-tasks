import os
import torch
import pandas as pd
from datasets import load_dataset
import numpy as np

from pytorch_transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import BertForQuestionAnswering
import torch.optim as optim
import json
from tqdm import trange

import sys
import os
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import EarlyStopping, epoch_time
from opacus import PrivacyEngine
from Transformer.prep_data import load_test_set
from Transformer.train_eval import evaluate
from Transformer.dataset.utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)
from Transformer.model import BERTQAModel
from Transformer.dataset.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad, plot_pr_curve

def QA(EP, input_file, output_file):
    print("in QA")
    np.random.seed(EP.seed)
    torch.manual_seed(EP.seed)
    torch.cuda.manual_seed_all(EP.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(EP.bert_model_type)
    print("laoding dataset")
    test_set, test_examples, test_features = load_test_set(tokenizer, EP, input_file)
    test_sampler = SequentialSampler(test_set)
    test_dataloader = DataLoader(test_set, sampler=test_sampler, batch_size=EP.BATCH_SIZE, drop_last=False)

    print("creating model")
    model = BERTQAModel(EP)
    model.to(device)

    # test the best model
    model.load_state_dict(torch.load(EP.model_path))
    test_loss = "NOT IMPLEMENTED"
    print("evaluating")
    evaluate(model, tokenizer, test_examples, device, test_dataloader, test_features, True, EP, output_test=output_file)
    print("done")

def main(EP, input_file, output_file):
    QA(EP, input_file, output_file)


if __name__ == "__main__":
    pass