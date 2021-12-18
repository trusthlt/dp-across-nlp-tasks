import os
import torch
import pandas as pd
from datasets import load_dataset
import numpy as np
from dataset.utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

from dataset.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad, plot_pr_curve
from pytorch_transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import BertForQuestionAnswering
import torch.optim as optim
from prep_data import load_test_set, load_train_set
from train_eval import train, evaluate
import json
from model import BERTQAModel
from tqdm import trange
import sys
import os
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import EarlyStopping, epoch_time
from opacus import PrivacyEngine

def QA(EP):
    np.random.seed(EP.seed)
    torch.manual_seed(EP.seed)
    torch.cuda.manual_seed_all(EP.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(EP.bert_model_type)

    test_set, test_examples, test_features = load_test_set(tokenizer, EP)

    test_sampler = SequentialSampler(test_set)
    test_dataloader = DataLoader(test_set, sampler=test_sampler, batch_size=EP.BATCH_SIZE, drop_last=False)

    
    model = BERTQAModel(EP)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=EP.LEARNING_RATE)


    # Train and eval
    gold_data_dir_test = f'{EP.dataset_dir}test-v2.0.json'


    # test the best model
    model.load_state_dict(torch.load(f'{EP.output_dir}model{EP.seed}.pt'))
    test_loss = "NOT IMPLEMENTED"
    test_exact, test_f1 = evaluate(model, tokenizer, test_examples, device, test_dataloader, test_features, gold_data_dir_test, True, EP)
    f = open(f'{EP.output_dir}{EP.output_file}{EP.seed}.txt', 'a')
    f.write(f'Test Loss: {test_loss} Test Exact: {test_exact:.3f} Test F1: {test_f1:.3f}\n')
    f.close()

def main(EP):
    QA(EP)


if __name__ == "__main__":
    pass