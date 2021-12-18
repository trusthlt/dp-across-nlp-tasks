import os
import torch
import torch.nn as nn
import pandas as pd
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
    
    train_set, train_examples, train_features, val_set, val_examples, val_features, sample_rate = load_train_set(tokenizer, EP)
    test_set, test_examples, test_features = load_test_set(tokenizer, EP)
    
    train_sampler = RandomSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=EP.BATCH_SIZE, drop_last=False)

    val_sampler = SequentialSampler(val_set)
    val_dataloader = DataLoader(val_set, sampler=val_sampler, batch_size=EP.BATCH_SIZE, drop_last=False)

    test_sampler = SequentialSampler(test_set)
    test_dataloader = DataLoader(test_set, sampler=test_sampler, batch_size=EP.BATCH_SIZE, drop_last=False)

    
    model = BERTQAModel(EP)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=EP.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # load checkpoint if necessary
    checkpoint_path = f"{EP.output_dir}model_dict{EP.seed}.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_f1 = checkpoint['best_f1']
    else:
        start_epoch = 0
        best_f1 = -100

    # do the differential privacy stuff
    if EP.privacy:
        model.train()
        if EP.alpha == None:
            privacy_engine = PrivacyEngine(model.trainable_layers,
                                        sample_rate=sample_rate,
                                        target_delta= EP.delta,
                                        target_epsilon=EP.epsilon,
                                        noise_multiplier=EP.noise_multiplier,
                                        epochs =EP.N_EPOCHS,
                                        max_grad_norm=EP.max_grad_norm)
        else:
            privacy_engine = PrivacyEngine(model.trainable_layers,
                                        sample_rate=sample_rate,
                                        target_delta= EP.delta,
                                        target_epsilon=EP.epsilon,
                                        noise_multiplier=EP.noise_multiplier,
                                        alphas=[EP.alpha],
                                        epochs =EP.N_EPOCHS,
                                        max_grad_norm=EP.max_grad_norm)
        privacy_engine = privacy_engine.to(device)
        privacy_engine.attach(optimizer)


    # Train and eval
    gold_data_dir_train = f'{EP.dataset_dir}my_train.json'
    gold_data_dir_val = f'{EP.dataset_dir}my_val.json'
    gold_data_dir_test = f'{EP.dataset_dir}test-v2.0.json'

    
    rest_epochs = EP.N_EPOCHS - start_epoch
    train_iterator = trange(rest_epochs, desc="Epoch")
    for epoch, _ in enumerate(train_iterator):
        start_time = time.time()
        train_exact, train_f1, train_loss = train(model, train_dataloader, optimizer, device, train_examples, train_features, gold_data_dir_train, criterion, EP)
        val_exact, val_f1, val_loss = evaluate(model, tokenizer, val_examples, device, val_dataloader, val_features, False, criterion, EP, gold_data_dir=gold_data_dir_val)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # print restuls
        f = open(f'{EP.output_dir}{EP.output_file}{EP.seed}.txt', 'a')
        f.write(f'Epoch: {start_epoch+epoch+1} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\t Train Loss: {train_loss:.3f} Train Exact: {train_exact:.3f} Train F1: {train_f1:.3f}\n')
        f.write(f'\t Val Loss: {val_loss:.3f} Val Exact: {val_exact:.3f} Val F1: {val_f1:.3f}\n')
        f.close()

        # save model and optimizer
        torch.save({
                    'epoch': start_epoch+epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'best_f1':best_f1,
                    }, checkpoint_path)

        # log privacy budget
        if EP.privacy:
            f = open(f'{EP.output_dir}{EP.priv_output}{EP.seed}.txt', 'a')
            f.write(f"Epoch: {epoch}\n")
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(EP.delta)
            nm = optimizer.privacy_engine.noise_multiplier
            print(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}")
            f.write(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}\n")
            f.close()

        if val_f1 > best_f1:
            # save best model
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{EP.output_dir}model{EP.seed}.pt")

    # test the best model
    model.load_state_dict(torch.load(f'{EP.output_dir}model{EP.seed}.pt'))
    test_loss = "NOT IMPLEMENTED"
    test_exact, test_f1 = evaluate(model, tokenizer, test_examples, device, test_dataloader, test_features, True, criterion, EP, gold_data_dir=gold_data_dir_test)
    f = open(f'{EP.output_dir}{EP.output_file}{EP.seed}.txt', 'a')
    f.write(f'Test Loss: {test_loss} Test Exact: {test_exact:.3f} Test F1: {test_f1:.3f}\n')
    f.close()

def main(EP):
    QA(EP)


if __name__ == "__main__":
    pass