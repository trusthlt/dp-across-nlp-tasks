import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
from prep_data import prep_data
from MyDataset import tag2idx, PAD_TAG
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import epoch_time
from train_eval import train, evaluate
from model import BiLSTMPOSTagger
from opacus import PrivacyEngine

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        

def POS(EP):
    print("start")
    random.seed(EP.SEED)
    np.random.seed(EP.SEED)
    torch.manual_seed(EP.SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab, weight_matrix, UD_TAGS, train_iterator, valid_iterator, test_iterator, sample_rate = prep_data(EP, device)

    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(tag2idx)
    PAD_IDX = tag2idx[PAD_TAG]

    model = BiLSTMPOSTagger(INPUT_DIM, OUTPUT_DIM, PAD_IDX, weight_matrix, EP)
    
    model.apply(init_weights)

    # get optimizer and critirion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

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

        privacy_engine.attach(optimizer)
        privacy_engine.to(device)

    # start training
    best_valid_acc = -100

    for epoch in range(EP.N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc, train_precision, train_recall, train_f1 = train(model, train_iterator, optimizer, criterion, PAD_IDX, device)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_iterator, criterion, PAD_IDX, device)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f'{EP.output_dir}/model{EP.SEED}.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.2f} | Train Prec: {train_precision:.2f} | Train Rec: {train_recall:.2f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Val. F1: {valid_f1:.2f} | Val. Prec: {valid_precision:.2f} | Val. Rec: {valid_recall:.2f}')
        f = open(f'{EP.output_dir}/{EP.output_file}{EP.SEED}.txt', "a")
        f.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.2f} | Train Prec: {train_precision:.2f} | Train Rec: {train_recall:.2f}\n')
        f.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Val. F1: {valid_f1:.2f} | Val. Prec: {valid_precision:.2f} | Val. Rec: {valid_recall:.2f}\n')    
        f.close

        # log privacy budget
        if EP.privacy:
            f = open(f'{EP.output_dir}/{EP.priv_output}{EP.SEED}.txt', 'a')
            f.write(f"Epoch: {epoch}\n")
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(EP.delta)
            nm = optimizer.privacy_engine.noise_multiplier
            print(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}")
            f.write(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}\n")
            f.close()

    # test best model
    model.load_state_dict(torch.load(f'{EP.output_dir}/model{EP.SEED}.pt'))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_iterator, criterion, PAD_IDX, device)
    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1:.2f} | Test Prec: {test_prec:.2f} | Test Rec: {test_rec:.2f}')
    f = open(f'{EP.output_dir}/{EP.output_file}{EP.SEED}.txt', "a")
    f.write(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1:.2f} | Test Prec: {test_prec:.2f} | Test Rec: {test_rec:.2f}\n')
    f.close()

def main(EP):
    POS(EP)