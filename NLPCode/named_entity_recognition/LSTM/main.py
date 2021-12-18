import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from dataset import NerDataset, TAGS, pad
from dataset_wikiann import NerDatasetWikiann, TAGS_wikiann, pad_Wikiann
import os
import numpy as np
import argparse
from train_eval import train, eval
import time
import sys
import os
from opacus import PrivacyEngine
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import epoch_time

def NER(EP, SEED):
    torch.manual_seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'the device is:{device}')

    if EP.use_conll:
        print("using conll")
        train_dataset = NerDataset(f"{EP.dataset_dir}/train_set.txt", EP)
        val_dataset = NerDataset(f"{EP.dataset_dir}/valid_set.txt", EP, word2idx=train_dataset.word2idx)
        test_dataset = NerDataset(f"{EP.dataset_dir}/test_set.txt", EP, word2idx=train_dataset.word2idx)
    
        train_iter = data.DataLoader(dataset=train_dataset,
                                batch_size=EP.BATCH_SIZE,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=pad)
    
        eval_iter = data.DataLoader(dataset=val_dataset,
                                    batch_size=EP.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad)
        test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=EP.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad)

        model = Net(len(TAGS), train_dataset.weights_matrix, device, EP).to(device)
    else:
        print("using wikiann")
        train_dataset = NerDatasetWikiann(f"{EP.dataset_dir}/train_set.txt", EP)
        val_dataset = NerDatasetWikiann(f"{EP.dataset_dir}/valid_set.txt", EP, word2idx=train_dataset.word2idx)
        test_dataset = NerDatasetWikiann(f"{EP.dataset_dir}/test_set.txt", EP, word2idx=train_dataset.word2idx)

        train_iter = data.DataLoader(dataset=train_dataset,
                                batch_size=EP.BATCH_SIZE,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=pad_Wikiann)
    
        eval_iter = data.DataLoader(dataset=val_dataset,
                                    batch_size=EP.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad_Wikiann)

        test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=EP.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad_Wikiann)
        
        model = Net(len(TAGS_wikiann), train_dataset.weights_matrix, device, EP).to(device)

    


    

    if not EP.privacy:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=EP.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # setup privacy
    if EP.privacy:
        sample_rate = EP.BATCH_SIZE / train_dataset.__len__()
        if EP.alpha == None:
            privacy_engine = PrivacyEngine(model,
                                        sample_rate=sample_rate,
                                        target_delta= EP.delta,
                                        target_epsilon=EP.epsilon,
                                        noise_multiplier=EP.noise_multiplier,
                                        epochs =EP.N_EPOCHS,
                                        max_grad_norm=EP.max_grad_norm)
        else:
            privacy_engine = PrivacyEngine(model,
                                        sample_rate=sample_rate,
                                        target_delta= EP.delta,
                                        target_epsilon=EP.epsilon,
                                        noise_multiplier=EP.noise_multiplier,
                                        alphas=[EP.alpha],
                                        epochs =EP.N_EPOCHS,
                                        max_grad_norm=EP.max_grad_norm)
        privacy_engine = privacy_engine.to(device)
        privacy_engine.attach(optimizer)

    # start the training
    best_valid_acc = -1
    print("start training")
    for epoch in range(1, EP.N_EPOCHS + 1):
        start_time = time.time()
        print(f'currently in epoch{epoch}')
        train_loss, train_acc, train_precision, train_recall, train_f1 = train(model, train_iter, optimizer, criterion, EP)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = eval(model, eval_iter, criterion, optimizer, EP)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # save model if it is the best
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if not os.path.exists(EP.output_dir): os.makedirs(EP.output_dir)
            torch.save(model.state_dict(), f'{EP.output_dir}/{EP.output_file}{SEED}.pt')

        # log measurments
        # if folder is not created create one
        if not os.path.exists(EP.output_dir): os.makedirs(EP.output_dir)
        print(f'{EP.output_dir}/{EP.output_file}{SEED}.txt')
        f = open(f'{EP.output_dir}/{EP.output_file}{SEED}.txt', 'a')
        f.write(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Prec.: {train_precision:.3f} | Train Rec.: {train_recall:.3f} | Train F1: {train_f1:.3f} \n')
        f.write(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}% | Valid Prec.: {train_precision:.3f} | Valid Rec.: {valid_recall:.3f} | Valid F1: {valid_f1:.3f} \n')
        f.close()

        # log privacy budget
        if EP.privacy:
            f = open(f'{EP.output_dir}/{EP.priv_output}{SEED}.txt', 'a')
            f.write(f"Epoch: {epoch}\n")
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(EP.delta)
            nm = optimizer.privacy_engine.noise_multiplier
            print(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}")
            f.write(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}\n")
            f.close()

    # test best model on test set
    model.load_state_dict(torch.load(f'{EP.output_dir}/{EP.output_file}{SEED}.pt'))
    test_loss, test_acc, test_precision, test_recall, test_f1 = eval(model, test_iter, criterion, optimizer, EP)

    # log
    f = open(f'{EP.output_dir}/{EP.output_file}{SEED}.txt', 'a')
    f.write(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test Prec.: {test_precision:.3f} | Test Rec.: {test_recall:.3f} | Test F1: {test_f1:.3f} \n')
    f.close()


def main(EP):
    for SEED in EP.seeds:
        NER(EP, SEED)


if __name__ == "__main__":
    # main()
    pass
