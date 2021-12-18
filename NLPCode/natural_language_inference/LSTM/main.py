import torch
import torch.nn as nn
import torch.optim as optim
from model import NLIBiLSTM
from torchtext.legacy import data, datasets
from train_eval import train, evaluate
import random
import numpy as np
import time
from opacus import PrivacyEngine
from tuning_structs import Privacy
from utils import epoch_time, EarlyStopping
import torchtext
import os

def data_preprocessing(EP, device):
    TEXT = data.Field(tokenize = 'spacy', lower = True)
    LABEL = data.LabelField()
    train_data, valid_data, test_data = datasets.SNLI.splits(TEXT, LABEL, root=EP.dataset_dir)

    MIN_FREQ = 2
    # load data path
    vec = torchtext.vocab.Vectors("glove.6B.300d.txt", cache=EP.glove_dir)
    TEXT.build_vocab(train_data, 
                    min_freq = MIN_FREQ,
                    vectors = vec,
                    unk_init = torch.Tensor.normal_)

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = EP.BATCH_SIZE,
        device = device)

    LABEL.build_vocab(train_data)
    
    return TEXT, LABEL, train_iterator, valid_iterator, test_iterator, EP.BATCH_SIZE / len(train_data)

def natural_language_inference(EP):
    random.seed(EP.SEED)
    np.random.seed(EP.SEED)
    torch.manual_seed(EP.SEED)
    torch.backends.cudnn.deterministic = True

    BATCH_SIZE = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEXT, LABEL, train_iterator, valid_iterator, test_iterator, sample_rate = data_preprocessing(EP, device)

    INPUT_DIM = len(TEXT.vocab)
    OUTPUT_DIM = len(LABEL.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = NLIBiLSTM(INPUT_DIM, EP, OUTPUT_DIM, PAD_IDX).to(device)

    # set the pretrained embeddings
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EP.EMBEDDING_DIM)
    model.embedding.weight.requires_grad = False

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)

    # load checkpoint if necessary
    checkpoint_path = f"{EP.output_dir}/model_dict{EP.SEED}.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_valid_acc = checkpoint['best_acc']
        set_patientce = checkpoint['ES']
    else:
        start_epoch = 0
        best_valid_acc = -10
        set_patientce = 0

    # do the differential privacy stuff
    if EP.privacy:
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

    early_stopping = EarlyStopping(EP.PATIENCE)
    early_stopping.patience = set_patientce
    print("training")
    for epoch in range(start_epoch, EP.N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc, train_precision, train_recall, train_f1 = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f'{EP.output_dir}/{EP.output_file}{EP.SEED}.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train Prec.: {train_precision:.3f} | Train Rec.: {train_recall:.3f} | Train F1: {train_f1:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Valid Prec.: {valid_precision:.3f} | Valid Rec.: {valid_recall:.3f} | Valid F1: {valid_f1:.3f}')

        f = open(f'{EP.output_dir}/{EP.output_file}{EP.SEED}.txt', 'a')
        f.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train Prec.: {train_precision:.3f} | Train Rec.: {train_recall:.3f} | Train F1: {train_f1:.3f}\n')
        f.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Valid Prec.: {valid_precision:.3f} | Valid Rec.: {valid_recall:.3f} | Valid F1: {valid_f1:.3f}\n')
        f.close()
    
        # log privacy budget
        if EP.privacy:
            f = open(f'{EP.output_dir}/{EP.priv_output}{EP.SEED}.txt', 'a')
            f.write(f"Epoch: {epoch}\n")
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(EP.delta)
            nm = optimizer.privacy_engine.noise_multiplier
            print(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}")
            f.write(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}\n")
            f.close()

        # save model and optimizer
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'best_acc':best_valid_acc,
                    'ES':early_stopping.patience,
                    }, checkpoint_path)
        
        # check for early stopping
        if EP.early_stopping_active and early_stopping.should_stop(valid_acc):
            print(f'Did early stoppin in epoch {epoch}')
            
            f = open(f'{EP.output_dir}/{EP.output_file}{EP.SEED}.txt', 'a')
            f.write(f'Did early stoppin in epoch {epoch}\n')
            f.close()
            break

    model.load_state_dict(torch.load(f'{EP.output_dir}/{EP.output_file}{EP.SEED}.pt'))
    test_loss, test_acc, test_precision, test_recall, test_f1  = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}% | Test Prec.: {test_precision:.3f} | Test Rec.: {test_recall:.3f} | Test F1: {test_f1:.3f}')
    f = open(f'{EP.output_dir}/{EP.output_file}{EP.SEED}.txt', 'a')
    f.write(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}% | Test Prec.: {test_precision:.3f} | Test Rec.: {test_recall:.3f} | Test F1: {test_f1:.3f}\n')
    f.close()

def main(EP):
    natural_language_inference(EP)

#if __name__ == "__main__":
    