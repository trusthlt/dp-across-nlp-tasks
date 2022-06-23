import torch
import torch.nn as nn
from torchtext.legacy import data
import torch.optim as optim
import math
from prep_data import split_and_cut, tokenizer, convert_to_int
from model import BERTNLIModel
import time
from train_eval import train, evaluate
from utils import EarlyStopping, epoch_time
from opacus import PrivacyEngine
import numpy as np
import os

def preprocessing(EP, device):
    cls_token_idx = tokenizer.cls_token_id
    sep_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    
    #For sequence
    TEXT = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = split_and_cut,
                    preprocessing = tokenizer.convert_tokens_to_ids,
                    pad_token = pad_token_idx,
                    unk_token = unk_token_idx)
    #For label
    LABEL = data.LabelField()
    #For Attention mask
    ATTENTION = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = split_and_cut,
                    preprocessing = convert_to_int,
                    pad_token = pad_token_idx)
    #For token type ids
    TTYPE = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = split_and_cut,
                    preprocessing = convert_to_int,
                    pad_token = 1)

    # Fields will help to map the column with the torchtext Field.
    fields = [('label', LABEL), ('sequence', TEXT), ('attention_mask', ATTENTION), ('token_type', TTYPE)]

    if EP.use_BERT:
        post_fix=""
    else:
        post_fix = "_distil"
        
    train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = f'{EP.dataset_dir}snli_1.0',
                                        train = f'{EP.dataset_dir}snli_1.0_train{post_fix}.csv',
                                        validation = f'{EP.dataset_dir}snli_1.0_dev{post_fix}.csv',
                                        test = f'{EP.dataset_dir}snli_1.0_test{post_fix}.csv',
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True)
    
    LABEL.build_vocab(train_data)

    #Create iterator
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = EP.BATCH_SIZE,
        sort_key = lambda x: len(x.sequence),
        sort_within_batch = False, 
        device = device)
    print(f'The train size is {len(train_data)}')
    return LABEL, train_iterator, valid_iterator, test_iterator, EP.BATCH_SIZE / len(train_data)

def natural_language_inference(EP):
    torch.manual_seed(EP.SEED)
    np.random.seed(EP.SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('start preprocessing')
    LABEL, train_iterator, valid_iterator, test_iterator, sample_rate = preprocessing(EP, device)
    print('end preprocessing')

    #defining model
    OUTPUT_DIM = len(LABEL.vocab)
    model = BERTNLIModel(EP, OUTPUT_DIM).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=EP.LEARNING_RATE,eps=1e-6)
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
            print(f'Setting the alpha to {EP.alpha}')
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
    for epoch in range(start_epoch, EP.N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc, train_precision, train_recall, train_f1 = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_iterator, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f"{EP.output_dir}/{EP.output_file}{EP.SEED}.pt")

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train Prec.: {train_precision:.3f} | Train Rec.: {train_recall:.3f} | Train F1: {train_f1:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Valid Prec.: {valid_precision:.3f} | Valid Rec.: {valid_recall:.3f} | Valid F1: {valid_f1:.3f}')

        f = open(f"{EP.output_dir}/{EP.output_file}{EP.SEED}.txt", "a")
        f.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train Prec.: {train_precision:.3f} | Train Rec.: {train_recall:.3f} | Train F1: {train_f1:.3f} \n')
        f.write(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Valid Prec.: {valid_precision:.3f} | Valid Rec.: {valid_recall:.3f} | Valid F1: {valid_f1:.3f} \n')
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
            f = open(f"{EP.output_dir}/{EP.output_file}{EP.SEED}.txt", "a")
            f.write(f'Did early stoppin in epoch {epoch}\n')
            f.close()
            break
    
    model.load_state_dict(torch.load(f"{EP.output_dir}/{EP.output_file}{EP.SEED}.pt"))
    test_loss, test_acc, test_precision, test_recall, test_f1  = evaluate(model, test_iterator, criterion, device)

    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}% | Test Prec.: {test_precision:.3f} | Test Rec.: {test_recall:.3f} | Test F1: {test_f1:.3f}')
    f = open(f"{EP.output_dir}/{EP.output_file}{EP.SEED}.txt", "a")
    f.write(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}% | Test Prec.: {test_precision:.3f} | Test Rec.: {test_recall:.3f} | Test F1: {test_f1:.3f}\n')
    f.close()

def main(EP):
    natural_language_inference(EP)    