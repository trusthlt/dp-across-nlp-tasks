from transformers import BertTokenizer
from zipfile import ZipFile
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
out_dir = '<output dir for logging>'
counter = 1

'''
Function only needed if zip needs to be extracted
'''
def download_data():
    file_name = "snli_1.0.zip"
    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()
        # extracting all the files
        zip.extractall()
'''
Function only needed for creation of csv file needs to be extracted
'''    
def create_csv_with_tokens():
    global counter
    f = open(f"{out_dir}output_data.txt", "a")
    f.write("in tokens\n")
    f.close()

    prefix = '<path to datset>/' # ukp
    
    #load dataset
    df_train = pd.read_csv(prefix + 'snli_1.0/snli_1.0_train.txt', sep='\t')
    df_dev = pd.read_csv(prefix + 'snli_1.0/snli_1.0_dev.txt', sep='\t')
    df_test = pd.read_csv(prefix + 'snli_1.0/snli_1.0_test.txt', sep='\t')

    f = open(f"{out_dir}output_data.txt", "a")
    f.write("dataset loaded\n")
    f.close()

    #Get neccesary columns
    df_train = df_train[['gold_label','sentence1','sentence2']]
    df_dev = df_dev[['gold_label','sentence1','sentence2']]
    df_test = df_test[['gold_label','sentence1','sentence2']]

    #Trim each sentence upto maximum length
    df_train['sentence1'] = df_train['sentence1'].apply(trim_sentence)
    df_train['sentence2'] = df_train['sentence2'].apply(trim_sentence)
    df_dev['sentence1'] = df_dev['sentence1'].apply(trim_sentence)
    df_dev['sentence2'] = df_dev['sentence2'].apply(trim_sentence)
    df_test['sentence1'] = df_test['sentence1'].apply(trim_sentence)
    df_test['sentence2'] = df_test['sentence2'].apply(trim_sentence)

    f = open(f"{out_dir}output_data.txt", "a")
    f.write("trimmed\n")
    f.close()

    #Add [CLS] and [SEP] tokens
    df_train['sent1'] = '[CLS] ' + df_train['sentence1'] + ' [SEP] '
    df_train['sent2'] = df_train['sentence2'] + ' [SEP]'
    df_dev['sent1'] = '[CLS] ' + df_dev['sentence1'] + ' [SEP] '
    df_dev['sent2'] = df_dev['sentence2'] + ' [SEP]'
    df_test['sent1'] = '[CLS] ' + df_test['sentence1'] + ' [SEP] '
    df_test['sent2'] = df_test['sentence2'] + ' [SEP]'

    f = open(f"{out_dir}output_data.txt", "a")
    f.write("added special tokens\n")
    f.close()

    #Apply Bert Tokenizer for tokeinizing
    counter = 1
    df_train['sent1_t'] = df_train['sent1'].apply(tokenize_bert)
    print('df_train[sent1]')
    counter = 1
    df_train['sent2_t'] = df_train['sent2'].apply(tokenize_bert)
    print('df_train[sent2]')
    counter = 1
    df_dev['sent1_t'] = df_dev['sent1'].apply(tokenize_bert)
    print('df_dev[sent1]')
    counter = 1
    df_dev['sent2_t'] = df_dev['sent2'].apply(tokenize_bert)
    print('df_dev[sent2]')
    counter = 1
    df_test['sent1_t'] = df_test['sent1'].apply(tokenize_bert)
    print('df_test[sent1]')
    counter = 1
    df_test['sent2_t'] = df_test['sent2'].apply(tokenize_bert)
    print('df_test[sent2]')

    f = open(f"{out_dir}output_data.txt", "a")
    f.write("applied bert\n")
    f.close()

    #Get Topen type ids for both sentence
    df_train['sent1_token_type'] = df_train['sent1_t'].apply(get_sent1_token_type)
    df_train['sent2_token_type'] = df_train['sent2_t'].apply(get_sent2_token_type)
    df_dev['sent1_token_type'] = df_dev['sent1_t'].apply(get_sent1_token_type)
    df_dev['sent2_token_type'] = df_dev['sent2_t'].apply(get_sent2_token_type)
    df_test['sent1_token_type'] = df_test['sent1_t'].apply(get_sent1_token_type)
    df_test['sent2_token_type'] = df_test['sent2_t'].apply(get_sent2_token_type)

    f = open(f"{out_dir}output_data.txt", "a")
    f.write("got type\n")
    f.close()

    #Combine both sequences
    df_train['sequence'] = df_train['sent1_t'] + df_train['sent2_t']
    df_dev['sequence'] = df_dev['sent1_t'] + df_dev['sent2_t']
    df_test['sequence'] = df_test['sent1_t'] + df_test['sent2_t']

    f = open(f"{out_dir}output_data.txt", "a")
    f.write("combined sentences\n")
    f.close()

    #Get attention mask
    df_train['attention_mask'] = df_train['sequence'].apply(get_sent2_token_type)
    df_dev['attention_mask'] = df_dev['sequence'].apply(get_sent2_token_type)
    df_test['attention_mask'] = df_test['sequence'].apply(get_sent2_token_type)

    f = open(f"{out_dir}output_data.txt", "a")
    f.write("attention mask\n")
    f.close()

    #Get combined token type ids for input
    df_train['token_type'] = df_train['sent1_token_type'] + df_train['sent2_token_type']
    df_dev['token_type'] = df_dev['sent1_token_type'] + df_dev['sent2_token_type']
    df_test['token_type'] = df_test['sent1_token_type'] + df_test['sent2_token_type']

    #Now make all these inputs as sequential data to be easily fed into torchtext Field.
    df_train['sequence'] = df_train['sequence'].apply(combine_seq)
    df_dev['sequence'] = df_dev['sequence'].apply(combine_seq)
    df_test['sequence'] = df_test['sequence'].apply(combine_seq)
    df_train['attention_mask'] = df_train['attention_mask'].apply(combine_mask)
    df_dev['attention_mask'] = df_dev['attention_mask'].apply(combine_mask)
    df_test['attention_mask'] = df_test['attention_mask'].apply(combine_mask)
    df_train['token_type'] = df_train['token_type'].apply(combine_mask)
    df_dev['token_type'] = df_dev['token_type'].apply(combine_mask)
    df_test['token_type'] = df_test['token_type'].apply(combine_mask)
    df_train = df_train[['gold_label', 'sequence', 'attention_mask', 'token_type']]
    df_dev = df_dev[['gold_label', 'sequence', 'attention_mask', 'token_type']]
    df_test = df_test[['gold_label', 'sequence', 'attention_mask', 'token_type']]
    df_train = df_train.loc[df_train['gold_label'].isin(['entailment','contradiction','neutral'])]
    df_dev = df_dev.loc[df_dev['gold_label'].isin(['entailment','contradiction','neutral'])]
    df_test = df_test.loc[df_test['gold_label'].isin(['entailment','contradiction','neutral'])]

    f = open(f"{out_dir}output_data.txt", "a")
    f.write(" before creation\n")
    f.close()


    #Save prepared data as csv file
    df_train.to_csv(prefix + 'snli_1.0/snli_1.0_train.csv', index=False)
    df_dev.to_csv(prefix + 'snli_1.0/snli_1.0_dev.csv', index=False)
    df_test.to_csv(prefix + 'snli_1.0/snli_1.0_test.csv', index=False)

    f = open(f"{out_dir}output_data.txt", "a")
    f.write("after creation\n")
    f.close()

# helper functions for dataframes
def tokenize_bert(sentence):
    global counter
    counter = counter + 1
    if not isinstance(sentence,str):
        print(counter)
        print(sentence)
    tokens = tokenizer.tokenize(sentence) 
    return tokens
def split_and_cut(sentence):
    max_input_length = 256
    tokens = sentence.strip().split(" ")
    tokens = tokens[:max_input_length]
    return tokens
def trim_sentence(sent):
    try:
        sent = sent.split()
        sent = sent[:128]
        return " ".join(sent)
    except:
        return sent
#Get list of 0s 
def get_sent1_token_type(sent):
    try:
        return [0]* len(sent)
    except:
        return []
#Get list of 1s
def get_sent2_token_type(sent):
    try:
        return [1]* len(sent)
    except:
        return []
#combine from lists
def combine_seq(seq):
    return " ".join(seq)
#combines from lists of int
def combine_mask(mask):
    mask = [str(m) for m in mask]
    return " ".join(mask)
# To convert back attention mask and token type ids to integer.
def convert_to_int(tok_ids):
    tok_ids = [int(x) for x in tok_ids]
    return tok_ids

def main():
    f = open(f"{out_dir}output_data.txt", "a")
    f.write("in main\n")
    f.close()
    create_csv_with_tokens()

if __name__ == "__main__":
    f = open(f"{out_dir}output_data.txt", "a")
    f.write("before main\n")
    f.close()
    create_csv_with_tokens()
