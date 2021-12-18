import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tuning_structs import Privacy, TrainingBERT, Tuning
from main import main

"""
The following are the used argument names for executing the model.
The argument value must be directly behind the name and contain no spaces (also for directories):
--seed   : for the seeds (seperated by;)
--lr     : the learning rate
--dp     : 1 for differential privacy 0 for none dp
--BLay   : the status on how many bert layers are frozen (0 -> freez all, 2 -> freez none, -1 - -11 freez last 1 - 11)
--rnn    : 0 for no lstm 1 for lstm
--out    : directory for the output of the program
--outM   : directory for the output of the model
--dsDir  : directory for the dataset
--BS     : the batchsize
--eps    : setting the epsilon for DP; "NONE" for NONE
--del    : setting the delta for DP
--noiseM : setting the noise multiplier for DP; "NONE" for NONE
--ES     : 1 for Early stopping 0 for no early stopping
"""


class ExperimentalParametersTransformers:
    def __init__(self):
         # bert model
        self.bert_model_type = 'bert-base-uncased'
        # get param list from function call
        args = sys.argv[1:]
        # seeds
        indx = args.index("--seed")
        self.seeds = args[indx + 1].split(";")
        self.seeds = [int(s) for s in self.seeds]
        # learning rate
        indx = args.index("--lr")
        self.LEARNING_RATE = float(args[indx + 1])
        # differential privacy
        indx = args.index("--dp")
        self.privacy = Privacy(bool(int(args[indx + 1])))
        # layers
        indx = args.index("--BLay")
        lay = TrainingBERT(int(args[indx + 1]))
        self.tuning = Tuning(self.privacy, lay, self.bert_model_type)
        # rnn
        indx = args.index("--rnn")
        self.use_rnn = bool(int(args[indx + 1]))
        # output dir
        indx = args.index("--out")
        self.output_dir_txt = args[indx + 1]
        # output model
        indx = args.index("--outM")
        self.output_dir_model = args[indx + 1]
        # dataset directory
        indx = args.index("--dsDir")
        self.dataset_dir = args[indx + 1]
        # batchsize
        indx = args.index("--BS")
        self.BATCH_SIZE = int(args[indx + 1])
        # epsilon
        indx = args.index("--eps")
        if "NONE" == args[indx + 1]:
            self.epsilon = None
        else:
            self.epsilon = float(args[indx + 1]) # 10;5;2;1
        # delta
        indx = args.index("--del")
        self.delta = float(args[indx + 1])
        # alpha
        indx = args.index("--alpha")
        if "NONE" == args[indx + 1]:
            self.alpha = None
        else:
            self.alpha = float(args[indx + 1]) 
        # noise multiplier
        indx = args.index("--noiseM")
        if "NONE" == args[indx + 1]:
            self.noise_multiplier = None
        else:
            self.noise_multiplier = float(args[indx + 1])
        # early stopping
        indx = args.index("--ES")
        self.early_stopping_active = bool(int(args[indx + 1]))
       

        # Constant paramters     

        # Model hyperparameters
        self.HIDDEN_DIM = 256
        self.OUTPUT_DIM = 1
        self.N_LAYERS = 2
        self.BIDIRECTIONAL = True
        self.DROPOUT = 0.25       

        # DP
        self.max_grad_norm = 1.0

        # for Training
        self.N_EPOCHS = 30
        self.PATIENCE = 5

        self.output_file = 'out_'
        self.priv_output = 'priv_out_'

        # call function for sentiment anal
        main(self)

if __name__ == "__main__":
    ExperimentalParametersTransformers()