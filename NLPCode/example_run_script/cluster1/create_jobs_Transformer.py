import sys
import os
import stat

"""
The following are the used argument names, the argument value must be directly behind the name and contain no spaces (also for directories):
--seed   : for the seeds (seperated by;)
--lr     : the learning rate
--dp     : 1 for differential privacy 0 for none dp
--BLay   : the status on how many bert layers are frozen 
           (0 -> freez all, 2 -> freez none, -1 - -11 freez last 1 - 11), 1-> freez embeds
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

path_to_main = "<pathToRepo>/dp-across-nlp-tasks/NLPCode/named_entity_recognition/Transformer/experiment_parameters.py"

# arrays for the experiments
num_exp = 1
req = "a100:"

#[1234, 6546,78126,51778,8672]
seed = [[1234,6546,78126,51778,8672]]
learning = [0.001]
dp = [1]
BLay = [1]
rnn = [0]
BS = [8]
epsilons = ["NONE"]
output = [f"<output dir>"]
output_M = output
dataset_dir = ["<dataset dir>"]
deltas = [1e-5]
noise_mul = [0.5391]
alpha = [3.5]
early_stop = [0]
j_name = [f"<name of job>_{seed[0][0]}"]


# enter the directory where all batch jobs should be stored
dirname = '<path to store slurm job'
if not os.path.exists(dirname):
    os.makedirs(dirname)

# sanity check if all arrays have the same length
assert(len(j_name) == len(seed) == len(learning) == len(alpha) == len(dp) == len(BLay) == len(rnn) == len(output)  == len(output_M) == len(dataset_dir) == len(BS) == len(epsilons) == len(deltas) == len(noise_mul) == len(early_stop) == num_exp)

for i in range(num_exp):
    for i_seed in range(len(seed[i])):
        filename = f"run_Tr_ALL_lr{str(learning[i]).replace('.', '_')}_epsilon5_{seed[0][i_seed]}.sh"
        fullpath = os.path.join(dirname, filename)

        with open(fullpath, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('#\n')
            f.write(f'#SBATCH --job-name={j_name[i]}\n')
            f.write(f'#SBATCH --output={output[i]}/out.txt\n')
            f.write(f'#SBATCH --error={output[i]}/error.txt\n')
            #f.write('#SBATCH --mail-user=manuel.senge@stud.tu-darmstadt.de\n')
            #f.write('#SBATCH --mail-type=ALL\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --mem=32GB\n')
            f.write('#SBATCH --cpus-per-task=2\n')
            f.write('#SBATCH --partition=<partition>\n')
            f.write(f'#SBATCH --gres=gpu:{req}1\n')
            
            f.write('source <path to conda env>/<conda_env>/bin/activate\n')
            f.write(f'python {path_to_main} --seed {seed[i][i_seed]} --lr {learning[i]} --dp {dp[i]} --BLay {BLay[i]} --rnn {rnn[i]}  --out {output[i]}  --outM {output_M[i]} --dsDir {dataset_dir[i]} --BS {BS[i]} --eps {epsilons[i]}  --del {deltas[i]}  --alpha {alpha[i]} --noiseM {noise_mul[i]} --ES {early_stop[i]}\n')
            f.write('sleep 1\n')
        st = os.stat(fullpath)
        os.chmod(fullpath, st.st_mode | stat.S_IEXEC)

