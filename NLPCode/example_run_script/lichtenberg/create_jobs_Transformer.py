import os
import stat
"""
The following are the used argument names, the argument value must be directly behind the name and contain no spaces (also for directories):
--seed   : for the seeds (seperated by;)
--lr     : the learning rate
--BLay   : the status on how many bert layers are frozen 
           (0 -> freez all, 2 -> freez none, -1 - -11 freez last 1 - 11, 1 -> freez embeds)
--rnn    : 0 for no lstm 1 for lstm
--dp     : 1 for differential privacy 0 for none dp
--out    : directory for the output of the program
--outM   : directory for the output of the model
--dsDir  : directory that contrains the dataset
--BS     : the batchsize
--eps    : setting the epsilon for DP; "NONE" for NONE
--del    : setting the delta for DP
--noiseM : setting the noise multiplier for DP; "NONE" for NONE
--ES     : 1 for Early stopping 0 for no early stopping
"""

path_to_main = "<pathToRepo>/dp-across-nlp-tasks/NLPCode/natural_language_inference/Transformer/experiment_parameters.py"

# arrays for the experiments
i = 0

#[1234, 6546,78126,51778,8672]
seed = [1234,6546,78126,51778,8672]
learning = [0.000001]
dp = [1]
BLay = [0]
rnn = [1]
output = ["<output directory>"]
dataset_dir = ["<path to dataset>"]
BS = [32]
epsilons = ["NONE"]
deltas = [1e-5]
noise_mul = [0.7350]
alpha = [10]
early_stop = [0]
j_name = [f"{learning}_dp{dp[0]}_Tr_LSTM_{s}" for s in seed]

# enter the directory where all batch jobs should be stored
dirname = f'<path to store slurm script>'
if not os.path.exists(dirname):
    os.makedirs(dirname)

# sanity check if all arrays have the same length
assert(len(learning) == len(dp) == len(output)  == len(dataset_dir) == len(BS) == len(epsilons) == len(deltas) == len(noise_mul) == len(early_stop))
assert(len(j_name) == len(seed))

for i_seed in range(len(seed)):
    filename = f"run_{j_name[i_seed]}.sh"
    fullpath = os.path.join(dirname, filename)

    with open(fullpath, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#\n')
        f.write(f'#SBATCH -J {j_name[i]}\n')
        f.write(f'#SBATCH -o {output[i]}/out.txt\n')
        f.write(f'#SBATCH -e {output[i]}/error.txt\n')
        #f.write('#SBATCH --mail-user=manuel.senge@stud.tu-darmstadt.de\n')
        #f.write('#SBATCH --mail-type=ALL\n')
        f.write('#SBATCH -n 1\n')
        f.write('#SBATCH --mem-per-cpu=8000\n')
        f.write('#SBATCH -c 2\n')
        f.write('#SBATCH -A project01460\n')
        f.write('#SBATCH -t 01-00\n')
        f.write('#SBATCH --gres=gpu:1\n')
        f.write('#SBATCH -p testgpu24\n')
        f.write('ml cuda\n')
        f.write('source <path to conda env>/miniconda3/bin/activate pytorch\n')
        # pip install --user torch=..
        f.write(f'python {path_to_main} --seed {seed[i_seed]} --lr {learning[i]} --dp {dp[i]} --BLay {BLay[i]} --rnn {rnn[i]}  --out {output[i]} --dsDir {dataset_dir[i]} --BS {BS[i]} --eps {epsilons[i]}  --del {deltas[i]}  --noiseM {noise_mul[i]} --alpha {alpha[i]} --ES {early_stop[i]}\n')
    st = os.stat(fullpath)
    os.chmod(fullpath, st.st_mode | stat.S_IEXEC)
