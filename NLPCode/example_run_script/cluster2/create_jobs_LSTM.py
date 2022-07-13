import sys
import os
import stat

"""
The following are the used argument names, the argument value must be directly behind the name and contain no spaces (also for directories):
--seed   : for the seeds (seperated by;)
--lr     : the learning rate
--dp     : 1 for differential privacy 0 for none dp
--out    : directory for the output of the program
--outM   : directory for the output of the model
--dsDir  : directory for the dataset
--BS     : the batchsize
--eps    : setting the epsilon for DP; "NONE" for NONE
--del    : setting the delta for DP
--noiseM : setting the noise multiplier for DP; "NONE" for NONE
--ES     : 1 for Early stopping 0 for no early stopping
"""

path_to_main = "<pathToRepo>/dp-across-nlp-tasks/NLPCode/natural_language_inference/LSTM/experiment_parameters.py"
i = 0

seed = [1234,6546,78126,51778,8672]
learning = [0.001]
dp = [1]
BS = [32]
epsilons = ["NONE"]
output = [f"<OUTPUT DIR>"]
dataset_dir = ["<path to dataset>"] 
glove_dir = ["<path to glove directory>"]
deltas = [1e-5]
noise_mul = [0.45]
alpha = [3.5]
early_stop = [0]

j_name = [f"<nameOfJob_{s}" for s in seed]

# enter the directory where all batch jobs should be stored
dirname = f'<path to store slurm script>'
if not os.path.exists(dirname):
    os.makedirs(dirname)

# sanity check if all arrays have the same length
assert(len(learning) == len(dp) == len(output) == len(dataset_dir) == len(BS) == len(epsilons) == len(deltas) == len(noise_mul) == len(early_stop))
assert(len(j_name) == len(seed))

for i_seed in range(len(seed)):
    filename = f"run_dp{dp[i]}_lr{str(learning[i]).replace('.', '_')}_e5_{seed[i_seed]}.sh"
    fullpath = os.path.join(dirname, filename)

    with open(fullpath, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#\n')
        f.write(f'#SBATCH -J {j_name[i_seed]}\n')
        f.write(f'#SBATCH -o {output[i]}/out.txt\n')
        f.write(f'#SBATCH -e {output[i]}/error.txt\n')
        #f.write('#SBATCH --mail-user=<e-mail>\n')
        #f.write('#SBATCH --mail-type=ALL\n')
        f.write('#SBATCH -n 1\n')
        f.write('#SBATCH --mem-per-cpu=32000\n')
        f.write('#SBATCH -c 2\n')
        f.write('#SBATCH -A project01460\n')
        f.write('#SBATCH -t 01-00\n')
        f.write('#SBATCH --gres=gpu:a100:1\n')
        f.write('#SBATCH -p testgpu24\n')
        f.write('ml cuda\n')
        f.write('source <path to conda env>/miniconda3/bin/activate pytorch\n')
        f.write(f'python {path_to_main} --seed {seed[i_seed]} --lr {learning[i]} --dp {dp[i]}  --out {output[i]} --dsDir {dataset_dir[i]} --glove {glove_dir[i]} --BS {BS[i]} --eps {epsilons[i]}  --del {deltas[i]} --alpha {alpha[i]} --noiseM {noise_mul[i]} --ES {early_stop[i]}\n')
        f.write('sleep 1\n')
    st = os.stat(fullpath)
    os.chmod(fullpath, st.st_mode | stat.S_IEXEC)

