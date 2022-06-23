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

path_to_main = "<pathToRepo>/dp-across-nlp-tasks/NLPCode/named_entity_recognition/LSTM/experiment_parameters.py"

# arrays for the experiments
num_exp = 1
req = ""

seed = [[1234,6546,78126,51778,8672]]
learning = [0.001]
dp = [1]
BS = [32]
epsilons = ["NONE"]
output = ["<output dir>"]
dataset_dir = ["<dataset dir>"]
glove = ["<glove dir>"]
deltas = [1e-5]
alpha = [17]
noise_mul = [1.24]
early_stop = [0]


j_name = [f"<nameOfJob>_{s}" for s in seed[0]]

# enter the directory where all batch jobs should be stored
dirname = '<path to store slurm job>'
if not os.path.exists(dirname):
    os.makedirs(dirname)

# sanity check if all arrays have the same length
assert(len(seed) == len(learning) == len(dp) == len(output) == len(alpha) == len(dataset_dir) == len(BS) == len(epsilons) == len(deltas) == len(noise_mul) == len(early_stop) == num_exp)
assert (len(j_name) == len(seed[0]))
for i in range(num_exp):
    for i_seed in range(len(seed[i])):
        filename = f"run_dp{dp[i]}_lr{str(learning[i]).replace('.', '_')}_epsilon1_{seed[i][i_seed]}.sh"
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
            f.write('#SBATCH --mem=8GB\n')
            f.write('#SBATCH --cpus-per-task=2\n')
            f.write('#SBATCH --partition=<partition>\n')
            f.write(f'#SBATCH --gres=gpu:{req}1\n')
            
            f.write('source <path to conda env>/pytorch_senge/bin/activate\n')
            f.write(f'python {path_to_main} --seed {seed[i][i_seed]} --lr {learning[i]} --dp {dp[i]}  --out {output[i]} --dsDir {dataset_dir[i]} --glove {glove[i]} --BS {BS[i]} --eps {epsilons[i]}  --del {deltas[i]} --alpha {alpha[i]} --noiseM {noise_mul[i]} --ES {early_stop[i]}\n')
            f.write('sleep 1\n')
        st = os.stat(fullpath)
        os.chmod(fullpath, st.st_mode | stat.S_IEXEC)

