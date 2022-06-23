#!/bin/bash
#
#SBATCH -J QLNe1_8672
#SBATCH -o <output file>
#SBATCH -e <error file>
#SBATCH -n 1
#SBATCH --mem-per-cpu=8000
#SBATCH -c 2
#SBATCH -A project01460
#SBATCH -t 01-00
#SBATCH --gres=gpu:1
#SBATCH -p testgpu24
ml cuda
source <path to conda env>/miniconda3/bin/activate pytorch
python <pathToRepo>/dp-across-nlp-tasks/NLPCode/natural_language_inference/Transformer/experiment_parameters.py --seed 8672 --lr 1e-05 --dp 1 --BLay 0 --rnn 1  --out <output dir>/ --dsDir <dataset dir>/ --BS 32 --eps NONE  --del 1e-05  --noiseM 0.85 --alpha 11 --ES 0
