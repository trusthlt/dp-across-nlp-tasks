#!/bin/bash
#
#SBATCH --job-name=<name>
#SBATCH --output=<output file>
#SBATCH --error=<error file>
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
source <path to conda env>/<conda_env>/bin/activate
python <pathToRepo>/dp-across-nlp-tasks/NLPCode/named_entity_recognition/LSTM/experiment_parameters.py --seed 6546 --lr 0.001 --dp 1  --out <output dir>/ --dsDir <dataset dir>/ --glove <glove dir> --BS 32 --eps NONE  --del 1e-05 --alpha 17 --noiseM 1.24 --ES 0
sleep 1
