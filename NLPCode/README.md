# A Survey on the Effectiveness of Differential Privacy in the NLP Domain

## different branches
This repository consists of two branches. All code to train and evaluate the models is present in the master branch. The code that was used to analyze the models, e.g. for error analysis, can be found in the second branch 'model_analysis'.

## Structure of the master repository
This repository consists of 7 root folders. 5 of these folders are the assessed tasks. The folder 'privacy_computation' contains two python files to calculate the privacy budged with Opacus or Tensorflow. Opacus can be used to see how much privacy is used during the experiments. Tensorflow can be used to find the correct noise-multiplier and alpha, to reach a certain target epsilon.

In the folder 'example_run_script' two subfolders can be found. Each subfolder contains two Python scripts that generate bash files for the UKP or Lichtenberg cluster. Additionally, each file contains one example script that can be executed by slurm via sbatch. These scripts can be changed to start different tasks with different hyperparameters. 

Each directory named after a task contains a folder called 'LSTM' and 
'Transformer'. In these folders, all model-specific code can be found, along with the training and evaluation code. The python files 'utils' and 'tuning_structs' contain general classes and structs used for more readable code, as well as methods used for both architectures.

A list of all used libraries and their versions can be found in the root file used_libraries.txt

## Running the models
Since it is common practice to test various hyperparameters on the same model and dataset a special wrapper class is written to execute the training process with a certain set of hyperparameters. This wrapper class (called ExperimentalParameters) reads in console attributes to set the hyperparameters. One example in starting a training process for a transformer model could be:
```
python <PATH TO REPO>/dp_for_different_nlp_tasks/natural_language_inference/Transformer/experiment_parameters.py --seed 1234 --lr 1e-05 --dp 1 --BLay 1 --rnn 0  --out <OUTPUT DIR> --dsDir <PATH TO REPO>/dp_for_different_nlp_tasks/natural_language_inference/Transformer/snli_1.0/ --BS 8 --eps NONE  --del 1e-05  --alpha 10.0 --noiseM 0.6834 --ES 0
```
The list of parameters for the BERT model is the following:
<ul>
<li>--seed : specifies the seed of for the run</li>
<li>--lr : specifies the learning rate</li>
<li>--dp : if set to one differential privacy is enabled, if set to 0 it is disabled</li>
<li>--BLay : specifies the number of finetuned layers of BERT (0: freez all, 2: freez none, -1 to -11 finetune the last 1 - 11 layers, 1: freez only the embedding)</li>
<li>--rnn : if set to 1 adds an LSTM to the model, if set to 0 does not add the LSTM</li>
<li>--out : specifies the output directory for the model and text files</li>
<li>--dsDir : specify the directory of the dataset</li>
<li>--BS : specifies the batch size</li>
<li>--eps : specifies the target epsilon, if user does not want to specify the target epsilon set it to 'NONE'</li>
<li>--del : specifies the delta</li>
<li>--alpha : specifies the alpha</li>
<li>--noiseM : specifies the noise muliplier, if user does not want to specify the noise muliplier set it to 'NONE'</li>
<li>--ES enable (1) or dissable (0) early stopping (not available for all models)</li>
</ul>

One example in starting a training process for an LSTM model could be:
```
python <PATH TO REPO>/dp_for_different_nlp_tasks/natural_language_inference/LSTM/experiment_parameters.py --seed 6546 --lr 0.0001 --dp 1  --out <OUTPUT DIR>  --dsDir <PATH TO REPO>/dp_for_different_nlp_tasks/natural_language_inference/LSTM/.data --glove <PATH TO REPO>/dp_for_different_nlp_tasks/natural_language_inference/LSTM/.vector_cache --BS 32 --eps NONE  --del 1e-05 --alpha 10 --noiseM 0.735 --ES 0
```
The list of parameters for the BERT model is the following:
<ul>
<li>--seed : specifies the seed of for the run</li>
<li>--lr : specifies the learning rate</li>
<li>--dp : if set to one differential privacy is enabled, if set to 0 it is disabled</li>
<li>--dsDir : specify the directory of the dataset</li>
<li>--glove : specify the directory of the glove embedding</li>
<li>--out : specifies the output directory for the model and text files</li>
<li>--BS : specifies the batch size</li>
<li>--eps : specifies the target epsilon, if user does not want to specify the target epsilon set it to 'NONE'</li>
<li>--del : specifies the delta</li>
<li>--alpha : specifies the alpha</li>
<li>--noiseM : specifies the noise muliplier, if user does not want to specify the noise muliplier set it to 'NONE'</li>
<li>--ES enable (1) or dissable (0) early stopping (not available for all models)</li>
</ul>

