# One size does not fit all: Investigating strategies for differentially-private learning across NLP tasks

Companion code to our arXiv preprint.

Pre-print available at: https://arxiv.org/abs/2112.08159

Please use the following citation

```plain
@journal{Senge.et.al.2021.arXiv,
    title = {{One size does not fit all: Investigating strategies
              for differentially-private learning across NLP tasks}},
    author = {Senge, Manuel and Igamberdiev, Timour and Habernal, Ivan},
    journal = {arXiv preprint},
    year = {2021},
    url = {https://arxiv.org/abs/2112.08159},
}
```

> **Abstract** Preserving privacy in training modern NLP models comes at a cost. We know that stricter privacy guarantees in differentially-private stochastic gradient descent (DP-SGD) generally degrade model performance. However, previous research on the efficiency of DP-SGD in NLP is inconclusive or even counter-intuitive. In this short paper, we provide a thorough analysis of different privacy preserving strategies on seven downstream datasets in five different `typical' NLP tasks with varying complexity using modern neural models. We show that unlike standard non-private approaches to solving NLP tasks, where bigger is usually better, privacy-preserving strategies do not exhibit a winning pattern, and each task and privacy regime requires a special treatment to achieve adequate performance.

**Contact person**: Ivan Habernal, ivan.habernal@tu-darmstadt.de. https://www.trusthlt.org

*This repository contains experimental software and is published for the sole purpose of giving additional background details on the publication.*



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

