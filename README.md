# Match-And-Deform: Time Series Domain Adaptation through Optimal Transport and Temporal Alignment

This repository provides code and datasets used in the "Match-And-Deform" paper.

Link to the paper: <https://arxiv.org/abs/2308.12686>

## Paper abstract: 

> While large volumes of unlabeled data are usually available, associated labels are often scarce. The unsupervised domain adaptation problem aims at exploiting labels from a source domain to classify data from a related, yet different, target domain.
> When time series are at stake, new difficulties arise as temporal shifts may appear in addition to the standard feature distribution shift.
> In this paper, we introduce the Match-And-Deform (MAD) approach that aims at finding correspondences between the source and target time series while allowing temporal distortions. The associated optimization problem allows simultaneously aligning the series thanks to an optimal transport loss and the time stamps through dynamic time warping. When embedded into a deep neural network, MAD helps learning new representations of time series that both align the domains and maximizes the discriminative power of the network.
> Empirical studies on benchmark datasets and remote sensing data demonstrate that MAD makes meaningful sample-to-sample pairing and time shift estimation, reaching similar or better classification performance than state-of-the-art deep time series domain adaptation strategies.

## Citation

When citing this work, please use the following BibTeX entry:

```
@inproceedings{mad2023,
  TITLE = {{Match-And-Deform: Time Series Domain Adaptation through Optimal Transport and Temporal Alignment}},
  AUTHOR = {Painblanc, François and Chapel, Laetitia and Courty, Nicolas and Friguet, Chloé and Pelletier, Charlotte and Tavenard, Romain},
  BOOKTITLE = {{European Conference on Machine Learning and Principles and Practice of Knowledge Discovery}},
  YEAR = {2023}
}
```


## Installation:

In order to make sure you have the correct package versions installed, you should run:

```bash
pip install -r requirements.txt
```

## Training models:

Training a model is done by calling the `main.py` command-line script.
There are several options that can be chosen when training a model :

```{python}
#Training specification:
    --iteration: {int} number of iterations
    --name_model: {str} name of the model for saving purposes
    --seed: {int} set seed for reproducible experiments

#Model specification:
    --model: {str: "MAD", "DeepJDOT-DTW", "basic CNN"} which model to use during training
    --per_class_dtw: {str: "True", "False"} in case the model is MAD, choses between MAD and C-MAD
    --target_class_proportion: {None or a list of int} proportion of classes in target dataset in case of weakly supervised learning

#Dataset
    --dataset_name: {str: "ucihar", "TarnBrittany", "MiniTimeMatch"} which dataset to train the model on
    Inside a dataset, domains are indicated by a number. 
    For "HAR" the domain id corresponds to one of the participant. 
    For "TarnBrittany", domain 1 corresponds to Tarn and domain 2 to Brittany.
    For "MiniTimeMatch", domain 1 correponds to FR1, domain 2 to FR2, domain 3 to DK1 and domain 4 to AT1
        --source_id : {int} the source domain
        --target_id : {int} the target domain

#Hyper-parameters:
    --alpha : the value of alpha in the loss equation
    --beta : the value of beta in the loss equation
    --learning_rate : learning rate of the optimizer
    --batchsize: the number of series in the batches (equal number for source and target)
```

### To train a model:

The python file `main.py` should be used to run models.

```{python}
python3 main.py --dataset_name "TinyTimeMatch" --model "MAD" --source_id 3 --target_id 4 --alpha 0.1 --beta 0.01 --learning_rate 0.0001 --iteration 3000 --name_model "Demo" --seed 100 --target_class_proportion "True" --batchsize 256 --per_class_dtw "True"
```

## Results in the papers

Each experiment are repeated 3 times and can be run via the python file `main.py`.

The batchsize is set to 256 for each experiment `--batchsize 256`

The learning rate is set to 0.0001 for each experiment `--learning_rate 0.0001`

For the models `Basic-CNN` (no adaptation) the number of iteration is set to 30,000, which corresponds to the number of iterations of `CoDATS` (not included) `--iteration 30000`

For the models `C-MAD`, `MAD` and `DeepJDOT-DTW` the number of iterations is set to 3,000 `--iteration 3000`

Moreover, for those last three models, the hyperparameter $\alpha$ is set to 0.1 and the hyperparameter $\beta$ to 0.01 (a tenth of $\alpha$) `--alpha 0.1 --beta 0.01`

To include the weakly supervised setup (only for `MAD` and `C-MAD`): `--target_class_proportion "True"`
This will calibrate the weights of the source domain during the transport step to be equal in proportion to the classes in the target domain.
And to not include this option, just remove `--target_class_proportion` from the command line.

### To reproduce results from the paper for the HAR dataset:

The HAR dataset is expected to be stored under `Dataset/ucihar`.
The dataset features 30 domains but only ten are used during the experiments.
