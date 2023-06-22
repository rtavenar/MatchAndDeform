# Match-And-Deform: Time Series Domain Adaptation through Optimal Transport and Temporal Alignment

This repository provides code and datasets used in the "Match-And-Deform" paper.

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

## Problem statement

Given two time series datasets $\textbf{X}$ and $\textbf{X}^{\prime}$, MAD is formally defined as:

$\text{MAD}(\textbf{X}, \textbf{X}^{\prime}) =  \arg\min_{\gamma \in\Gamma(\textbf{w},\textbf{w}^{\prime}), \pi \in \mathcal{A}(T, T^{\prime})} \langle \textbf{L}(\textbf{X},\textbf{X}^{\prime}) \otimes \gamma, \pi \rangle  $
$\text{MAD}(\textbf{X}, \textbf{X}^{\prime}) = \arg\min_{\gamma \in\Gamma(\textbf{w},\textbf{w}^{\prime})  \pi \in \mathcal{A}(T, T^{\prime})} \sum_{i,j} \sum_{s,t} d(x^i_s, x_t^{\prime j}) \pi_{s, t} \gamma_{ij}$

Here, $\textbf{L}(\textbf{X},\textbf{X}^{\prime})$ is a 4-dimensional tensor whose elements are $L^{i,j}_{s,t}=d(x^i_s,x^{\prime j}_t)$, with $d : \mathbb{R}^q \times \mathbb{R}^q \rightarrow \mathbb{R}^+$ being a distance.
$\otimes$ is the tensor-matrix multiplication. $\pi$ is a global DTW alignment between timestamps and $\gamma$ is a transport plan between samples from $\textbf{X}$ and $\textbf{X}^{\prime}$.

The optimization problem in previous equation can be further extended to the case of distinct DTW mappings for each class $c$ in the source data. This results in the following optimization problem, coined $|\mathcal{C}|\text{-MAD}$:

$$
 |\mathcal{C}|\text{-MAD}(\textbf{X}, \textbf{X}^\prime, \textbf{Y}) =
      \arg\min_{\gamma \in \Gamma(\textbf{w},\textbf{w}^{\prime}),  \forall c, \pi^{(c)} \in \mathcal{A}(T, T^{\prime})} \sum_{i,j} \sum_{s,t} L^{i,j}_{s,t}
      \pi^{(y^i)}_{s t} \gamma_{ij} \, .
$$     

In that case, $|\mathcal{C}|$ DTW alignments are involved, one for each class $c$. $\pi^{(y^i)}$ denotes the DTW matrix associated to the class $y^i$ of $x^i$.
This more flexible formulation allows adapting to different temporal distortions that might occur across classes.

When embedded into a deep neural network, MAD helps learning new representations of time series that both align the domains and maximizes the discriminative power of the network, allowing the model to reach state-of-the-art performances.

We minimize the following overall loss function  over $\{\pi^{(c)}\}_c$, $\gamma$, $\Omega$ and $\theta$ :

$$\mathcal{L}(\textbf{X}, \textbf{Y}, \textbf{X}^{\prime}) =
\frac{1}{n} \sum_{i} \mathcal{L}_{s}(y^{i}, f_\theta(g_\Omega(\textbf{x}^{i}))) +
\sum_{i, j} \gamma_{ij} \Big( \alpha \sum_{s, t} \pi_{s t}^{(y^i)}
L\left(g_\Omega(\textbf{X}), g_\Omega(\textbf{X}^\prime)\right)^{i,j}_{s, t} + \beta \mathcal{L}_{t}(y^{i}, f_\theta(g_\Omega(\textbf{x}^{\prime j}))) \Big)$$


where $\mathcal{L}_s$ and $\mathcal{L}_t$ are cross entropy losses, $\gamma$ (resp. $\{\pi^{(c)}\}_c$) is the transport plan (resp. the set of DTW paths) yielded by $|\mathcal{C}|\text{-MAD}$.

In pratice, $\alpha = 0.1$ is the value that equalizes terms in this sum and we set $\beta = 0.01$, a tenth of $\alpha$.


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
