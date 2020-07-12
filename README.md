# Global Relational Models of Source Code
This repository contains the data and code to replicate our [ICLR 2020 paper](http://vhellendoorn.github.io/PDF/iclr2020.pdf) on models of source code that combine global and structural information, including the Graph-Sandwich model family and the GREAT (Graph-Relational Embedding Attention Transformer) model.

This repository will also serve as the *public benchmark* for any models of source code that address this task on our dataset. Now that the subset of our paper's data that allows public release is available (see [Status](#status)), we will retrain the models described in our work and track both their, and any newly submitted models', performance on this page.

## Quick Start
The modeling code is written in Python (3.6+) and uses Tensorflow (recommended 2.2.x+). For a quick setup, run `pip install -r requirements.txt`.

To run training, first clone the [data repository](https://github.com/google-research-datasets/great) and note its location (lets call it `*data_dir*`). Then, from the main directory of this repository, run: `python running/train.py *data_dir* vocab.txt config.yml`, to train the model configuration specified in `config.yml`, periodically writing checkpoints (to `models/` and evaluation results (to `log.txt`). Both output paths can be optionally set with `-m` and `-l` respectively.

To customize the model configuration, you can change both the hyper-parameters for the various model types available (transformer, GREAT, GGNN, RNN) in `config.yml`, and the overall model architecture itself under `model: configuration`. For instance, to train the RNN Sandwich architecture from our paper, set the RNN and GGNN layers to reasonable values (e.g. RNN to  2 layers and the GGNN's `time_steps` to \[3, 1\] layers as in the paper) and specify the model configuration: `rnn ggnn rnn ggnn rnn`.

## Status (07/09/2020)
Update: as of July 9th, 2020, the data has been [released](https://github.com/google-research-datasets/great). I reconstructed the data loading & model running setup today (and fixed some bugs in the [models](#code)) and am currently running the various benchmarks from the paper. There are probably still a few small bugs in the code, but the general setup from the paper works: just modify the architecture(s) in config.yml, especially the model description under `model: configuration` to any configuration as desired (e.g. `great`, `rnn ggnn rnn ggnn`, etc.).

## Data
The data for this project consists of up to three bugs per function for every function in the re-releasable subset of the Py150 corpus, paired with the original, non-buggy code. This data is now publicly available from [https://github.com/google-research-datasets/great](https://github.com/google-research-datasets/great).

Secondly, we will release a real-world evaluation dataset containing bugs found in real Github programs with their descriptors.

## Code
We proposed a broad family of models that incorporate global and structural information in various ways. This repository provides an implementation of both each individual model (in `models`) and a library for combining these into arbitrary configurations (including the Sandwich models described in the paper, in `running`) for the purposes of joint localization and repair tasks. This framework is generally applicable to any task that transforms code tokens to states that are useful for downstream tasks.

Since the models in this paper rely heavily on the presence of "edge" information (e.g., relational data such as data-flow connections), we also provide a library for reading such data from our own JSON format and providing it to the various models. These files additionally contain meta-information for the task of interest in this paper (joint localization and repair), for which we provide an output layer and train/eval optimizer loop. These components are contingent on the data release [status](#status).

## Benchmark Results
The following parameters ought to be held fixed for all models, most of which are set correctly by default in config.yml:

- Each model is trained on 25 million samples (repeating the training data ~14 times) on functions up to 512 tokens (real tokens, not BPE) <sup>1</sup>.
- The models are evaluated every 250,000 (quarter million) samples on the same ~25,000 held-out samples.
- Assessed on the full eval portion of the dataset, along the accuracy metrics described below.
- Every model uses the same shared embedding and prediction layer (averaged sub-token embedding and two-pointer prediction) and differs only in how the embedding states are transformed into the states used to predict the location and repair by the model.
- The included vocabulary is used, which contains the 14,280 BPE sub-tokens that occur at least 1,000 times in the training data (from tokens no longer than 15 characters).
- Tentatively, at most 10 sub-tokens are included per token.
- Using the same, referenced [dataset](https://github.com/google-research-datasets/great) which includes Python functions and a wide range of edge types.
- Where possible, all models are run on a single *NVidia RTX Titan GPU* with 24GB of memory. If not the case, this should be noted; the memory size strongly dictates the batch size, which can make a large difference in ultimate performance.

<sup>1</sup>: Note that this is substantially larger than the threshold in the paper (250 tokens); this increase is to support generalization to real bugs, which tend to occur in longer functions, and makes little difference in training time on this dataset (average functions span just ~100 tokens).

The following results and variables should be reported for each run:

- The highest accuracy reached in 100 steps in terms of: no bug prediction accuracy (indicates false alarm rate), bug localization accuracy, bug repair accuracy, and joint bug localization and repair accuracy.
- The time needed to train on the full dataset *and* to train to converged accuracy (assumed to be the highest joint localization & accuracy reached in 100 steps).
- The total number of parameters used by the model (printed at the start of training).
- The maximum batch size in terms of total tokens; batchers are grouped by similar size for efficiency. Users are encouraged to use the default (12,500) for comparability.
- The learning rate.
- The model architecture (new innovations, with paper, are encouraged!) and hyper-parameters (e.g. hidden dimension)

Results: W.I.P.
