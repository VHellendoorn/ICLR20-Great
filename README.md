# Global Relational Models of Source Code
This repository will contain the data and code replication for our [ICLR 2020 paper](http://vhellendoorn.github.io/PDF/iclr2020.pdf) on models of source code that combine global and structural information, including the Graph-Sandwich model family and the GREAT (Graph-Relational Embedding Attention Transformer) model.

This repository will also serve as the *public benchmark* for any models of source code that address this task on our dataset. Once the (permissable subset of our) data is available for release (see [Status](#status)), we will retrain the models described in our work and track both their, and any newly submitted models', performance in this document.

## Status
As of April 17th, the data used has not yet been released due to some minor legal constraints with re-releasing some of the repositories included in this dataset (ETH's Py150) data. We are working on filtering the dataset, removing the small subset of projects that cannot be re-released, and will release the full dataset here when ready. When this happens, we will also rerun all the models described in the paper and publish their results here, as mentioned above, to provide a fair comparison with other work, as the underlying data will necessarily be slightly different compared to the published paper.

While waiting for the above, we are releasing implementations of individual [models](#code) for those interested; the more complete library that allows arbitrary composition and a complete input -> prediction pipeline will follow as soon as the data is available.

## Data
The data for this project consists primarily of up to three bugs per function for every function in the re-releasable subset of the Py150 corpus, paired with the original, non-buggy code. Secondly, we will release a Github dataset containing bugs found in real Github programs with their descriptors.

As described under [Status](#status); we will release the full dataset as soon as it becomes feasible.

## Code
We proposed a broad family of models that incorporate global and structural information in various ways. The 'code' directory on this repository provides an implementation of both each individual model and a library for combining these into arbitrary configurations (including the Sandwich models described in the paper) for the purposes of joint localization and repair tasks. This framework is generally applicable to any task that transforms code tokens to states that are useful for downstream tasks.

Since the models in this paper rely heavily on the presence of "edge" information (e.g., relational data such as data-flow connections), we also provide a library for reading such data from our own JSON format and providing it to the various models. These files additionally contain meta-information for the task of interest in this paper (joint localization and repair), for which we provide an output layer and train/eval optimizer loop. These components are contingent on the data release [status](#status).

