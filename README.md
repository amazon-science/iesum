# Enhancing Multi-Document Summarization with Cross-Document Graph-based Information Extraction

This repository contains the code used for the paper:
[**Enhancing Multi-Document Summarization with Cross-Document Graph-based Information Extraction (EACL 2023)**](https://aclanthology.org/2023.eacl-main.124.pdf)

Zixuan Zhang, Heba Elfardy, Markus Dreyer, Kevin Small, Heng Ji, Mohit Bansal, EACL 2023.


### Overview

In this paper, we propose a multi-document text summarization model which is enhanced by cross-document Information Extraction (IE) graphs. Specifically, given a cluster of documents related to the same topic, we first use a cross-document fine-grained IE system to extract a cluster-level information graph. Then we use an edge-conditioned graph attention network to encode the IE graph and to merge the graph information into the sequence-to-sequence summary generation pipeline. To better utilize the signals from IE, we further propose two novel training objectives. First, to help the model better recognize and remember the important events and entities, we propose an auxiliary task of entity and event recognition, where an additional classification module is incorporated to train the model to select the important entities and event triggers while performing summarization. Second, to mitigate the errors and inconsistencies caused by noise in the data, we propose a graph and text alignment loss that minimizes the distance between IE graph nodes and their corresponding text segments in a shared latent embedding space. A detailed model design is shown as follows.

<img width="1483" alt="image" src="https://user-images.githubusercontent.com/80446849/205963630-4363acef-a15e-4859-bc40-4048c81ab4fe.png">

### Code Installation
Our code is tested on Python `3.9.0` and PyTorch `1.12.1` with CUDA `11.3`. To install the environment, please first create your own Python virtual environment and then run
```
pip install -r requirements.txt
```
Deep Graph Library (DGL) for the graph neural networks used in our model, will then need to be installed. 
To install the latest version of DGL, please run:
```
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
```

### Data Preparation
In this project, we perform experiments on three datasets [Multi-News](https://github.com/Alex-Fabbri/Multi-News), [WCEP](https://github.com/complementizer/wcep-mds-dataset) and [DUC-2004](https://duc.nist.gov/duc2004/). 
#### Running Information Extraction (IE) models to obtain the graphs.
In this project, we need to run IE systems prior to model training to first get the graphs for each document cluster. We first use [RefinED](https://github.com/amazon-science/ReFinED) to extract entities in each cluster and then use [RESIN-11](https://github.com/RESIN-KAIROS/RESIN-11) to extract events. Please make sure that the original data and the IE results are processed into the following `json` format:
```
{
    "articles": [
        "...",
        ..."
    ],
    "summary": " ...",
    "nodes": {
        "0": {
                "type": "entity",
                "spans": [
                    [
                        0,
                        24,
                        29
                    ],
                    [
                        1,
                        345,
                        350
                    ],
                    [
                        1,
                        793,
                        808
                    ]
                ]
            },
        "1": ...
    }
    "edges": {
            "0": [
                "1",
                "2"
            ],
            "1": [
                "2"
            ]
        }
}
```
where each training example has four dict keys: 
+ `articles`: a list of strings for the input document cluster
+ `summary`: the reference summary
+ `nodes`: a `dict` where each `key` is the string ID of an IE node, and each value contains `type` and `spans` of the node. Each `span` is a 3-tuple that represents `[document_id, start_offset, end_offset]`.


### Running Examples
We provide two bash scripts `train_multinews.sh` and `train_wcep.sh` for training and testing models on `MultiNews` and `WCEP` respectively. To train the model and reproduce the results, just run:

```
bash train_multinews.sh
bash train_wcep.sh
```

## License

The code is licensed under Amazon Software License [here](LICENSE).
