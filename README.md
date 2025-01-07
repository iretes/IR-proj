# Product Quantization for Nearest Neighbor Search

This project implements the method proposed by Jégou et al. in their article, "Product Quantization for Nearest Neighbor Search" [1]. It replicates the experimental results presented in the paper and explores alternative strategies that extend and build upon the original approach.

## How to Replicate the Experiments

To replicate the experiments, install the dependencies executing the following command:
```bash
pip install -r requirements.txt
```

The datasets can be be downloaded by running the following commands:
```bash
# Download the siftsmall dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz -O siftsmall.tar.gz
tar -xvzf siftsmall.tar.gz

# Download the sift dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -O sift.tar.gz
tar -xvzf sift.tar.gz

# Download the glove dataset
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip -O glove.6B.zip
unzip glove.6B.zip -d glove
# Subsample the glove dataset and convert it to numpy arrays
python create_glove_dataset.py
```

## Directory structure

The project's directory structure includes the following main files and folders:

```
IR-proj # TODO:
  │── docs # folder storing the html documentation
  |── glove # folder storing the glove dataset
  |── img # folder storing images of the experiments
  |── sift # folder storing the sift dataset
  |── siftsmall # folder storing the siftsmall dataset
  |── create_glove_dataset.py # generates the glove dataset
  |── faiss_comparison.ipynb # compares the performance of the faiss library with the implemented method
  |── large_scale_experiments.ipynb # large scale experiments
  |── results_comparison.md # comparison of the results obtained in the original article
  |── search_approaches.py # implementation of the search approaches
  |── small_scale_experiments.ipynb # small scale experiments
  └── utils.py # implementation of utility functions
```

## References

[1] Hervé Jégou, Matthijs Douze and Cordelia Schmid. "Product Quantization for Nearest Neighbor Search". IEEE transactions on pattern analysis and machine intelligence 33.1 (2010): 117-128.

---

This project was developed for the “Information Retrieval” course at the University of Pisa (a.y. 2024/2025).