# IR-proj

## Download the dataset

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
# Create the glove dataset
python create_glove_dataset.py
```