# Personalized News Recommendation

This repository contains PyTorch implementations of trend-aware news recommendation methods, created for Final project in DS535 class.



## Base model NRMS

| Name   | Paper                                                                                                               | Notes                               |
|--------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| NRMS   | [Neural News Recommendation with Multi-Head Self-Attention](https://www.aclweb.org/anthology/D19-1671)              |                                     |



## Getting started

After cloning, install dependencies using [Poetry](https://python-poetry.org/):

    poetry install

    poetry run python src/sample_data.py

Training a model 

    poetry run python src/train_recommender.py +model=nrms

After training, files containing checkpoints and metrics can be found in the `outputs/` directory.


## Acknowledgements

- We use baseline code from [https://github.com/koengommers/news-recommendation]
- Microsoft News Dataset (MIND), see [https://msnews.github.io/](https://msnews.github.io/).
- NRMS are adapted from implementation of yusanshi, see [https://github.com/yusanshi/news-recommendation](https://github.com/yusanshi/news-recommendation).
