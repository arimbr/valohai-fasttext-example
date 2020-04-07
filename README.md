# Production Machine Learning Pipeline for Text Classification

This repository is the support for two articles:
- [Production Machine Learning Pipeline for Text Classification with fastText](https://blog.valohai.com/production-machine-learning-pipeline-text-classification-fasttext)
- [Classifying 4M Reddit posts in 4k subreddits: an end-to-end machine learning pipeline](https://blog.valohai.com/machine-learning-pipeline-classifying-reddit-posts)

Libraries and code structure:
- [fastText](https://fasttext.cc/) is library for efficient text classification and representation learning. Check the code in [commands.py](https://github.com/arimbr/valohai-fasttext-example/blob/master/models/classification/commands.py) to see how to use fastText's Python bindings.
- [Valohai](https://valohai.com) is a machine learning platform that automates MLOps and record keeping. Check the code in [valohai.yaml](https://github.com/arimbr/valohai-fasttext-example/blob/master/valohai.yaml) to see how to integrate your custom ML code with Valohai.
- [FastAPI](https://fastapi.tiangolo.com/) is a web framework for high performance, easy to learn, fast to code and ready for production. Check the code in [api.py](https://github.com/arimbr/valohai-fasttext-example/blob/master/api.py) to see how to create models and prediction endpoints.
