# NLP Challenge

Python implementation of the solution for the 3 tasks:

1. Out of the Box Sentiment Analysis 
2. NER
3. Translation APIs comparison

## Description

This project uses primarily HuggingFace and Tensorflow/Keras for model implementation and training/testing.

For the first task, a transformer pipeline was used to extract the sentiment for each review.

The second task uses the *distilbert_base_uncased* transformer model for training. 
Additionally, the **English** subset of the *wikiAnn* dataset was used for training, testing and validation.
The <code>seqeval</code> module provided the functionality to compute metrics of the model.

I picked **Google** and **AWS** translation APIs to complete the last one.
It is worth noting that the 1-gram BLEU score is computed for score comparison between them.

The datasets for the first and third tasks were provided by the professor.

### Installing

I used anaconda to for dependency installation, so run this command in anaconda prompt:

	conda create --name <env_name> --file requirements.txt

Activation of the environment may be needed, in order to do so, run:
	
	conda activate <env_name>

Furthermore *API Keys* for AWS and Google API keys should be provided in the ***.env*** file.

### Executing program

Just run the script:
	
	python run.py

### Tests

Tests are performed at the end of the execution.

## Authors

Andrick Perusquia