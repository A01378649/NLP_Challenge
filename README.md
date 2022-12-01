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
The <code>seqeval</code> module provided the functionality to compute metrics of the model:

Here are the graphs for the metrics during training over the full dataset for 10 epochs:

![batch loss](https://github.com/A01378649/NLP_Challenge/blob/main/loss.png?raw=true)

![epoch metrics](https://github.com/A01378649/NLP_Challenge/blob/main/split.png?raw=true)

I picked **Google** and **AWS** translation APIs to complete the last one.
It is worth noting that the 1-gram BLEU score is computed for score comparison between them.

The datasets for the first and third tasks were provided by the professor.

### Installing

First create a <code>virtualenv</code>
	
	virtualenv <env_name>

Head to <env_name>/Scripts and execute:
	
	activate

I used a normal Python 3.9 virtual environment. For dependency installation, run this command:

	pip install -r requirements.txt

Furthermore *API Keys* for AWS and Google API keys should be provided in the ***.env*** file.

### Executing program

Just run the script:
	
	python run.py

### Tests

Head to the *tests* folder and execute each py file containing the set of unit tests for each functionality (one file per task).

	python <testname.py>

## Authors

Andrick Perusquia