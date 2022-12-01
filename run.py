import matplotlib.pyplot as plt
from transformers import DataCollatorForTokenClassification
from src.SentimentAnalysis import SentimentAnalysis
from src.NER import NER
from src.NER_TF_Utils import NER_TF_Utils
from src.TransformerMetrics import TransformerMetrics
from src.LossHistory import LossHistory
from src.TranslationEvaluationENES import TranslationEvaluationENES

#----------------------RUN-------------------------

#PART 1
print("Part 1:\n-------------------Out of the Box Sentiment Analysis----------------------")

REVIEWS_PATH = "tiny_movie_reviews_dataset.txt"

task1 = SentimentAnalysis()

#Extracting the reviews ot txt into a list...
task1.set_movie_reviews(REVIEWS_PATH)

#The 18th review is too long so we truncate its beginning elements
task1.truncate_review(17, 259)

#We introduce each review into the pipeline and extract the result
task1.print_sentiment()

#PART 2
print("Part 2:\n-------------------NER----------------------")
'''
This is a tweaked version of this notebook: https://github.com/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb
'''

#We are going to use the "distilled" (light) version of the multilingual cased BERT transformer
model_checkpoint = "distilbert-base-uncased"
batch_size = 10                                        #Batch size has to be this low because the gpu can't handle any bigger batch sizes
LR = 2e-5
epochs = 5
TRAIN_PERCENTAGE = 0.01
VALIDATION_PERCENTAGE = 0.01
TEST_PERCENTAGE = 0.01

#Instantiating helper classes
ner = NER(model_checkpoint)                                     #Dataset processing
utils = NER_TF_Utils(batch_size, LR, epochs)                    #Tensorflow utilities for training and evaluation

#WikiANN dataset consists of Wikipedia articles with their respective NER annotations
ner.load_dataset("wikiann","en")                                    #We are going to choose the english subset of the database

#Initializing tokenizer to generate features important to the transformer
ner.set_tokenizer()

#Generating new features for transformer processing
tokenized_datasets = ner.get_tokenized_datasets()

label_list = ner.datasets["train"].features["ner_tags"].feature.names
model = utils.get_TF_model(model_checkpoint, len(label_list))

num_train_steps = (len(tokenized_datasets["train"]) // batch_size) * epochs
model.compile(optimizer=utils.get_tf_optimizer(num_train_steps))

#Collator for data preprocessing like adding padding or even some data augmentation for each batch
data_collator = DataCollatorForTokenClassification(ner.tokenizer, return_tensors="tf")

#Creating splits based on provided percentage
train_set, validation_set, test_set = utils.get_model_splits(model, tokenized_datasets, data_collator, TRAIN_PERCENTAGE, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)

#Instantiating keras metrics callbacks for each split...
train_metrics = TransformerMetrics(label_list)
validation_metrics = TransformerMetrics(label_list)
test_metrics = TransformerMetrics(label_list)

tr_callback = train_metrics.get_callback(train_set)
val_callback = validation_metrics.get_callback(validation_set)
test_callback = test_metrics.get_callback(test_set)
batch_history = LossHistory()

#Putting it all together to train the model

callbacks = [tr_callback, val_callback, test_callback, batch_history]

history = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=epochs,
    callbacks=callbacks
)

#Plotting loss per batch
plt.xlabel("Batch #")
plt.ylabel("Loss")
plt.title("Loss variation")

plt.plot(batch_history.losses, linewidth=1, color='green')
plt.show()

#Plotting accuracy for the three splits (more than 1 epoch to be appreciated)
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.title("Split comparison")

plt.plot(train_metrics.accuracy, linewidth=3, color='blue', label = "Train")
plt.plot(validation_metrics.accuracy, linewidth=2, color='yellow', label = "Validation")
plt.plot(test_metrics.accuracy, linewidth=3, color='red', label = "Test")

plt.legend()

plt.show()

#PART 3
print("Part 3:\n-------------------Translation API's comparison----------------------")

N_SENTENCES = 100                     #Number of samples to take
EN_PATH = "dataset_en.txt"                 #English dataset 
ES_PATH = "dataset_es.txt"                 #Spanish dataset

evaluator = TranslationEvaluationENES()

#Read txt files with proper encoding
en_sentences = evaluator.read_english(EN_PATH)
es_sentences = evaluator.read_spanish(ES_PATH)

#Generates a preprocessed pair of lists. Due to constraints of the BLEU score function, the reference dataset needs to be split futher into words for each sentence.
#Reference is the dataset whose language is intended to be evaluated with the BLUE score. Source language is the dataset to be translated.
en_subset, es_subset = evaluator.get_subset_pair(en_sentences, es_sentences, N_SENTENCES)
google_translations = evaluator.google_translate_sentence(es_subset, "en")
aws_translations = evaluator.aws_translate_sentence(es_subset, "es", "en")

#Obtaining scores for both APIs
google_score = evaluator.average_bleu_score(en_subset, google_translations)
aws_score = evaluator.average_bleu_score(en_subset, aws_translations)

#Print results
print(f"AMAZON_TRANSLATOR: {aws_score}")
print(f"GOOGLE_TRANSLATOR: {google_score}")
