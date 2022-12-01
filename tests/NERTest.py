import sys

# setting path
sys.path.append('../.')

from src.NER import NER
from src.NER_TF_Utils import NER_TF_Utils
from src.TransformerMetrics import TransformerMetrics
import tensorflow as tf
from transformers import DataCollatorForTokenClassification
from transformers.keras_callbacks import KerasMetricCallback
from transformers.optimization_tf import AdamWeightDecay

utils = NER_TF_Utils(8, 2e-5, 5)
transf_metrics = TransformerMetrics(["a","b","c","d","e","f","g"])
ner = NER("distilbert-base-uncased")


def get_TF_model_test(model_checkpoint, num_labels):
    assert utils.get_TF_model(model_checkpoint, num_labels) != True
    
def get_tf_optimizer_test(n_steps):
    assert isinstance(utils.get_tf_optimizer(n_steps), AdamWeightDecay) == True
    
def get_callback_test():
    model = utils.get_TF_model("distilbert-base-uncased", 7)
    model.compile(optimizer=utils.get_tf_optimizer(1000))
    td = ner.get_tokenized_datasets()
    data_collator = DataCollatorForTokenClassification(ner.tokenizer, return_tensors="tf")
    train_set, validation_set, test_set = utils.get_model_splits(model, td, data_collator)
    
    assert isinstance(transf_metrics.get_callback(test_set), KerasMetricCallback)
    
def load_dataset_test(name, lang):
    ner.load_dataset(name, lang)
    assert ner.datasets != None 
    
def set_tokenizer_test():
    ner.set_tokenizer()
    assert ner.tokenizer != None

def run_tests():
    get_TF_model_test("distilbert-base-uncased", 7)
    get_tf_optimizer_test(100)
    load_dataset_test("wikiann","en")
    set_tokenizer_test()
    get_callback_test()
    print("All tests were executed succesfull :^)")

run_tests()