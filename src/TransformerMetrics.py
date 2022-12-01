import numpy as np
import transformers
from transformers.keras_callbacks import KerasMetricCallback
from transformers import pipeline, AutoTokenizer, create_optimizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TFAutoModelForTokenClassification
from datasets import load_dataset, load_metric
from nltk.translate.bleu_score import sentence_bleu
import keras

class TransformerMetrics:
    def __init__(self, label_list):
        #The metrics to compute Transformer performance
        self.accuracy = []
        self.precision = []
        self.f1 = []
        self.recall = []
        self.label_list = label_list
        
    '''
    This method will use seqeval method to compute metrics of the model using a specified dataset split (defined in a later function)
    The tags for special tokens will be removed.
    This function was mostly unchanged from the source.
    '''
        
    def _compute_metrics(self, p):
        metric = load_metric("seqeval")

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        result_dc = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        return result_dc
    
    #We define the callbacks using auxiliary functions to update each metrics object
    def _update(self, p):
        metrics = self._compute_metrics(p)
        self.accuracy.append(metrics["accuracy"])
        self.precision.append(metrics["precision"])
        self.f1.append(metrics["f1"])
        self.recall.append(metrics["recall"])
        return metrics

    #Returns Keras Callback Object based on the private functions used to update elements of the lisy
    def get_callback(self, eval_dataset):
        return KerasMetricCallback(
            metric_fn=self._update, eval_dataset=eval_dataset
        )
