import tensorflow
import random
import transformers
from transformers.keras_callbacks import KerasMetricCallback
from transformers import pipeline, AutoTokenizer, create_optimizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TFAutoModelForTokenClassification
from datasets import load_dataset, load_metric
from nltk.translate.bleu_score import sentence_bleu
import keras

class NER_TF_Utils():
    def __init__(self, bs, lr, epochs):
        self.batch_size = bs
        self.lr = lr
        self.epochs = epochs

    #Instantiates Tensorflow version of the HuggingFace model
    def get_TF_model(self, model_checkpoint, num_labels):
        return TFAutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    #Create an optimizer without defining loss because an external tool is used
    def get_tf_optimizer(self, n_steps):
        optimizer, lr_schedule = create_optimizer(
            init_lr=self.lr,
            num_train_steps=n_steps,
            weight_decay_rate=0.01,
            num_warmup_steps=0)

        return optimizer
    
    #Get dataset splits for training, validation and test
    def get_model_splits(self, model, datasets, data_collator, tr_percent=0.1, v_percent=0.1, te_percent=0.1):
        n_train = int(len(datasets["train"]) * tr_percent)
        n_validation = int(len(datasets["validation"]) * v_percent)
        n_test = int(len(datasets["test"]) * te_percent)

        return[model.prepare_tf_dataset(
            datasets["train"].select(range(n_train)),
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=data_collator
            ),

        model.prepare_tf_dataset(
            datasets["validation"].select(range(n_validation)),
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=data_collator
            ),

        model.prepare_tf_dataset(
            datasets["test"].select(range(n_test)),
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=data_collator
            )]
