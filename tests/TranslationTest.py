import sys

# setting path
sys.path.append('../.')

from src.TranslationEvaluationENES import TranslationEvaluationENES
import string

evaluator = TranslationEvaluationENES()
EN_PATH = "../dataset_en.txt"                 #English dataset 
ES_PATH = "../dataset_es.txt"                 #Spanish dataset

def read_english_test(en_path):
    assert len(evaluator.read_english(en_path)) > 0

def read_spanish_test(es_path):
    assert len(evaluator.read_spanish(es_path)) > 0

def get_subset_pair_test(reference, source, n_sentences):
    a, b = evaluator.get_subset_pair(reference, source, n_sentences)
    assert len(a) == len(b)
                                       
def aws_translate_sentence_test(texts, source, target):
    assert len(evaluator.aws_translate_sentence(texts, source, target)) > 0
                                       
def google_translate_sentence_test(texts, target):
    assert len(evaluator.google_translate_sentence(texts, target)) > 0

def average_bleu_score_test(references, translations):
    assert evaluator.average_bleu_score(references, translations) >= 0

def run_tests():
    read_english_test(EN_PATH)
    read_spanish_test(ES_PATH)
    es = ["Esto es una prueba de NLP.", "La gallina cruzó la calle.", "Hola"]
    en = ["This is a NLP test.", "The hen crossed the street.", "Hello"]
    get_subset_pair_test(es, en, 1)
    t, s = evaluator.get_subset_pair(es, en, 1)
    aws_translate_sentence_test(s, "es", "en")
    google_translate_sentence_test(s, "en")
    translations = ["This is a random sentence"]
    average_bleu_score_test(t, translations)
    print("All tests were executed succesfull :^)")

run_tests()