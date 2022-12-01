import numpy as np
import os
import random
import boto3
import awscli
from nltk.translate.bleu_score import sentence_bleu
from dotenv import load_dotenv
from google.cloud import translate

class TranslationEvaluationENES():
    def __init__(self):
        #Instantiate both Google and AWS Client using API keys
        
        #Load .env variables
        load_dotenv()
        
        #Change this variable to your Google api key location
        JSON_PATH = os.getenv('GOOGLE_JSON_PATH')
        #Change this variable to your Google project ID
        PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID')
        self.parent = f"projects/{PROJECT_ID}"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = JSON_PATH

        self.google_client = translate.TranslationServiceClient()
        
        #Copy-paste keys here
        AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')

        #Creates an AWS session using keys
        self.AWS_session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN
        )

    #Reads english text with proper encoding (returns list of sentences)    
    def read_english(self, en_path):
        with open(en_path, "r", encoding="utf-8") as f:
            return f.readlines()

    #Reads spanish text with proper encoding (returns list of sentences)  
    def read_spanish(self, es_path):
        with open(es_path, "r", encoding="iso8859") as f:
            return f.readlines()
        
    #Generates a preprocessed pair of lists. Due to constraints of the BLEU score function, the reference dataset needs to be split futher into words for each sentence.
    #Reference is the dataset whose language is intended to be evaluated with the BLUE score. Source language is the dataset to be translated.
    def get_subset_pair(self, reference, source, n_sentences):
        n = len(reference)
        
        reference_sub = []
        source_sub = []

        for i in range(n_sentences):
            index = random.randrange(n)
            reference_sub.append(reference[index].split())
            source_sub.append(source[index])

        return [reference_sub, source_sub]
    
    #Uses google's API to return a list of translated sentences
    def google_translate_sentence(self, texts, target):
        response = self.google_client.translate_text(contents=texts, target_language_code=target, parent=self.parent)
        translations = []

        for translation in response.translations:
            translations.append(translation.translated_text)    

        return translations
    
    #Uses AWS API to translate a set of texts
    def aws_translate_sentence(self, texts, source, target):
        translate = self.AWS_session.client(service_name='translate', region_name='us-east-1', use_ssl=True)
        translations = []

        for text in texts:
            translations.append(translate.translate_text(Text=text, SourceLanguageCode=source, TargetLanguageCode=target).get('TranslatedText'))

        return translations
    
    #The implemented function uses 1-gram BLEU score and computes the average for all provided translations
    #The index of a translation must correspond to the index of its reference
    def average_bleu_score(self, references, translations):
        N = len(references)
        acc = 0

        for i in range(N):
            acc += sentence_bleu([references[i]], translations[i].split(), weights=(1, 0, 0, 0))

        return acc




