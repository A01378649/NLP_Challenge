from transformers import pipeline

class SentimentAnalysis():
    def __init__(self):
        self._reviews = []
        self._classifier = pipeline("sentiment-analysis")
        
    def set_movie_reviews(self, path):
        with open(path, "r") as f:
            self._reviews = f.readlines()
        
    def truncate_review(self, pos, skip):
        self._reviews[pos] = self._reviews[pos][skip:]
        return True
        
    def print_sentiment(self):
        results = self._classifier(self._reviews)

        for result in results:
            print(result['label'])

        return True
        
