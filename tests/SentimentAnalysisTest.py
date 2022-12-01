import sys

# setting path
sys.path.append('../.')

from src.SentimentAnalysis import SentimentAnalysis

SA = SentimentAnalysis()

REVIEWS_PATH = "../tiny_movie_reviews_dataset.txt"

def set_movie_reviews_test(path):
    SA.set_movie_reviews(path)
    assert len(SA._reviews) > 0
    
def truncate_review_test(pos, skip):
    pre = len(SA._reviews[pos])
    SA.truncate_review(pos,skip)
    assert len(SA._reviews[pos]) == pre - skip
    
def print_sentiment_test():
    assert SA.print_sentiment() == True

def run_tests():
    set_movie_reviews_test(REVIEWS_PATH)
    truncate_review_test(17, 259)
    print_sentiment_test()
    print("All tests were executed succesfull :^)")

run_tests()