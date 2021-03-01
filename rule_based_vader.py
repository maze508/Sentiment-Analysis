from vaderSentiment.vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

analyzer = SentimentIntensityAnalyzer()

sentence = "The food there is quite literally the worst I've ever tasted" 

#! Checking Sentiment Score
def sentiment_analyser(sentence):
    score = analyzer.polarity_scores(sentence)
    print(f"{sentence} : {str(score)}")
    return score

score = sentiment_analyser(sentence)

print(score)
print(score['compound'])