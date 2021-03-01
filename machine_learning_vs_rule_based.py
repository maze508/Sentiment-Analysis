import pandas as pd
import pickle
import spacy
import numpy as np
import sklearn as sk
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
import string
from spacy.lang.en import English
# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Sklearn Pipeline
from sklearn.pipeline import Pipeline
# Vader (Rule Based)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


#* Data Engineering
path = r"C:\Users\Admin\Downloads\archive\Tweets.csv"

raw_df = pd.read_csv(path)
# print(df)
col_name = []
for cols in raw_df.columns:
    col_name.append(cols)
print("Column Names :", col_name)
print()

airline_sentiment_list = raw_df["airline_sentiment"].tolist()
text_list = raw_df["text"].tolist()

df = pd.DataFrame()
df["Text"] = text_list
df["Airline Sentiment"] = airline_sentiment_list

#* Checking Dataset
pos_count = 0
neg_count = 0
neu_count = 0
for i in airline_sentiment_list:
    if i == "positive":
        pos_count += 1
    elif i == "negative":
        neg_count += 1
    else:
        neu_count += 1

total_count = pos_count + neg_count + neu_count

print("Total Count :", total_count)
print("Positive Count :", pos_count, "/", total_count, f"({round(pos_count / total_count * 100, 1)}%)")
print("Negative Count :", neg_count, "/", total_count, f"({round(neg_count / total_count * 100, 1)}%)")
print("Neutral Count :", neu_count, "/", total_count, f"({round(neu_count / total_count * 100, 1)}%)")
print()

#* Adding Labels to data
df["Label"] = np.nan

pos_index = df[df["Airline Sentiment"] == "positive"].index.tolist()
neg_index = df[df["Airline Sentiment"] == "negative"].index.tolist()
neu_index = df[df["Airline Sentiment"] == "neutral"].index.tolist()

df["Label"].loc[pos_index] = 1
df["Label"].loc[neg_index] = 0
df["Label"].loc[neu_index] = -1

#* Setting up ML Model

#! Definitely an easier way to change this
nlp = en_core_web_sm.load()

#* Stop Words Initialising
stopwords = list(STOP_WORDS)
# print(stopwords)
#* Punctuation Initialising
punctuations = string.punctuation
# print(punctuations)
#* Initialising Parser
parser = English()

#* Tokenizer
#! Dealing with OOV Words Using BPE in tokenization --> https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/
def my_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    return mytokens


# Vectorization
#? By Using CountVectorizer function we can convert the text to a matrix 
#? By Using CountVectorizer we produce a spare matrix, but take note that it is sometimes not suited for some ML models and should be converted to a dense matrix first
#? --> https://medium.com/@paritosh_30025/natural-language-processing-text-data-vectorization-af2520529cf7

vectorizer = CountVectorizer(tokenizer = my_tokenizer, ngram_range=(1,1)) 
tfidf_vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8)

# Classifiers
from sklearn.svm import LinearSVC


#! More About Classifiers :
#! --> https://www.youtube.com/watch?v=84gqSbLcBFE

# Splitting Data Set
from sklearn.model_selection import train_test_split

# Features and Labels
X = df['Text']
ylabels = df['Label']

#* Splits the data sets into training and test data set
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=10)

# Create the  pipeline to clean, tokenize, vectorize, and classify using"Count Vectorizor"
#? Videos on Pipeline : https://www.youtube.com/watch?v=w9IGkBfOoic

classifier_svc = LinearSVC()

model_svc = Pipeline([
                 ('vectorizer', tfidf_vectorizer),
                 ('classifier', classifier_svc)])

#! Pipeline Details

    #! The first initial steps are all the data processing one wishes to do while the last one is the classification algorithm
    #! Last Method ONLY Fit will apply while for the others Fit and Transform will apply


# Fit our data
model_svc.fit(X_train,y_train)
# Predicting with a test dataset
sample_prediction = model_svc.predict(X_test)

# Prediction Results
# for (sample, pred) in zip(X_test, sample_prediction):
#     print(sample,"Prediction :",pred)

# Accuracy
print("MLT Test Dataset Accuracy: ", str(round(model_svc.score(X_test,y_test)*100, 2))+"%" )
# print("Accuracy: ",model_svc.score(X_test,sample_prediction))
# Accuracy
print("MLT Training Dataset Accuracy: ", str(round(model_svc.score(X_train,y_train)*100, 2))+"%" )
print()


X_test_list = X_test.tolist()
X_train_list = X_train.tolist()
Y_test_list = y_test.tolist()
Y_train_list = y_train.tolist()

analyzer = SentimentIntensityAnalyzer()

#! Checking Sentiment Score
def sentiment_analyser(sentence):
    score = analyzer.polarity_scores(sentence)
    # print(f"{sentence} : {str(score)}")
    return score

X_test_result = []
X_train_result = []


for i, sentence in enumerate(X_test_list):
    score = sentiment_analyser(sentence)
    score_total = score['compound']
    if -0.1 < int(score_total) < 0.1:
        X_test_result.append(-1)
    elif score_total > 0:
        X_test_result.append(1)
    else:
        X_test_result.append(0)

for i, sentence in enumerate(X_train_list):
    score = sentiment_analyser(sentence)
    score_total = score['compound']
    if -0.05 < int(score_total) < 0.05:
        X_train_result.append(-1)
    elif score_total > 0:
        X_train_result.append(1)
    else:
        X_train_result.append(0)

counter_train = 0
counter_test = 0

for i, x in enumerate(X_test_result):
    if x == Y_test_list[i]:
        counter_test += 1

for i, x in enumerate(X_train_result):
    if x == Y_train_list[i]:
        counter_train += 1

print("Rule Based Test Dataset Accuracy :", str(counter_test) + " / " + str(len(Y_test_list)) + " (" + str(round(counter_test/len(Y_test_list)*100, 2)) + "%)" )
print("Rule Based Training Dataset Accuracy :", str(counter_train) + " / " + str(len(Y_train_list)) + " (" + str(round(counter_train/len(Y_train_list)*100, 2)) + "%)" )


#* Saving Trained Model
filename = 'finalized_model.sav'
pickle.dump(model_svc, open(filename, 'wb'))

#* Loading Back Trained Model
# loaded_model = pickle.load(open(filename, 'rb'))