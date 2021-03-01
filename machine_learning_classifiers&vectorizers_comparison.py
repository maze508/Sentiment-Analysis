import pandas as pd
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
# Transformer Mixin



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
neu_count =0
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
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    return mytokens


# Basic function to clean the text 
def clean_text(text):     
    return text.strip().lower()

# Vectorization
#? By Using CountVectorizer function we can convert the text to a matrix 
#? By Using CountVectorizer we produce a spare matrix, but take note that it is sometimes not suited for some ML models and should be converted to a dense matrix first
#? --> https://medium.com/@paritosh_30025/natural-language-processing-text-data-vectorization-af2520529cf7

vectorizer = CountVectorizer(tokenizer = my_tokenizer, ngram_range=(1,1)) 
tfidf_vecotirzer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8)


# Classifiers
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

#! Simple Explanation on ML :
#! --> https://www.youtube.com/watch?v=84gqSbLcBFE

# Splitting Data Set
from sklearn.model_selection import train_test_split

# Features and Labels
X = df['Text']
ylabels = df['Label']

#* Splits the data sets into training and test data set
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

# Create the  pipeline to clean, tokenize, vectorize, and classify using"Count Vectorizor"
#! Videos on Pipeline : https://www.youtube.com/watch?v=w9IGkBfOoic
classifier_svc = LinearSVC()
classifier_sgd = SGDClassifier()
#! Need to add max_iter=greater than no. of data passed through because logisticregression defaults it to 100. Else if the error of solving is varying noticeable, the ALgorithm will fail to converge. (NOTE : When algorithm fails to convert, it doesn't always mean different accuracy)
#! https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
classifier_log = LogisticRegression(max_iter=15000)
classifier_MNB = MultinomialNB()
classifier_DTC = DecisionTreeClassifier()
classifier_RFC = RandomForestClassifier()
classifier_ada = AdaBoostClassifier()
classifier_knc = KNeighborsClassifier()

model_svc = Pipeline([
                 ('vectorizer', tfidf_vecotirzer),
                 ('classifier', classifier_svc)])
model_sgd = Pipeline([
                 ('vectorizer', tfidf_vecotirzer),
                 ('classifier', classifier_sgd)])
model_log = Pipeline([
                 ('vectorizer', tfidf_vecotirzer),
                 ('classifier', classifier_log)])
model_mnb = Pipeline([
                 ('vectorizer', tfidf_vecotirzer),
                 ('classifier', classifier_MNB)])
model_dtc = Pipeline([
                 ('vectorizer', tfidf_vecotirzer),
                 ('classifier', classifier_DTC)])
model_rfc = Pipeline([
                 ('vectorizer', tfidf_vecotirzer),
                 ('classifier', classifier_RFC)])
model_ada = Pipeline([
                 ('vectorizer', tfidf_vecotirzer),
                 ('classifier', classifier_ada)])
model_knc = Pipeline([
                 ('vectorizer', tfidf_vecotirzer),
                 ('classifier', classifier_knc)])

print()
print()
pipelines = [model_svc, model_sgd, model_log, model_mnb, model_dtc, model_rfc, model_ada, model_knc]
pipeline_names = ["LinearSVC", "SGDClassifier", "LogisticRegression", "MultinomialNB", "DecisionTreeClassifier", "RandomForestClassifier", "AdaBoostClassifier", "KNeighborsClassifier"]

for i, pipe in enumerate(pipelines):
    pipe.fit(X_train,y_train)
    print(f"{pipeline_names[i]} :")
    print()
    print("Test Dataset Accuracy: ",pipe.score(X_test,y_test))
    print("Training Dataset Accuracy: ",pipe.score(X_train,y_train))
    print("SUM of Accuracies :", round(pipe.score(X_test,y_test) + pipe.score(X_train,y_train), 4))
    print('\n')