import nltk
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# import data
data = pd.read_csv('C:/Users/paris/PycharmProjects/Python/MGMT590AUD/Project/output.csv',
                   usecols = ['Title', 'Description', 'Category'])


# select useful information
data[['Category', 'Owner']] = data['Category'].str.split('-',expand=True)
df=data[['Category', 'Title', 'Description']]
#df['des'] = df['Title'] + ' ' + df['Description']
#df.des = df.des.astype(str)
df.Category = df.Category.astype(str)
df.Title = df.Title.astype(str)
#df.Description = df.Description.astype(str)


# select 500 records from each category to shrink data size
df_small=df.groupby('Category', group_keys=False).apply(lambda x: x.sample(min(len(x), 500))).reset_index()


# process text
lemmatizer = nltk.stem.WordNetLemmatizer()
data_processed = []
title=df_small.Title
for i in range(df_small.shape[0]):
    token_1 = nltk.word_tokenize(title[i].lower())  # tokenize
    lemmatize_2 = [lemmatizer.lemmatize(token) for token in token_1]  # lemmatize
    stop_words_removed = [token for token in lemmatize_2
                          if not token in stopwords.words('english')
                          if token.isalpha()]  # remove stop-words and punctuations
    data_processed.append(' '.join(stop_words_removed))  # store into list

cat = df_small.Category.tolist()

df_processed = pd.DataFrame({'category': cat, 'title': data_processed})


# takeout genertal category as holdout set
Train = df_processed[~df_processed['category'].isin(['general for sale '])]
Holdout = df_processed[df_processed['category'] == 'general for sale ']
Holdout_x = Holdout.title


# split tr and te
train_x, test_x, train_y, test_y = train_test_split(Train.title, Train.category, test_size=0.3, random_state=3)


# TFIDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

def tk(doc):
    return doc

vec = TfidfVectorizer(analyzer='word', tokenizer=tk, preprocessor=tk, token_pattern=None,
                      min_df=5, ngram_range=(1,5), stop_words='english')
vec.fit(train_x)
train_x = vec.transform(train_x)
test_x = vec.transform(test_x)
holdout_x = vec.transform(Holdout_x)


# train Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
NBmodel.fit(train_x, train_y)
y_pred_NB = NBmodel.predict(test_x)

# train Logit
from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()
Logitmodel.fit(train_x, train_y)
y_pred_logit = Logitmodel.predict(test_x)

# train Random Forest
from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier(n_estimators=100, max_depth=100, bootstrap=True, random_state=0)  # with 10 trees
RFmodel.fit(train_x, train_y)
y_pred_RF = RFmodel.predict(test_x)

# train SVM
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()
SVMmodel.fit(train_x, train_y)
y_pred_SVM = SVMmodel.predict(test_x)

# train polynomial SVM
from sklearn.svm import SVC
SVMPmodel = SVC(kernel = 'poly',degree=3)
SVMPmodel.fit(train_x, train_y)
y_pred_SVMP = SVMPmodel.predict(test_x)



## performance
from sklearn.metrics import accuracy_score
acc_NB = accuracy_score(test_y, y_pred_NB)  # evaluate accuracy rate of Naive Bayes model
print("Naive Bayes model Accuracy::{:.2f}%".format(acc_NB*100))

acc_logit = accuracy_score(test_y, y_pred_logit)  # evaluate accuracy rate of Logit model model
print("Logit model Accuracy:: {:.2f}%".format(acc_logit*100))

acc_RF = accuracy_score(test_y, y_pred_RF)  # evaluate accuracy rate of Random Forest model
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))

acc_SVM = accuracy_score(test_y, y_pred_SVM)  # evaluate accuracy rate of SVM model
print("SVM model Accuracy::{:.2f}%".format(acc_SVM*100))

acc_SVMP = accuracy_score(test_y, y_pred_SVMP)  # evaluate accuracy rate of SVM poly model
print("SVMP model Accuracy::{:.2f}%".format(acc_SVM*100))

# Neural Network
from sklearn.neural_network import MLPClassifier
NNmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,), random_state=1)
NNmodel.fit(train_x, train_y)
y_pred_NN= NNmodel.predict(test_x)
acc_NN = accuracy_score(test_y, y_pred_NN)
print("NN model Accuracy: {:.2f}%".format(acc_NN*100))

# Deep learning
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 10, 10), random_state=1)
DLmodel.fit(train_x, train_y)
y_pred_DL= DLmodel.predict(test_x)
acc_DL = accuracy_score(test_y, y_pred_DL)
print("DL model Accuracy: {:.2f}%".format(acc_DL*100))


# predict holdout set
holdout_y_pred_SVM = SVMmodel.predict(holdout_x)

pre_cat = holdout_y_pred_SVM.tolist()
holdout_title = Holdout.title.tolist()

df_holdout_pred = pd.DataFrame({'Pred_Cat': pre_cat, 'Title': holdout_title})

df_holdout_pred.to_csv('C:/Users/paris/PycharmProjects/Python/MGMT590AUD/Project/df_holdout_pred.csv',
                   index=True, header=True)