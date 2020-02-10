


%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

import nltk
from nltk.corpus import stopwords

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB




df['Product_id'] = df['Product'].factorize()[0]   #label encoder !
# df['Product_id']



pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])


for category in categories:
    print('... Processing {}'.format(category))
    SVC_pipeline.fit(X_train, train[category])
    pred = SVC_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], pred)))




# Stemming and Lemmatizing
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        final_doc.append(porter.stem(word))
        #final_doc.append(snowball.stem(word))
        #final_doc.append(wordnet.lemmatize(word))

    preprocessed_docs.append(final_doc)

print(preprocessed_docs)






# Cleaning text of stopwords
from nltk.corpus import stopwords

tokenized_docs_no_stopwords = []

for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)

    tokenized_docs_no_stopwords.append(new_term_vector)

print(tokenized_docs_no_stopwords)







# Stemming and Lemmatizing
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        final_doc.append(porter.stem(word))
        #final_doc.append(snowball.stem(word))
        #final_doc.append(wordnet.lemmatize(word))

    preprocessed_docs.append(final_doc)

print(preprocessed_docs)





X = df[ 'Issue' ]
y = df[ 'Product' ]

models = [
    LinearSVC(),
    LogisticRegression(random_state=0)
]

CV = 3
def  model_Kfold(model_list, CV=3, Xtrain, ytrain, scoring="accuracy"  ) :
  cv_df = pd.DataFrame(index=range(CV * len(models)))
  entries = []
  for model in model_list:
    model_name = model.__class__.__name__
    accuracies = cross_val_score( model,
                                  Xtrain, ytrain, scoring=scoring , cv=CV )

  for fold_idx, accuracy in enumerate(accuracies):
    a = (model_name, fold_idx, accuracy)
    entries.append(a)
    print( a )

  cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
  return cv_df





#####################################################################################################
######### Term Frequency   ##########################################################################
If you need the term frequency (term count) vectors for different tasks, use Tfidftransformer.
If you need to compute tf-idf scores on documents within your “training” dataset, use Tfidfvectorizer
If you need to compute tf-idf scores on documents outside your “training” dataset, use either one, both will work.













#####################################################################################################
### The sklearn.feature_extraction.text submodule gathers utilities to build feature vectors from text documents.

feature_extraction.text.CountVectorizer([ÿ])	Convert a collection of text documents to a matrix of token counts
feature_extraction.text.HashingVectorizer([ÿ])	Convert a collection of text documents to a matrix of token occurrences
feature_extraction.text.TfidfVectorizer([ÿ])	Convert a collection of raw documents to a matrix of TF-IDF features.



np.concatenate((a, b.T), axis=1)














