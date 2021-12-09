import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def find(news):
    print(news)
    classifier=MultinomialNB()
    df=pd.read_csv('detection_app/mlmodel/fake-news/train.csv')

    ## Get the Independent Features
    X=df.drop('label',axis=1)
    ## Get the Dependent features
    y=df['label']
    df=df.dropna()
    messages=df.copy()
    messages.reset_index(inplace=True)
    print("pass 1")
    ps = PorterStemmer()
    corpus = []

    print("pass 2")
    test_predict=news
    # X = X.drop(X.index[len(X)-1])
    # messages = messages.drop(messages.index[len(messages)-1])
    for i in range(0, len(messages)):
        print(i)
        review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    test_predict = re.sub('[^a-zA-Z]', ' ', test_predict)
    test_predict = test_predict.lower()
    test_predict = test_predict.split()

    test_predict = [ps.stem(word) for word in test_predict if not word in stopwords.words('english')]
    test_predict = ' '.join(test_predict)
    corpus.pop()
    corpus.append(test_predict)


    cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
    # print(corpus)
    X = cv.fit_transform(corpus).toarray()

    y=messages['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle=False)
    count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print(score)
    print(X_test)

    print(pred[len(pred)-1])
    return pred[len(pred)-1].tolist()
