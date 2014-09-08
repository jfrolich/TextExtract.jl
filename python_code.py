from contextlib import contextmanager
from time import time
from sys import stdout
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import pandas as pd
@contextmanager

# A Timer function
def duration(outfile=stdout):
    start = time()
    yield
    end = time()
    outfile.write(str(end - start))

PATH = "/Users/jfrolich/Documents/Data sets/Reuters/fetch/export"
metadata = pd.read_table(os.path.join(PATH, 'metadata.csv'), sep=',')
y = np.array(metadata.acq)

stdout.write('Python,x,')

with duration():
    content = [open(os.path.join(PATH, 'text', str(f))).read() for f in metadata.id]

stdout.write(',')

with duration():
    vectorizer = CountVectorizer(decode_error = 'ignore', ngram_range = (1,3))
    X = vectorizer.fit_transform(content)

stdout.write(',')
with duration():
  select = SelectKBest(chi2, k=1000)
  X = select.fit_transform(X, y)

stdout.write('\n')
# print X.shape
# print X.nnz
# y = metadata.acq
#
#
# Xw = TfidfTransformer().fit_transform(X)
# y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33, random_state = 999)
# select = SelectKBest(chi2, k=5000)
# Xs_train = select.fit_transform(X_train, y_train).toarray()
# Xs_test  = select.transform(X_test).toarray()

#with duration():
#    clf = RandomForestClassifier(250)
#    clf.fit(Xs_train, y_train)
