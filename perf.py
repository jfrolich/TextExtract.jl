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

# A timer function
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
    feature_names = vectorizer.get_feature_names()

stdout.write(',')
with duration():
  select = SelectKBest(chi2, k=1000)
  X = select.fit_transform(X, y)

stdout.write('\n')
