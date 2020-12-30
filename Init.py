## 1. Import Packages

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.feature_extraction.text import *
from sklearn.metrics import *

## 2. Path

'''
I didn't find a way to create a path to the github repository so I use my data stored localy
'''

path = 'C:/Users/nicol/OneDrive/fac/M2/S1/Maths for M&D Learning 2/Projet/'
