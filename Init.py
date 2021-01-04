## 1. Import Packages

import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.model_selection import *


## 2. Files

liste_fichier = ['Nokia_6610.txt', 'Apex_AD2600_Progressive_scan_DVD_player.txt', 'Canon_G3.txt', 'Creative_Labs_Nomad_Jukebox_Zen_Xtra_40GB.txt', 'Nikon_coolpix_4300.txt']
