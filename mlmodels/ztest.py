# -*- coding: utf-8 -*-
import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import scipy as sci

import sklearn as sk
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pandas_profiling

    import tensorflow as tf

    print(tf, tf.__version__)


except Exception as e:
    print(e)


print("os.getcwd", os.getcwd())
