from sklearn import set_config
import pickle
set_config(display='diagram')
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
# import streamlit.components.v1 as components


import warnings
warnings.filterwarnings('ignore')

st.title("Car Price and Feature Estimator")

model=pickle.load('model.pkl','rb')
vectorizer=pickle.load('vectorizer.pkl','rb')

