from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import sys
import statsmodels as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX



siteHeader = st.beta_container()
dataExploration = st.beta_container()
newFeatures = st.beta_container()
modelTraining = st.beta_container()


with siteHeader:
    st.title('Technology Sales Forecasting Project')
    st.text('In this project I look into ... And I try ... I worked with the dataset from ...')

with dataExploration:
    st.header('SALES FORECASTING DATASET')
    st.text('I found this dataset at...  I decided to work with it because ...')
    sales_data = pd.read_csv('Superstore.csv')
    # Category = sales_data['Category'].drop_duplicates()
    # select_category = st.sidebar.selectbox('Select Category :',Category)
    technology = sales_data.loc[sales_data['Category'] == 'Technology']
    st.dataframe(technology)
    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name']
    technology.drop(cols, axis=1, inplace=True)
    technology = technology.sort_values('Order Date')
    technology.isnull().sum()
    technology = technology.groupby('Order Date')['Sales'].sum().reset_index()
    technology['Order Date'] = pd.to_datetime(technology['Order Date'])
    technology.set_index('Order Date', inplace=True)
    st.dataframe(technology)
    y = technology['Sales'].resample('MS').mean()
    y.head()
    st.line_chart(technology)
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')

with newFeatures:
    st.header('New features I came up with')
    st.text('Let\'s take a look into the features I generated.')
    
with modelTraining:
    st.header('Model training')
    st.text('In this section you can select the hyperparameters!')

# st.title('Improved Home Electricity Forecasting')
# st.subheader('Project Objective:')
# st.text('The aim of the project is to carry out a short term forecast on electricity consumption of\na single home using ARIMA model')
# st.subheader('Dataset Information:')
# st.text('Dataset is take from UCI Machine Learning repositiory.\nThis archive contains 2075259 measurements gathered in\na house located in Sceaux(7km of Paris, France) between\nDecember 2006 and November 2010 (47 months)\n')