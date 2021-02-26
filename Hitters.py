#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################
#############################################
#  HITTERS: END TO END APPLICATION
#############################################

# AtBat: Number of times at bat in 1986
# Hits: Number of hits in 1986
# HmRun: Number of home runs in 1986
# Runs: Number of runs in 1986
# RBI: Number of runs batted in in 1986
# Walks: Number of walks in 1986
# PutOuts: Number of put outs in 1986
# Assists: Number of assists in 1986
# Errors: Number of errors in 1986
# CAtBat: Number of times at bat during his career
# CHits: Number of hits during his career
# CHmRun: Number of home runs during his career
# CRuns: Number of runs during his career
# CRBI: Number of runs batted in during his career
# CWalks: Number of walks during his career
# Years:Number of years in the major leagues
# League: A factor with levels A and N indicating player's league at the end of 1986
# Division: A factor with levels E and W indicating player's division at the end of 1986
# NewLeague: A factor with levels A and N indicating player's league at the beginning of 1987
# Salary: 1987 annual salary on opening day in thousands of dollars

import seaborn as sns
import numpy as np
import pandas as pd
from helpers.data_prep import *
from helpers.eda import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_07\hitters.csv')
df.info
df.describe().T
check_df(df)
#Salary değişkeninde 59 adet NA değer var. Eksik değerler median ile dolduruldu.

df["Salary"] = df["Salary"].fillna(df['Salary'].median())
check_df(df)
df.dropna(inplace=True)
df.info()
df.shape #(322,20)

#######################
# 1. FEATURE ENGINEERING
#############################################
df["Hit_seg"] = pd.qcut(df['Hits'], 4, labels=['D','C','B','A'])


df['HitRatio'] = df['Hits'] / df['AtBat']

df['RunRatio'] = df['HmRun'] / df['Runs']

df['CHitRatio'] = df['CHits'] / df['CAtBat']

df['CRunRatio'] = df['CHmRun'] / df['CRuns']

# Average of numerical features
df['Avg_AtBat'] = df['CAtBat'] / df['Years']
df['Avg_Hits'] = df['CHits'] / df['Years']
df['Avg_HmRun'] = df['CHmRun'] / df['Years']
df['Avg_Runs'] = df['CRuns'] / df['Years']
df['Avg_RBI'] = df['CRBI'] / df['Years']
df['Avg_Walks'] = df['CWalks'] / df['Years']
df['Avg_PutOuts'] = df['PutOuts'] / df['Years']
df['Avg_Assists'] = df['Assists'] / df['Years']
df['Avg_Errors'] = df['Errors'] / df['Years']
df['Avg_Salary'] = df['Salary'] / df['Years']


df.loc[(df['Years'] <= 5), 'EXPERIENCE'] = 'Noob'
df.loc[(df['Years'] > 5) & (df['Years'] <= 10), 'EXPERIENCE'] = 'Incipient'
df.loc[(df['Years'] > 10) & (df['Years'] <= 15), 'EXPERIENCE'] = 'Average'
df.loc[(df['Years'] > 15) & (df['Years'] <= 20), 'EXPERIENCE'] = 'Experienced'
df.loc[(df['Years'] > 25), 'EXPERIENCE'] = 'Senior'

dff = df.copy()
dff.head()
dff = df.drop(['AtBat','Hits','HmRun','Salary','Runs','RBI'], axis = 1)

######################
# OUTLIERS
######################
num_cols = [col for col in dff.columns if len(dff[col].unique()) > 20 and dff[col].dtypes != 'O']

for col in num_cols:
    replace_with_thresholds(dff,col)

dff.shape
#(322,30)

######################
# LABEL ENCODING
######################

binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

for col in binary_cols:
    dff = label_encoder(dff, col)

########################
# ONE HOT ENCODING
########################

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols)
df.columns = [col.upper() for col in df.columns]

check_df(df)
df.describe().T
df.shape
#(322, 42)