#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

#############################################
#  TITANIC : END TO END APPLICATION
#############################################

# PassengerId is the unique id of the row and it doesn't have any effect on target
# Survived is the target variable we are trying to predict (0 or 1):
# 1 = Survived
# 0 = Not Survived

# Pclass (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has 3 unique values (1, 2 or 3):
# 1 = Upper Class
# 2 = Middle Class
# 3 = Lower Class

# Name, Sex and Age are self-explanatory
# SibSp is the total number of the passengers' siblings and spouse
# Parch is the total number of the passengers' parents and children
# Ticket is the ticket number of the passenger
# Fare is the passenger fare
# Cabin is the cabin number of the passenger

# Embarked is port of embarkation and it is a categorical feature which has 3 unique values (C, Q or S):
# C = Cherbourg
# Q = Queenstown
# S = Southampton


import numpy as np
import pandas as pd
from helpers.data_prep import *
from helpers.eda import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_07\titanic.csv')
df.info
check_df(df)

#############################################
# FEATURE ENGINEERING
#############################################

df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype('int')
df["NEW_NAME_COUNT"] = df["Name"].str.len()
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df["NEW_TITLE"] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), "NEW_AGE_CAT"] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

df.columns = [x.upper() for x in df.columns]
df

#############################################
# 1.OUTLIERS
#############################################
num_cols = [col for col in df.columns if len(df[col].unique()) > 20
            and df[col].dtypes != 'O'
            and col not in "PASSENGERID"]

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

check_df(df)



#############################################
# 2. MISSING VALUES
#############################################

missing_values_table(df)

df.drop(["CABIN", "TICKET", "NAME"], inplace=True, axis=1)
missing_values_table(df)
df

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
missing_values_table(df)

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

missing_values_table(df)

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

missing_values_table(df)

#############################################
# 3. LABEL ENCODING
#############################################
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

for col in binary_cols:
    df = label_encoder(df, col)

df.head()


#############################################
# 4. RARE ENCODING
#############################################

rare_analyser(df, "SURVIVED", 0.05)
df = rare_encoder(df, 0.01)
rare_analyser(df, "SURVIVED", 0.01)

#############################################
# 5. ONE-HOT ENCODING
#############################################

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()

#############################################
# 6. STANDART SCALER
#############################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df[["AGE"]])
df["AGE"] = scaler.transform(df[["AGE"]])
check_df(df)


#############################################
# 7. MODEL
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier().fit(X, y)
y_pred = rf_model.predict(X)
accuracy_score(y_pred, y)

from matplotlib import pyplot as plt

def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('importance-01.png')
    plt.show()

plot_importance(rf_model, X)



