import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'martial-status', 'occupation', 'relationship', 'race',
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'label']

train_df = pd.read_csv('adult.data',names = cols, header = None ,index_col = False)
test_df = pd.read_csv('adult.test',names = cols, header = None ,index_col = False, skiprows = 1)

train_df['education-num'].unique()

# Data cleaning:
# education num is a label encoding, we want to do one-hot so we drop education-num
train_df = train_df.drop('education-num', axis='columns')
test_df = test_df.drop('education-num', axis='columns')

# drop all rows with unknown values
def clear_unknown_rows(df):
    df = df.replace(' ?',np.NaN)
    # existed in test but not training set, can do this in a better way but skipping for now
    df = df.replace(' Holand-Netherlands', np.NaN)
    df = df.dropna()
    return df

train_df = clear_unknown_rows(train_df)
test_df = clear_unknown_rows(test_df)

# one hot encoding
def encode_categories(df):
    # Workclass is 
    df = df.join(pd.get_dummies(df.pop('workclass')))
    
    df = df.join(pd.get_dummies(df.pop('education')))
    df = df.join(pd.get_dummies(df.pop('martial-status')))
    df = df.join(pd.get_dummies(df.pop('occupation')))
    df = df.join(pd.get_dummies(df.pop('relationship')))
    df = df.join(pd.get_dummies(df.pop('race')))
    df = df.join(pd.get_dummies(df.pop('sex')))
    df = df.join(pd.get_dummies(df.pop('native-country')))
    # label needs to be one column instead of two
    # need 4 because there's one with a period and one without
    df.replace({' <=50K.': 0, ' >50K.' : 1, ' <=50K': 0, ' >50K' : 1}, inplace=True)
    return df

train_df = encode_categories(train_df)
test_df = encode_categories(test_df)

train_df.head()

plt.figure(figsize = (50,50))
pd.plotting.scatter_matrix(train_df.head(), diagonal = 'hist')
plt.show()
