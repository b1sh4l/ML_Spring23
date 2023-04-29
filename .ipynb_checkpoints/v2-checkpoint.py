# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:59:57 2023

@author: User
"""

!pip install xgboost
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head().T

df.drop(columns=['id'], inplace=True)
df.info()

def piedist(data, column, labels):
    """
    Plots the distribution percentage of a categorical column
    in a pie chart.
    """
    dist = data[column].value_counts()
    colors = ['#66b3ff', '#99ff99', '#ff9999', '#ffcc99', '#be99ff']
    plt.pie(x=dist, labels=labels, autopct='%1.2f%%', pctdistance=0.5, colors=colors)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
sns.kdeplot(data=df, x='bmi', hue='stroke')
plt.title('Stroke vs BMI')
plt.legend(['Stroke', 'No stroke'])

plt.title('Gender Distribution')
sns.countplot(x=df['gender'])

# Number of people with gender of "Other"
(df['gender'] == 'Other').sum()

# There's only one person with a gender of other, we can drop it
df.drop(df[df['gender'] == 'Other'].index, axis=0, inplace=True)

fig = plt.figure(figsize=(8, 5))

ax = plt.subplot2grid((1, 2), (0, 0))
plt.title('Stroke vs Gender')
piedist(data=df[df['stroke'] == 1], column='gender', labels=['Female', 'Male'])

ax = plt.subplot2grid((1, 2), (0, 1))
plt.title('No stroke vs Gender')
piedist(data=df[df['stroke'] == 0], column='gender', labels=['Female', 'Male'])

sns.boxplot(x=df['age'])

sns.kdeplot(data=df, x='age', hue='stroke')
plt.title('Stroke vs Age')
plt.legend(['Stroke', 'No stroke'])