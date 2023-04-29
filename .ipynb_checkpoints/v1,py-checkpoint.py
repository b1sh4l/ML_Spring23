# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:54:29 2023

@author: User
"""

import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
import seaborn as sns
from imblearn.over_sampling import SMOTE
import scikitplot as skplt


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score


from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# Show Missing Data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print('======\tMissing Data\t======')
df.isnull().sum()

# Using Decision Tree to fill up missing Data of BMI

DT_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))
                              ])
X = df[['age','gender','bmi']].copy()
X.gender = X.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8) # Classifying Gender

Missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
Y = X.pop('bmi')
DT_bmi_pipe.fit(X,Y)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]),index=Missing.index)
df.loc[Missing.index,'bmi'] = predicted_bmi
print('Present Missing Values:\t',sum(df.isnull().sum()))

variables = [variable for variable in df.columns if variable not in ['id','stroke']]
conts = ['age','avg_glucose_level','bmi']

# NUMERIC VARIABLE DISTRIBUTION

fig = plt.figure(figsize=(15, 12), dpi=300, facecolor='#c1dce6')
gs = fig.add_gridspec(5, 5)
gs.update(wspace=0.1, hspace=0.5)

background_color = "#c1dce6"

plot = 0
for row in range(0, 1):
    for col in range(0, 3):
        locals()["ax"+str(plot)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(plot)].set_facecolor(background_color)
        locals()["ax"+str(plot)].tick_params(axis='y', left=False)
        locals()["ax"+str(plot)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(plot)].spines[s].set_visible(False)
        plot += 1

plot = 0
for variable in conts:
        sns.kdeplot(df[variable] ,ax=locals()["ax"+str(plot)], color='#2e0742', shade=True, linewidth=1.5, ec='#ffbf00',alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(plot)].grid(which='major', axis='x', zorder=0, color='red', linestyle=':', dashes=(1,5))
        plot += 1
        
ax0.set_xlabel('Age')
ax1.set_xlabel('Avg. Glucose Levels')
ax2.set_xlabel('BMI')

ax0.text(-20, 0.022, 'Numeric Variable Distribution', fontsize=24, fontweight='bold', fontfamily='Quicksand')
ax0.text(-20, 0.019, 'We see a positive skew in BMI and Glucose Level', fontsize=12, fontweight='light', fontfamily='Quicksand')

plt.show()


fig = plt.figure(figsize=(10, 12), dpi=300,facecolor=background_color)
gs = fig.add_gridspec(4, 3)
gs.update(wspace=0.1, hspace=0.5)

# NUMERIC VARIABLES BY STROKE AND NO-STROKE

plot = 0
for row in range(0, 1):
    for col in range(0, 3):
        locals()["ax"+str(plot)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(plot)].set_facecolor(background_color)
        locals()["ax"+str(plot)].tick_params(axis='y', left=False)
        locals()["ax"+str(plot)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(plot)].spines[s].set_visible(False)
        plot += 1

plot = 0

s = df[df['stroke'] == 1]
ns = df[df['stroke'] == 0]

for feature in conts:
        sns.kdeplot(s[feature], ax=locals()["ax"+str(plot)], color='#c5bfff', shade=True, linewidth=1.5, ec='#ffbf00',alpha=0.9, zorder=3, legend=False)
        sns.kdeplot(ns[feature],ax=locals()["ax"+str(plot)], color='#2e0742', shade=True, linewidth=1.5, ec='#ffbf00',alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(plot)].grid(which='major', axis='x', zorder=0, color='red', linestyle=':', dashes=(1,5))
        locals()["ax"+str(plot)].set_xlabel(feature)
        plot += 1

ax0.set_xlabel('Age')
ax1.set_xlabel('Avg. Glucose Levels')
ax2.set_xlabel('BMI')
        
ax0.text(-20, 0.056, 'Numeric Variables by Stroke & No Stroke', fontsize=24, fontweight='bold', fontfamily='Quicksand')
ax0.text(-20, 0.05, 'Age looks to be a prominent factor - this will likely be a salient feautre in our models', 
         fontsize=12, fontweight='light', fontfamily='Quicksand')

plt.show()



str_only = df[df['stroke'] == 1]
no_str_only = df[df['stroke'] == 0]

# Setting up figure and axes

fig = plt.figure(figsize=(10,16),dpi=300,facecolor=background_color) 
gs = fig.add_gridspec(6, 2)
gs.update(wspace=0.5, hspace=0.2)
ax0 = fig.add_subplot(gs[0, 0:2])
ax1 = fig.add_subplot(gs[1, 0:2]) 

ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)

# GLUCOSE LEVEL by AGE
sns.regplot(no_str_only['age'],y=no_str_only['avg_glucose_level'],  
            color='#c5bfff', scatter_kws={'edgecolors':['#2e0742'],
                                          'linewidth': 0.2},
            logx=True,
            ax=ax0)

sns.regplot(str_only['age'],y=str_only['avg_glucose_level'],  
            color='#2e0742',
            logx=True,scatter_kws={'edgecolors':['#ffbf00'], 
                                   'linewidth': 0.2},
            ax=ax0)

ax0.set(ylim=(0, None))
ax0.set_xlabel(" ",fontsize=12,fontfamily='Quicksand')
ax0.set_ylabel("Avg. Glucose Level",fontsize=12,fontfamily='Quicksand',loc='bottom')

ax0.tick_params(axis='x', bottom=False)
ax0.get_xaxis().set_visible(False)

for s in ['top','left','bottom']:
    ax0.spines[s].set_visible(False)

# BMI by AGE
sns.regplot(no_str_only['age'],y=no_str_only['bmi'],  
            color='#c5bfff', scatter_kws={'edgecolors':['#2e0742'], 
                                              'linewidth': 0.2},
            logx=True,
            ax=ax1)

sns.regplot(str_only['age'],y=str_only['bmi'],  
            color='#2e0742', scatter_kws={'edgecolors':['#ffbf00'], 
                                              'linewidth': 0.2},
            logx=True,
            ax=ax1)

ax1.set_xlabel("Age",fontsize=12,fontfamily='Quicksand',loc='left')
ax1.set_ylabel("BMI",fontsize=12,fontfamily='Quicksand',loc='bottom')

for s in ['top','left','right']:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)

ax0.text(-5,350,'Strokes by Age, Glucose Level, and BMI',fontsize=22,fontfamily='Quicksand',fontweight='bold')
ax0.text(-5,320,'Age appears to be a very important factor',fontsize=12,fontfamily='Quicksand')

ax0.tick_params(axis=u'both', which=u'both',length=0)
ax1.tick_params(axis=u'both', which=u'both',length=0)

plt.show()



fig = plt.figure(figsize=(8, 5), dpi=300,facecolor=background_color)
gs = fig.add_gridspec(2,1)
gs.update(wspace=0.15, hspace=0.5)
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor(background_color)

df['age'] = df['age'].astype(int)

rate = []
for i in range(df['age'].min(), df['age'].max()):
    rate.append(df[df['age'] < i]['stroke'].sum() / len(df[df['age'] < i]['stroke']))

sns.lineplot(data=rate,color='#2e0742',ax=ax0)

for s in ["top","right","left"]:
    ax0.spines[s].set_visible(False)

ax0.tick_params(axis='both', which='major', labelsize=8,)
ax0.tick_params(axis=u'both', which=u'both',length=0)

ax0.text(-3,0.055,'Risk Increase by Age',fontsize=18,fontfamily='Quicksand',fontweight='bold')
ax0.text(-3,0.047,'As age increase, so too does risk of having a stroke',fontsize=12,fontfamily='Quicksand')

plt.show()



from pywaffle import Waffle

fig = plt.figure(figsize=(3, 2),dpi=300,facecolor=background_color,
    FigureClass=Waffle,
    rows=2,
    values=[1, 19],
    colors=['#2e0742', "#c5bfff"],
    characters='⬤',
    font_size=18, vertical=True,
)

fig.text(0.035,0.78,'People Affected by a Stroke in our dataset',fontfamily='Quicksand',fontsize=12,fontweight='bold')
fig.text(0.035,0.70,'This is around 1 in 20 people [249 out of 5000]',fontfamily='Quicksand',fontsize=6)

plt.show()




# Drop single 'Other' gender
no_str_only = no_str_only[(no_str_only['gender'] != 'Other')]

import matplotlib.patheffects as pe

fig = plt.figure(figsize=(28,22))
gs = fig.add_gridspec(3, 3)
gs.update(wspace=0.35, hspace=0.27)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])
ax6 = fig.add_subplot(gs[2, 0])
ax7 = fig.add_subplot(gs[2, 1])
ax8 = fig.add_subplot(gs[2, 2])

background_color = "#c1dce6"
fig.patch.set_facecolor(background_color) # figure background color
# Plot

## Age
ax0.grid(color='#2e0742', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
positive = pd.DataFrame(str_only["age"])
negative = pd.DataFrame(no_str_only["age"])
sns.kdeplot(positive["age"], ax=ax0,color="#ffbf00", shade=True, alpha=0.9, ec='#2e0742', label="positive")
sns.kdeplot(negative["age"], ax=ax0,color="#c5bfff", shade=True, alpha=0.9, ec='#2e0742', label="negative")

ax0.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax0.set_ylabel('')    
ax0.set_xlabel('')
ax0.text(-20, 0.0465, 'Age', fontsize=28, fontweight='bold', fontfamily='Quicksand', color="#2e0742")

# Smoking
positive = pd.DataFrame(str_only["smoking_status"].value_counts())
positive["Percentage"] = positive["smoking_status"].apply(lambda x: x/sum(positive["smoking_status"])*100)
negative = pd.DataFrame(no_str_only["smoking_status"].value_counts())
negative["Percentage"] = negative["smoking_status"].apply(lambda x: x/sum(negative["smoking_status"])*100)

ax1.text(0, 4, 'Smoking Status', fontsize=28, fontweight='bold', fontfamily='Quicksand', color="#2e0742")
ax1.barh(positive.index, positive['Percentage'], color="#ffbf00", zorder=3, ec='#2e0742', height=0.7)
ax1.barh(negative.index, negative['Percentage'], color="#c5bfff", zorder=3, ec='#2e0742', height=0.3)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
ax1.xaxis.set_major_locator(mtick.MultipleLocator(10))

# Ax2 - GENDER 
positive = pd.DataFrame(str_only["gender"].value_counts())
positive["Percentage"] = positive["gender"].apply(lambda x: x/sum(positive["gender"])*100)
negative = pd.DataFrame(no_str_only["gender"].value_counts())
negative["Percentage"] = negative["gender"].apply(lambda x: x/sum(negative["gender"])*100)

x = np.arange(len(positive))
ax2.text(-0.4, 68.5, 'Gender', fontsize=28, fontweight='bold', fontfamily='Quicksand', color="#2e0742")
ax2.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
ax2.bar(x, height=positive["Percentage"], zorder=3, color="#ffbf00", ec='#2e0742', width=0.4)
ax2.bar(x+0.4, height=negative["Percentage"], zorder=3, color="#c5bfff", ec='#2e0742', width=0.4)
ax2.set_xticks(x + 0.4 / 2)
ax2.set_xticklabels(['Male','Female'])
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.yaxis.set_major_locator(mtick.MultipleLocator(10))
for i,j in zip([0, 1], positive["Percentage"]):
    ax2.annotate(f'{j:0.0f}%',xy=(i, j/2), color='#000000', fontweight='bold', horizontalalignment='center', verticalalignment='center')
for i,j in zip([0, 1], negative["Percentage"]):
    ax2.annotate(f'{j:0.0f}%',xy=(i+0.4, j/2), color='#000000', fontweight='bold', horizontalalignment='center', verticalalignment='center')

# Heart Disease
positive = pd.DataFrame(str_only["heart_disease"].value_counts())
positive["Percentage"] = positive["heart_disease"].apply(lambda x: x/sum(positive["heart_disease"])*100)
negative = pd.DataFrame(no_str_only["heart_disease"].value_counts())
negative["Percentage"] = negative["heart_disease"].apply(lambda x: x/sum(negative["heart_disease"])*100)

x = np.arange(len(positive))
ax3.text(-0.3, 110, 'Heart Disease', fontsize=28, fontweight='bold', fontfamily='Quicksand', color="#2e0742")
ax3.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
ax3.bar(x, height=positive["Percentage"], zorder=3, color="#ffbf00", ec='#2e0742', width=0.4)
ax3.bar(x+0.4, height=negative["Percentage"], zorder=3, color="#c5bfff", ec='#2e0742', width=0.4)
ax3.set_xticks(x + 0.4 / 2)
ax3.set_xticklabels(['No History','History'])
ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
ax3.yaxis.set_major_locator(mtick.MultipleLocator(20))
for i,j in zip([0, 1], positive["Percentage"]):
    ax3.annotate(f'{j:0.0f}%',xy=(i, j/2), color='#000000', fontweight='bold', horizontalalignment='center', verticalalignment='center')
for i,j in zip([0, 1], negative["Percentage"]):
    ax3.annotate(f'{j:0.0f}%',xy=(i+0.4, j/2), color='#000000', fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
## AX4 - TITLE
ax4.spines["bottom"].set_visible(False)
ax4.tick_params(left=False, bottom=False)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.text(0.5, 0.6, 'Can we see patterns for\n\n\n patients in our data?', horizontalalignment='center', verticalalignment='center',
         fontsize=46, fontweight='bold', fontfamily='Quicksand', color="#2e0742")

ax4.text(0.0,0.55,"Stroke", fontweight="bold", fontfamily='Quicksand', fontsize=52, color='#ffbf00', path_effects=[pe.withStroke(linewidth=2, foreground="#c94000")])
ax4.text(0.40,0.55,"&", fontweight="bold", fontfamily='Quicksand', fontsize=46, color='#2e0742')
ax4.text(0.50,0.55,"No-Stroke", fontweight="bold", fontfamily='Quicksand', fontsize=52, color='#c5bfff', path_effects=[pe.withStroke(linewidth=2, foreground="#2e0742")])

# Glucose
ax5.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
positive = pd.DataFrame(str_only["avg_glucose_level"])
negative = pd.DataFrame(no_str_only["avg_glucose_level"])
sns.kdeplot(positive["avg_glucose_level"], ax=ax5,color="#ffbf00",ec='#2e0742', shade=True, alpha=0.9,  label="positive")
sns.kdeplot(negative["avg_glucose_level"], ax=ax5, color="#c5bfff", ec='#2e0742',shade=True, alpha=0.9, label="negative")
ax5.text(-55, 0.01855, 'Avg. Glucose Level', 
         fontsize=28, fontweight='bold', fontfamily='Quicksand', color="#2e0742")
ax5.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax5.set_ylabel('')    
ax5.set_xlabel('')

## BMI
ax6.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
positive = pd.DataFrame(str_only["bmi"])
negative = pd.DataFrame(no_str_only["bmi"])
sns.kdeplot(positive["bmi"], ax=ax6,color="#ffbf00", ec='#2e0742',shade=True, alpha=0.9,  label="positive")
sns.kdeplot(negative["bmi"], ax=ax6, color="#c5bfff",ec='#2e0742', shade=True, alpha=0.9, label="negative")
ax6.text(-0.06, 0.09, 'BMI', 
         fontsize=28, fontweight='bold', fontfamily='Quicksand', color="#2e0742")
ax6.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax6.set_ylabel('')    
ax6.set_xlabel('')

# Work Type
positive = pd.DataFrame(str_only["work_type"].value_counts())
positive["Percentage"] = positive["work_type"].apply(lambda x: x/sum(positive["work_type"])*100)
positive = positive.sort_index()

negative = pd.DataFrame(no_str_only["work_type"].value_counts())
negative["Percentage"] = negative["work_type"].apply(lambda x: x/sum(negative["work_type"])*100)
negative = negative.sort_index()

ax7.bar(negative.index, height=negative["Percentage"], zorder=3, color="#c5bfff", ec='#2e0742', width=0.05)
ax7.scatter(negative.index, negative["Percentage"], zorder=3, s=100, color="#c5bfff", ec='#2e0742')
ax7.bar(np.arange(len(positive.index))+0.4, height=positive["Percentage"], zorder=3, color="#ffbf00", ec='#2e0742', width=0.05)
ax7.scatter(np.arange(len(positive.index))+0.4, positive["Percentage"], zorder=3, s=100, color="#ffbf00", ec='#2e0742')

ax7.yaxis.set_major_formatter(mtick.PercentFormatter())
ax7.yaxis.set_major_locator(mtick.MultipleLocator(10))
ax7.set_xticks(np.arange(len(positive.index))+0.4 / 2)
ax7.set_xticklabels(list(positive.index),rotation=0)
ax7.text(-0.5, 66, 'Work Type', fontsize=28, fontweight='bold', fontfamily='Quicksand', color="#2e0742")

# hypertension
positive = pd.DataFrame(str_only["hypertension"].value_counts())
positive["Percentage"] = positive["hypertension"].apply(lambda x: x/sum(positive["hypertension"])*100)
negative = pd.DataFrame(no_str_only["hypertension"].value_counts())
negative["Percentage"] = negative["hypertension"].apply(lambda x: x/sum(negative["hypertension"])*100)

x = np.arange(len(positive))
ax8.text(-0.45, 100, 'Hypertension', fontsize=28, fontweight='bold', fontfamily='Quicksand', color="#2e0742")
ax8.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
ax8.bar(x, height=positive["Percentage"], zorder=3, color="#ffbf00", ec='#2e0742', width=0.4)
ax8.bar(x+0.4, height=negative["Percentage"], zorder=3, color="#c5bfff", ec='#2e0742', width=0.4)
ax8.set_xticks(x + 0.4 / 2)
ax8.set_xticklabels(['No History','History'])
ax8.yaxis.set_major_formatter(mtick.PercentFormatter())
ax8.yaxis.set_major_locator(mtick.MultipleLocator(20))
for i,j in zip([0, 1], positive["Percentage"]):
    ax8.annotate(f'{j:0.0f}%',xy=(i, j/2), color='#000000', fontweight='bold', horizontalalignment='center', verticalalignment='center')
for i,j in zip([0, 1], negative["Percentage"]):
    ax8.annotate(f'{j:0.0f}%',xy=(i+0.4, j/2), color='#000000', fontweight='bold', horizontalalignment='center', verticalalignment='center')

# tidy up
for s in ["top","right","left"]:
    for i in range(0,9):
        locals()["ax"+str(i)].spines[s].set_visible(False)
        
for i in range(0,9):
        locals()["ax"+str(i)].set_facecolor(background_color)
        locals()["ax"+str(i)].tick_params(axis=u'both', which=u'both',length=0)
        locals()["ax"+str(i)].set_facecolor(background_color) 

plt.show()



# Encoding categorical values
df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
df['Residence_type'] = df['Residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)


# Inverse of Null Accuracy
print('\tInverse of Null Accuracy: ',249/(249+4861))
print('\tNull Accuracy: ',4861/(4861+249))


X  = df[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']]
y = df['stroke']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
X_test.head(5)

oversample = SMOTE()
X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train.ravel())

# Models

# Scale our data in pipeline, then split
rf_pipeline = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42))])
svm_pipeline = Pipeline(steps = [('scale',StandardScaler()),('SVM',SVC(random_state=42))])
logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(random_state=42))])


#Implementing 10-fold CV on "RANDOM FOREST"--"SUPPORT VECTOR MACHINE"--"LOGISTIC REGRESSION"
rf_cv = cross_val_score(rf_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')
svm_cv = cross_val_score(svm_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')
logreg_cv = cross_val_score(logreg_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')

print('====== MEAN F1 SCORES (TRAIN)======')
print('Random Forest Mean :\t\t',rf_cv.mean(),' ≈ ', round(rf_cv.mean()*100, 2),'%')
print('Support Vector Machine Mean :\t',svm_cv.mean(),' ≈ ', round(svm_cv.mean()*100, 2),'%')
print('Logistic Regression Mean :\t',logreg_cv.mean(),' ≈ ', round(logreg_cv.mean()*100, 2),'%')


rf_pipeline.fit(X_train_resh,y_train_resh)
svm_pipeline.fit(X_train_resh,y_train_resh)
logreg_pipeline.fit(X_train_resh,y_train_resh)

rf_pred   =rf_pipeline.predict(X_test)
svm_pred  = svm_pipeline.predict(X_test)
logreg_pred   = logreg_pipeline.predict(X_test)

rf_cm  = confusion_matrix(y_test,rf_pred )
svm_cm = confusion_matrix(y_test,svm_pred)
logreg_cm  = confusion_matrix(y_test,logreg_pred )

rf_f1  = f1_score(y_test,rf_pred)
svm_f1 = f1_score(y_test,svm_pred)
logreg_f1  = f1_score(y_test,logreg_pred)

print('====== MEAN F1 SCORES (TEST)======\n')

print('Random Forest Mean :\t\t',rf_f1)
print('Support Vector Machine Mean :\t',svm_f1)
print('Logistic Regression Mean :\t',logreg_f1)

from imblearn.over_sampling import SMOTE
from collections import Counter

counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE
smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
counter = Counter(y_train_sm)
print('After',counter)

from sklearn.metrics import confusion_matrix, classification_report

print('==================== SCORES FOR RANDOM FOREST ====================\n')
print(classification_report(y_test,rf_pred))
accuracy = accuracy_score(y_test, rf_pred)
print('Accuracy Score:  %.8f'% accuracy)
precision = precision_score(y_test, rf_pred, average='binary')
print('Precision Score: %.8f'% precision)
recall = recall_score(y_test, rf_pred, average='binary')
print('Recall Score:    %.8f' % recall)
score = f1_score(y_test, rf_pred, average='binary')
print('F-Measure:       %.8f' % score)

print('==================== SCORES FOR SVM ====================\n')
print(classification_report(y_test,svm_pred))
accuracy = accuracy_score(y_test, svm_pred)
print('Accuracy Score:  %.8f'% accuracy)
precision = precision_score(y_test, svm_pred, average='binary')
print('Precision Score: %.8f'% precision)
recall = recall_score(y_test, svm_pred, average='binary')
print('Recall Score:    %.8f' % recall)
score = f1_score(y_test, svm_pred, average='binary')
print('F-Measure:       %.8f' % score)


print('==================== SCORES FOR LOGISTIC REGRESSION ====================\n')
print(classification_report(y_test, logreg_pred))
accuracy = accuracy_score(y_test, logreg_pred)
print('Accuracy Score:  %.8f'% accuracy)
precision = precision_score(y_test, logreg_pred, average='binary')
print('Precision Score: %.8f'% precision)
recall = recall_score(y_test, logreg_pred, average='binary')
print('Recall Score:    %.8f' % recall)
score = f1_score(y_test, logreg_pred, average='binary')
print('F-Measure:       %.8f' % score)


#I will try using a grid search to find the optimal parameters for our Models

from sklearn.model_selection import GridSearchCV

n_estimators =[64,100,128,200]
max_features = [2,3,5,7]
bootstrap = [True,False]

param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap}
rfc = RandomForestClassifier()


rfc = RandomForestClassifier(max_features=2,n_estimators=100,bootstrap=True)
rfc.fit(X_train_resh,y_train_resh)
rfc_tuned_pred = rfc.predict(X_test)

print('==================== TUNED SCORES FOR RANDOM FOREST ====================')
print(classification_report(y_test,rfc_tuned_pred))
accuracy = accuracy_score(y_test, rfc_tuned_pred)
print('Accuracy Score:  %.8f'% accuracy)
precision = precision_score(y_test, rfc_tuned_pred, average='binary')
print('Precision Score: %.8f'% precision)
recall = recall_score(y_test, rfc_tuned_pred, average='binary')
print('Recall Score:    %.8f' % recall)
score = f1_score(y_test, rfc_tuned_pred, average='binary')
print('F-Measure:       %.8f' % score)


penalty = ['l1','l2']
C = [0.001, 0.01, 0.1, 1, 10, 100] 
log_param_grid = {'penalty': penalty, 
                  'C': C}
logreg = LogisticRegression()
grid = GridSearchCV(logreg,log_param_grid)


logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(C=0.1,penalty='l2',random_state=42))])
logreg_pipeline.fit(X_train_resh,y_train_resh)
logreg.fit(X_train_resh,y_train_resh)
logreg_tuned_pred   = logreg_pipeline.predict(X_test)

print('==================== TUNED SCORES FOR LOGISTIC REGRESSION ====================\n')
print(classification_report(y_test, logreg_tuned_pred))
accuracy = accuracy_score(y_test,logreg_tuned_pred)
print('Accuracy Score:  %.8f'% accuracy)
precision = precision_score(y_test, logreg_tuned_pred, average='binary')
print('Precision Score: %.8f'% precision)
recall = recall_score(y_test, logreg_tuned_pred, average='binary')
print('Recall Score:    %.8f' % recall)
score = f1_score(y_test, logreg_tuned_pred, average='binary')
print('F-Measure:       %.8f' % score)

svm_pipeline = Pipeline(steps = [('scale',StandardScaler()),('SVM',SVC(C=1000,gamma=0.01,kernel='rbf',random_state=42))])
svm_pipeline.fit(X_train_resh,y_train_resh)
svm_tuned_pred   = svm_pipeline.predict(X_test)

print('==================== TUNED SCORES FOR SVM ====================\n')
print(classification_report(y_test, svm_tuned_pred))
accuracy = accuracy_score(y_test, svm_tuned_pred)
print('Accuracy Score:  %.8f'% accuracy)
precision = precision_score(y_test, svm_tuned_pred, average='binary')
print('Precision Score: %.8f'% precision)
recall = recall_score(y_test, svm_tuned_pred, average='binary')
print('Recall Score:    %.8f' % recall)
score = f1_score(y_test, svm_tuned_pred, average='binary')
print('F-Measure:       %.8f' % score)


# source code: https://www.kaggle.com/ilyapozdnyakov/rain-in-australia-precision-recall-curves-viz
# heeavily modified plotting
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

ns_probs = [0 for _ in range(len(y_test))]
lr_probs = logreg_pipeline.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

y_scores = logreg_pipeline.predict_proba(X_train_resh)[:,1]
precisions, recalls, thresholds = precision_recall_curve(y_train_resh, y_scores)

# Plots
fig = plt.figure(figsize=(13,5), dpi=300,facecolor=background_color)
gs = fig.add_gridspec(1,2, wspace=0.1, hspace=0.5)
ax = gs.subplots()

background_color = "#c1dce6"
fig.patch.set_facecolor(background_color) # figure background color
ax[0].set_facecolor(background_color) 
ax[1].set_facecolor(background_color)

ax[0].grid(color='#2e0742', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
ax[1].grid(color='#2e0742', linestyle=':', axis='y',  dashes=(1,5))

y_scores = logreg_pipeline.predict_proba(X_train_resh)[:,1]

precisions, recalls, thresholds = precision_recall_curve(y_train_resh, y_scores)

ax[0].plot(thresholds, precisions[:-1], 'b--', linewidth=1.5, label='Precision',color='#ffbf00')
ax[0].plot(thresholds, recalls[:-1], 'b.', linewidth=1.5, label='Recall',color='#c94000')
ax[0].set_ylabel('True Positive Rate',loc='bottom')
ax[0].set_xlabel('Thresholds',loc='left')
#plt.legend(loc='center left')
ax[0].set_ylim([0,1])

# plot the roc curve for the model
ax[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Dummy Classifer',color='#c5bfff')
ax[1].plot(lr_fpr, lr_tpr, marker='.', linewidth=2,color='#2e0742')
ax[1].set_xlabel('False Positive Rate',loc='left')
ax[1].set_ylabel('')
ax[1].set_ylim([0,1])

for s in ["top","right","left"]:
    ax[0].spines[s].set_visible(False)
    ax[1].spines[s].set_visible(False) 
    
ax[0].text(-0.1,2,'Model Selection: Considerations',fontsize=32, fontfamily='Quicksand',fontweight='bold')
ax[0].text(-0.1,1.26,
'''
Here we observe how our Logistic Regression model performs when we change the threshold.

We'd like a model that predicts all strokes, but in reality, this would come at a cost.
In fact we can create a model that succeeds in that goal, but it would mean predicting
most people to have a stroke - which in itself would have negative effects.

Therefore, we need to choose a model which not only predicts, correctly, those who have
strokes, but also those who do not.
''',fontsize=18,fontfamily='Quicksand')

ax[0].text(-0.1,1.1,'Precision & Recall',fontsize=14,fontfamily='Quicksand',fontweight='bold', color='#2e0742')
ax[1].text(-0.1,1.1,'ROC: True Positives & False Positives',fontsize=14,fontfamily='Quicksand', fontweight='bold', color='#2e0742')
ax[1].tick_params(axis='y', colors=background_color)

plt.show()

# Make dataframes to plot

rf_df = pd.DataFrame(data=[f1_score(y_test,rf_pred),accuracy_score(y_test, rf_pred), recall_score(y_test, rf_pred),
                   precision_score(y_test, rf_pred), roc_auc_score(y_test, rf_pred)], 
             columns=['Random Forest Score'],
             index=["F1","Accuracy", "Recall", "Precision", "ROC AUC Score"])

svm_df = pd.DataFrame(data=[f1_score(y_test,svm_pred),accuracy_score(y_test, svm_pred), recall_score(y_test, svm_pred),
                   precision_score(y_test, svm_pred), roc_auc_score(y_test, svm_pred)], 
             columns=['Support Vector Machine (SVM) Score'],
             index=["F1","Accuracy", "Recall", "Precision", "ROC AUC Score"])

lr_df = pd.DataFrame(data=[f1_score(y_test,logreg_tuned_pred),accuracy_score(y_test, logreg_tuned_pred), recall_score(y_test, logreg_tuned_pred),
                   precision_score(y_test, logreg_tuned_pred), roc_auc_score(y_test, logreg_tuned_pred)], 
             columns=['Tuned Logistic Regression Score'],
             index=["F1","Accuracy", "Recall", "Precision", "ROC AUC Score"])


df_models = round(pd.concat([rf_df, svm_df, lr_df], axis=1),3)
import matplotlib
colors = ["#2e0742","#c5bfff","#ffbf00"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

background_color = "#c1dce6"

fig = plt.figure(figsize=(14,12), dpi=400, facecolor=background_color) # create figure
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :])

sns.heatmap(df_models.T, cmap=colormap, annot=True, fmt=".2%", vmin=0, vmax=0.95, linewidths=2, cbar=False, ax=ax0, annot_kws={"fontsize":18, "fontfamily":'Quicksand'})
fig.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color) 

ax0.text(0,-2.15,'Model Comparison',fontsize=30,fontweight='bold',fontfamily='Quicksand')
ax0.text(0,-0.9,'Random Forest performs the best for overall Accuracy,but is this enough? \nIs not Recall and preccision are more important in this case?  What about SVM overall score?',fontsize=18,fontfamily='Quicksand')
ax0.tick_params(axis=u'both', which=u'both',length=0)

plt.show()


df_models = round(pd.concat([rf_df, svm_df, lr_df], axis=1),3)
import matplotlib
colors = ["#2e0742","#c5bfff","#ffbf00"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

background_color = "#c1dce6"

fig = plt.figure(figsize=(14,12), dpi=400, facecolor=background_color) # create figure
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :])

sns.heatmap(df_models.T, cmap=colormap, annot=True, fmt=".2%", vmin=0, vmax=0.95, linewidths=2, cbar=False, ax=ax0, annot_kws={"fontsize":18, "fontfamily":'Quicksand'})
fig.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color) 

ax0.text(0,-2.15,'Model Comparison',fontsize=30,fontweight='bold',fontfamily='Quicksand')
ax0.text(0,-0.9,'Random Forest performs the best for overall Accuracy,but is this enough? \nIs not Recall and preccision are more important in this case?  What about SVM overall score?',fontsize=18,fontfamily='Quicksand')
ax0.tick_params(axis=u'both', which=u'both',length=0)

plt.show()