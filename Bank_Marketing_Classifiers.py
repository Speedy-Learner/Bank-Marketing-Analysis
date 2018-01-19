import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from prettytable import PrettyTable
from pandas import ExcelWriter,ExcelFile
from sklearn.preprocessing import LabelEncoder

#Reading Data
df = pd.read_csv('first.csv')
y = np.array(df['y'])

print("\nMean Of Age,Balance and Campaign For Marital Category:\n",df[['marital','age', 'balance','campaign']].groupby(['marital'], as_index=False).mean())
print("\nMean Of Age,Balance and Campaign For  Education Category:\n",df[['education','age', 'balance','campaign']].groupby(['education'], as_index=False).mean())
print("\nMean Of Age,Balance and Campaign For Job Category:\n",df[['job','age', 'balance','campaign']].groupby(['job'], as_index=False).mean())
print("\nMean Of Age,Balance and Campaign For Contact Category:\n",df[['contact','age', 'balance', 'campaign']].groupby(['contact'], as_index=False).mean())
print("\nMean Of Age,Balance and Campaign For Month Category:\n",df[['month','age', 'balance', 'campaign']].groupby(['month'], as_index=False).mean())

#Heat map of correlation for  understanding of which variables are important
f, ax = plt.subplots(figsize=(10, 13))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),annot=True, square=True, ax=ax)

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    plt.savefig("plot_distribution.png")
#plot_distribution( df , var = 'age' , target = 'y' , row = 'balance' )

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()
    plt.savefig("plot_categories.png")

mixed_data=df[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']]

df['n_age'] = pd.cut(df['age'], [0,18,32,48,64,100],labels=['0-18','18-32','32-48','48-64','64-above'])
df['n_balance'] = pd.cut(df['balance'], [-10000,-5000,0,2000,5000,10000,20000,40000,60000,80000,120000],labels=['-10k-5k','-5k-0','0-2k','2k-5k','5k-10k','10k-20k',
                                                                                                        '20k-40k','40k-60k','60k-80k','80k-above'])
df['n_campaign'] = pd.cut(df['campaign'], [0,1,2,4,6,8,10,15,20,40,100],labels=['0-1','1-2','2-4','4-6','6-8','8-10','10-15','15-20','20-40','40-above'])

categorical_data=df[['n_campaign','n_balance','n_age','job','marital','education','default','housing','loan','contact','month','y']]

binary_data=pd.get_dummies(df[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']])
print("Binary Header:-----------------\n",list(binary_data))


#Feature Importance:
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame(
        model.feature_importances_  ,
        columns = [ 'Importance' ] ,
        index = X.columns
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    plt.show()
    print (model.score( X , y ))

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )

#plot_variable_importance(binary_data,y)


#Shuffling and Spliting Data into train and test dataset by 70:30
X_train, X_test, y_train, y_test = train_test_split(binary_data, y, test_size = 0.34)

#Logistic Regression:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
prediction_data = pd.read_csv('Prediction_binarized_data.csv')
#prediction_data=pd.get_dummies(df11[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']])

print("prediction_data:",prediction_data)
scaler = StandardScaler().fit(X_train['age'])
X_train['age'] = scaler.transform(X_train['age'].values.reshape(-1, 1))
X_test['age'] = scaler.transform(X_test['age'].values.reshape(-1, 1))
prediction_data['age'] = scaler.transform(prediction_data['age'].values.reshape(-1, 1))


scaler = StandardScaler().fit(X_train['campaign'].values.reshape(-1, 1))
X_train['campaign'] = scaler.transform(X_train['campaign'].values.reshape(-1, 1))
X_test['campaign'] = scaler.transform(X_test['campaign'].values.reshape(-1, 1))
prediction_data['campaign'] = scaler.transform(prediction_data['campaign'].values.reshape(-1, 1))


scaler = StandardScaler().fit(X_train['balance'].values.reshape(-1, 1))
X_train['balance'] = scaler.transform(X_train['balance'].values.reshape(-1, 1))
X_test['balance'] = scaler.transform(X_test['balance'].values.reshape(-1, 1))
prediction_data['balance'] = scaler.transform(prediction_data['balance'].values.reshape(-1, 1))

from sklearn.metrics import confusion_matrix
def prediction(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = cross_val_score(model, X_train, y_train, cv = 5)
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Unbalanced confusion_matrix\n",confusion_matrix)
    return [model, accuracy]
#class_weight='balanced'
lr  = LogisticRegression()
acc = prediction(lr, X_train, y_train, X_test, y_test)
print("5 Cross-fold:\n",acc[0],'\n Accuracy:',acc[1])

#training
clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Assigning Weight: Confusion Matrix:\n",confusion_matrix)


fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,2,1)
sns.heatmap(confusion_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
plt.title('Confusion Matrix')
plt.ylabel('Real Classes')
plt.xlabel('Predicted Classes')
plt.show()

accuracy = clf.score(X_test,y_test)
print('\n Accuracy:',accuracy)
print("Predictions:",clf.predict(prediction_data))
