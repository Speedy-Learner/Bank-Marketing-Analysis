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


#print("Mean Balance For All Job Categories:\n",df[['job', 'balance']].groupby(['job'], as_index=False).mean())
#print("Mean campaign  For All Job Categories:\n",df[['job', 'campaign']].groupby(['job'], as_index=False).mean())

#print("Important Month:\n",df['month'].value_counts())
#df['y'] = LabelEncoder().fit_transform(df['y'])

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


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# Compute fpr, tpr, thresholds and roc auc
probs = clf.predict_proba(X_test)
# Compute ROC curve and area the curve

#print("******",roc_curve(y_test, probs))

'''
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])'''









#ensemble
df1 = pd.read_csv('Bank_Data12_less6.csv')
y1 = np.array(df1['y'])
X1=pd.get_dummies(df1[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']])
scaler = StandardScaler().fit(X1['age'])
X1['age'] = scaler.transform(X1['age'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X1['campaign'])
X1['campaign'] = scaler.transform(X1['campaign'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X1['balance'])
X1['balance'] = scaler.transform(X1['balance'].values.reshape(-1, 1))

'''
df2 = pd.read_csv('second.csv')
y2 = np.array(df2['y'])
X2=pd.get_dummies(df2[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']])
scaler = StandardScaler().fit(X2['age'])
X2['age'] = scaler.transform(X2['age'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X2['campaign'])
X2['campaign'] = scaler.transform(X2['campaign'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X2['balance'])
X2['balance'] = scaler.transform(X2['balance'].values.reshape(-1, 1))

df3 = pd.read_csv('third.csv')
y3 = np.array(df3['y'])
X3=pd.get_dummies(df3[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']])
scaler = StandardScaler().fit(X3['age'])
X3['age'] = scaler.transform(X3['age'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X3['campaign'])
X3['campaign'] = scaler.transform(X3['campaign'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X3['balance'])
X3['balance'] = scaler.transform(X3['balance'].values.reshape(-1, 1))


df4 = pd.read_csv('fourth.csv')
y4 = np.array(df4['y'])
X4=pd.get_dummies(df4[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']])
scaler = StandardScaler().fit(X4['age'])
X4['age'] = scaler.transform(X4['age'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X4['campaign'])
X4['campaign'] = scaler.transform(X4['campaign'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X4['balance'])
X4['balance'] = scaler.transform(X4['balance'].values.reshape(-1, 1))


df5 = pd.read_csv('Bank_Data12_less6.csv')
y5 = np.array(df5['y'])
X5=pd.get_dummies(df5[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']])
scaler = StandardScaler().fit(X5['age'])
X5['age'] = scaler.transform(X5['age'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X5['campaign'])
X5['campaign'] = scaler.transform(X5['campaign'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X5['balance'])
X5['balance'] = scaler.transform(X5['balance'].values.reshape(-1, 1))'''


import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()


df = pd.read_csv('first.csv')
#data_y = np.array(df['y'])
df['y'] = LabelEncoder().fit_transform(df['y'])
y1=pd.DataFrame(df['y'])

X1=pd.get_dummies(df[['campaign','balance','age','job','marital','education','default','housing','loan','contact','month']])
scaler = StandardScaler().fit(X1['age'])
X1['age'] = scaler.transform(X1['age'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X1['campaign'])
X1['campaign'] = scaler.transform(X1['campaign'].values.reshape(-1, 1))
scaler = StandardScaler().fit(X1['balance'])
X1['balance'] = scaler.transform(X1['balance'].values.reshape(-1, 1))

eclf = VotingClassifier(estimators=[('lr1', clf1), ('lr2', clf2),('lr2', clf3)], voting='soft')

eclf1 = eclf.fit(X1, y1)
#scores = model_selection.cross_val_score(eclf1, X1, y1, cv=10)
#print("Accuracy of Voting Classifier: %0.2f " % (scores.mean()))


# predict class probabilities for all classifiers
probas = [c.fit(X1, y1).predict_proba(X1) for c in (clf1, clf2, clf3, eclf)]
#print("Voting Classifier:",eclf1.predict(X1))

class1_1 = [pr[0, 0] for pr in probas]
class2_1 = [pr[0, 1] for pr in probas]
print(class1_1,"\n",class2_1)

# plotting
N = 4  # number of groups
ind = np.arange(N)  # group positions
width = 0.35  # bar width
fig, ax = plt.subplots()
# bars for classifier 1-3
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')
# bars for VotingClassifier
p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
            color='blue', edgecolor='k')
p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
            color='steelblue', edgecolor='k')
# plot annotations
plt.axvline(2.8, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['LogisticRegr\nweight 1',
                    'GaussianNB\nweight 1',
                    'RandomForest\nweight 5',
                    'VotingClassifier\n(average probabilities)'],
                   rotation=30,
                   ha='right')
plt.ylim([0, 1])
plt.title('Class probabilities for 1st sample by different classifiers')
plt.legend([p1[0], p2[0]], ['No', 'Yes'], loc='upper left')
plt.show()


from sklearn import metrics as m
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve

log_cols = ["Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]
log = pd.DataFrame(columns=log_cols)

y_pred=eclf1.predict(X1)
accuracy = m.accuracy_score(y1, y_pred)
precision = m.precision_score(y1, y_pred, average='macro')
recall = m.recall_score(y1, y_pred, average='macro')
roc_auc = roc_auc_score(y_pred, y1)
f1_score = m.f1_score(y1, y_pred, average='macro')
log_entry = pd.DataFrame([[accuracy, precision, recall, f1_score, roc_auc]], columns=log_cols)
log = log.append(log_entry)
print("\n",log)

#ROC CURVE
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlim([-0.05, 1.05])
ax1.set_ylim([-0.05, 1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([-0.05, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

pred_prob = eclf1.predict_proba(X1)[:, 1]
p, r, _ = precision_recall_curve(y1, pred_prob)
tpr, fpr, _ = roc_curve(y1, pred_prob)
ax1.plot(r, p)
ax2.plot(tpr, fpr)
ax1.legend(loc='lower left')
ax2.legend(loc='lower left')
plt.show()





'''
eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
eclf2 = eclf2.fit(X, y)
print(eclf2.predict(X))

eclf3 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[2,1,1], flatten_transform=True)
eclf3 = eclf3.fit(X, y)
print(eclf3.predict(X))
print(eclf3.transform(X).shape)'''





