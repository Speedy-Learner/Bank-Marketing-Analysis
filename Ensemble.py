
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()


df = pd.read_csv('data.csv')
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
