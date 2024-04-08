import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import(roc_curve, accuracy_score, roc_auc_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, recall_score)

trainingDataWalk = pd.read_csv('./good training data/good training walking.csv')

trainingDataJump = pd.read_csv('./good training data/good training jumping.csv')

testingDataWalk = pd.read_csv('./good testing data/good testing walking.csv')

testingDataJump = pd.read_csv('./good testing data/good testing jumping.csv')

combinedTrainingDF = pd.concat([trainingDataWalk, trainingDataJump], ignore_index=True)

combinedTestingDF = pd.concat([testingDataWalk, testingDataJump], ignore_index=True)

# iloc is indexing


# Removes the label
trainLabel = combinedTrainingDF.iloc[:, -1]

# Removes the label
testLabel = combinedTestingDF.iloc[:, -1]


# Does the logistical regression
logReg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), logReg)

clf.fit(combinedTrainingDF, trainLabel)
y_pred = clf.predict(combinedTestingDF)
y_clf_prob = clf.predict_proba(combinedTestingDF)


#print('y_pred is: ', y_pred)
#print('y_clf_prob is: ', y_clf_prob)

# Finds the accuracy of the dataset
print('Accuracy is: ', accuracy_score(testLabel, y_pred))

# Finds recall
print('Recall is: ', recall_score(testLabel, y_pred))

cm = confusion_matrix(testLabel, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

fpr, tpr, _ = roc_curve(testLabel, y_clf_prob[:, -1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

auc = roc_auc_score(testLabel, y_clf_prob[:, -1])
print('The AUC is: ', auc)

# Saves the classifier
joblib.dump(clf, 'logregressionmodel.plk')