import pandas as pd
import joblib
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

# Loading the train data with sentiment labels
data = pd.read_csv('trainData.csv', encoding='latin1')

# Getting 20000 positive and negative tweets from
# the train data and combining them
data1 = data[data['label'] == 0][:20000]
data2 = data[data['label'] == 1][:20000]
data = pd.concat([data1, data2])
data = data.reset_index(drop=True)

# Converting to view objects
labels = data['label'].values
tw = data['tweets'].values

# Splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(tw, labels, test_size=0.30, random_state=0, stratify=labels)

# Vectorization
start2 = time.process_time()
vec = CountVectorizer()
vec.fit(X_train)
train_cv = vec.transform(X_train)
test_cv = vec.transform(X_test)
timeElapsed2 = time.process_time() - start2

print("Processing time for DT-CV:")
print(timeElapsed2, "seconds")

#DT
dt = DecisionTreeClassifier()
# Fitting vectorized X_train and associated labels into the DT algorithm
notuning = dt.fit(train_cv, Y_train)

# K-fold cross validation
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
params = {
    'min_samples_leaf': [1, 5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
# Hyperparameter Tuning and cross validation
grid_dt = GridSearchCV(dt,
                        param_grid=params,
                        cv=kfolds, n_jobs=-1, verbose=1, scoring = "accuracy")

# Fitting the train data into grid search cv to get tuned model
start = time.process_time()
grid_dt.fit(train_cv, Y_train)
timeElapsed = time.process_time() - start

# Printing the best parameters, mean cross validation score, and processing time
print('BEST PARAMETERS: {0}'.format(grid_dt.best_params_))
print('BEST MEAN CROSS-VALIDATION SCORE: {0:.4f}'.format(grid_dt.best_score_))
print("Processing time for hyperparameter tuning:")
print(timeElapsed, "seconds")

# Converting the results to data frame and showing the parameter results in detail
parameterResults = pd.concat([pd.DataFrame(grid_dt.cv_results_["params"]),
                              pd.DataFrame(grid_dt.cv_results_["mean_test_score"], columns=["Accuracy"]),
                              pd.DataFrame(grid_dt.cv_results_["mean_score_time"], columns=["Mean Score Time"])],
                             axis=1)
parameterResults = parameterResults.sort_values("Accuracy", ascending=False)
print(parameterResults)

# Predicting the actual class label and class probability
pred = grid_dt.best_estimator_.predict(test_cv)
pred_proba = grid_dt.best_estimator_.predict_proba(test_cv)[:, 1]
pred_notuning = notuning.predict(test_cv)
pred_proba_notuning = notuning.predict_proba(test_cv)[:, 1]

# Printing the evaluation metrics
print("The test set accuracy of the DT classifier (without tuning) is ", accuracy_score(Y_test, pred_notuning))
print("\nThe classification report with metrics (no tuning) - \n", classification_report(Y_test, pred_notuning))
print("ROC_AUC score (NO TUNING):")
print(roc_auc_score(Y_test, pred_proba_notuning))
print("The test set accuracy of the DT classifier is ", accuracy_score(Y_test, pred))
print("\nThe classification report with metrics - \n", classification_report(Y_test, pred))
print("ROC_AUC score:")
print(roc_auc_score(Y_test, pred_proba))

# Creating the confusion matrix for the tuned model
cm = confusion_matrix(Y_test, pred, labels=grid_dt.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_dt.best_estimator_.classes_)
disp.plot()
plt.show()

# Creating the confusion matrix for the model trained without tuning
cm2 = confusion_matrix(Y_test, pred_notuning, labels=notuning.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=notuning.classes_)
disp2.plot()
plt.show()

# Saving the DT Models and pipelines
joblib.dump(grid_dt.best_estimator_, 'decisionTreeModel_CV.pkl', compress = 1)

joblib.dump(notuning, 'decisionTreeModel_CV_notuning.pkl', compress = 1)

dtPipeline = Pipeline(
    [('vect', vec), ('DT', grid_dt.best_estimator_)]
)

joblib.dump(dtPipeline, 'dtPipeline_CV.pkl', compress = 1)

dtPipeline2 = Pipeline(
    [('vect', vec), ('DT', notuning)]
)

joblib.dump(dtPipeline2, 'dtPipeline_CV_notuning.pkl', compress = 1)

# Function to create the roc curve
def get_roc_curve(model, X, y):
  pred_proba = model.predict_proba(X)[:, 1]
  fpr, tpr, _ = roc_curve(y, pred_proba)
  return fpr, tpr

# Passing in tuned model, vectorized X test, and Y test to the roc curve function
roc_dt = get_roc_curve(grid_dt.best_estimator_, test_cv, Y_test)
fpr, tpr = roc_dt
plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()

# Passing in model without tuning, vectorized X test, and Y test to the roc curve function
roc_dt2 = get_roc_curve(notuning, test_cv, Y_test)
fpr, tpr = roc_dt2
plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()


