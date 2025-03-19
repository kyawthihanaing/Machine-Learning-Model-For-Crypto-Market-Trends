import pandas as pd
import numpy as np
import joblib
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve


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

print("Processing time for NB-CV:")
print(timeElapsed2, "seconds")

#NB
nb = MultinomialNB()
# Fitting vectorized X_train and associated labels into the NB algorithm
notuning = nb.fit(train_cv, Y_train)

# K-fold cross validation
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
param_grid = [{'alpha': np.linspace(0.01, 1, 100)}]
# Hyperparameter Tuning and cross validation
grid_NB = GridSearchCV(estimator=nb, param_grid=param_grid, scoring='accuracy', cv=kfolds, n_jobs=-1, verbose=1)

# Fitting the train data into grid search cv to get tuned model
start = time.process_time()
grid_NB.fit(train_cv, Y_train)
timeElapsed = time.process_time() - start

# Printing the best parameters, mean cross validation score, and processing time
print('BEST PARAMETERS: {0}'.format(grid_NB.best_params_))
print('BEST MEAN CROSS-VALIDATION SCORE: {0:.4f}'.format(grid_NB.best_score_))
print("Processing time for hyperparameter tuning:")
print(timeElapsed, "seconds")

# Converting the results to data frame and showing the parameter results in detail
parameterResults = pd.concat([pd.DataFrame(grid_NB.cv_results_["params"]),
                              pd.DataFrame(grid_NB.cv_results_["mean_test_score"], columns=["Accuracy"]),
                              pd.DataFrame(grid_NB.cv_results_["mean_score_time"], columns=["Mean Score Time"])],
                             axis=1)
parameterResults = parameterResults.sort_values("Accuracy", ascending=False)
print(parameterResults)


model = grid_NB.best_estimator_
# Predicting the actual class label and class probability
predicted_labels = model.predict(test_cv)
pred_proba = model.predict_proba(test_cv)[:, 1]
predicted_labels_notuning = notuning.predict(test_cv)
pred_proba_notuning = notuning.predict_proba(test_cv)[:, 1]

# Printing the evaluation metrics
print("The test set accuracy of the MNB classifier (without tuning) is ", accuracy_score(Y_test, predicted_labels_notuning))
print("\nThe classification report with metrics (no tuning) - \n", classification_report(Y_test, predicted_labels_notuning))
print("ROC_AUC score (NO TUNING):")
print(roc_auc_score(Y_test, pred_proba_notuning))
print("The test set accuracy of the MNB classifier is ", accuracy_score(Y_test, predicted_labels))
print("\nThe classification report with metrics - \n", classification_report(Y_test, predicted_labels))
print("ROC_AUC score:")
print(roc_auc_score(Y_test, pred_proba))

# Creating the confusion matrix for the tuned model
cm = confusion_matrix(Y_test, predicted_labels, labels=grid_NB.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_NB.best_estimator_.classes_)
disp.plot()
plt.show()

# Creating the confusion matrix for the model trained without tuning
cm2 = confusion_matrix(Y_test, predicted_labels_notuning, labels=notuning.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=notuning.classes_)
disp2.plot()
plt.show()

# Saving the NB Models and pipelines
joblib.dump(grid_NB.best_estimator_, 'nbModel_CV.pkl', compress = 1)

joblib.dump(notuning, 'nbModel_CV_notuning.pkl', compress = 1)

nbPipeline = Pipeline(
    [('vect', vec), ('NB', grid_NB.best_estimator_)]
)

joblib.dump(nbPipeline, 'nbPipeline_CV.pkl', compress = 1)

nbPipeline2 = Pipeline(
    [('vect', vec), ('NB', notuning)]
)

joblib.dump(nbPipeline2, 'nbPipeline_CV_notuning.pkl', compress = 1)

# Function to create the roc curve
def get_roc_curve(model, X, y):
  pred_proba = model.predict_proba(X)[:, 1]
  fpr, tpr, _ = roc_curve(y, pred_proba)
  return fpr, tpr

# Passing in tuned model, vectorized X test, and Y test to the roc curve function
roc_nb1 = get_roc_curve(grid_NB.best_estimator_, test_cv, Y_test)
fpr, tpr = roc_nb1
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
roc_nb2 = get_roc_curve(notuning, test_cv, Y_test)
fpr, tpr = roc_nb2
plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()

