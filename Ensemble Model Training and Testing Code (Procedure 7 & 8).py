import pandas as pd
import joblib
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve


# Loading the train data with sentiment labels
df = pd.read_csv('trainData.csv', encoding='latin1')

# Getting 20000 positive and negative tweets from
# the train data and combining them
data1 = df[df['label'] == 0][:20000]
data2 = df[df['label'] == 1][:20000]
data = pd.concat([data1, data2])
data = data.reset_index(drop=True)

# Converting to view objects
X = data['tweets'].values
Y = data['label'].values

# Splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

# Vectorization
start2 = time.process_time()
vec = CountVectorizer()
vec.fit(X_train)
train_cv = vec.transform(X_train)
test_cv = vec.transform(X_test)
timeElapsed2 = time.process_time() - start2

print("Processing time for Ensemble-CV:")
print(timeElapsed2, "seconds")

# Loading the SVM, DT, and NB models
model1 = joblib.load('svmModel.pkl')

model2 = joblib.load('nbModel.pkl')

model3 = joblib.load('decisionTreeModel.pkl')

# Specifying the estimators for stacking classifier
estimators = [
    ('SVM',model1),
    ('NB',model2),
    ('DT',model3)

]

# Stacking the models to create an ensemble model
stack_model = StackingClassifier(
    estimators = estimators, final_estimator=LogisticRegression()
)
# Fitting vectorized X_train and associated labels into the stacking classifier
notuning = stack_model.fit(train_cv, Y_train)

# K-fold cross validation
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
params = {
    'final_estimator__C': [0.1, 10]
}
# Hyperparameter Tuning and cross validation
grid_ensemble = GridSearchCV(stack_model,
                  param_grid = params,
                  cv = kfolds,
                  scoring="accuracy",
                  verbose=1,
                  n_jobs=-1)

# Fitting the train data into grid search cv to get tuned model
start = time.process_time()
grid_ensemble.fit(train_cv, Y_train)
timeElapsed = time.process_time() - start

# Printing the best parameters, mean cross validation score, and processing time
print('BEST PARAMETERS: {0}'.format(grid_ensemble.best_params_))
print('BEST MEAN CROSS-VALIDATION SCORE: {0:.4f}'.format(grid_ensemble.best_score_))
print("Processing time for hyperparameter tuning:")
print(timeElapsed, "seconds")

# Converting the results to data frame and showing the parameter results in detail
parameterResults = pd.concat([pd.DataFrame(grid_ensemble.cv_results_["params"]),
                              pd.DataFrame(grid_ensemble.cv_results_["mean_test_score"], columns=["Accuracy"]),
                              pd.DataFrame(grid_ensemble.cv_results_["mean_score_time"], columns=["Mean Score Time"])],
                             axis=1)
parameterResults = parameterResults.sort_values("Accuracy", ascending=False)
print(parameterResults)

# Predicting the actual class label and class probability
pred = grid_ensemble.best_estimator_.predict(test_cv)
pred_proba = grid_ensemble.best_estimator_.predict_proba(test_cv)[:, 1]
pred_notuning = notuning.predict(test_cv)
pred_proba_notuning = notuning.predict_proba(test_cv)[:, 1]

# Printing the evaluation metrics
print("The test set accuracy of the Ensemble classifier (without tuning) is ", accuracy_score(Y_test, pred_notuning))
print("\nThe classification report with metrics (no tuning) - \n", classification_report(Y_test, pred_notuning))
print("ROC_AUC score (NO TUNING):")
print(roc_auc_score(Y_test, pred_proba_notuning))
print("The test set accuracy of the Ensemble classifier is ", accuracy_score(Y_test, pred))
print("\nThe classification report with metrics - \n", classification_report(Y_test, pred))
print("ROC_AUC score:")
print(roc_auc_score(Y_test, pred_proba))

# Creating the confusion matrix for the tuned model
cm = confusion_matrix(Y_test, pred, labels=grid_ensemble.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_ensemble.best_estimator_.classes_)
disp.plot()
plt.show()

# Creating the confusion matrix for the model trained without tuning
cm2 = confusion_matrix(Y_test, pred_notuning, labels=notuning.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=notuning.classes_)
disp2.plot()
plt.show()

# Saving the DT Models and pipelines
joblib.dump(grid_ensemble.best_estimator_, 'ensembleModel_CV.pkl', compress = 1)

joblib.dump(notuning, 'ensembleModel_CV_notuning.pkl', compress = 1)

ensemblePipeline = Pipeline(
    [('vect', vec), ('Ensemble', grid_ensemble.best_estimator_)]
)

joblib.dump(ensemblePipeline, 'ensemblePipeline_CV.pkl', compress = 1)

ensemblePipeline2 = Pipeline(
    [('vect', vec), ('Ensemble', notuning)]
)

joblib.dump(ensemblePipeline2, 'ensemblePipeline_CV_notuning.pkl', compress = 1)

# Function to create the roc curve
def get_roc_curve(model, X, y):
  pred_proba = model.predict_proba(X)[:, 1]
  fpr, tpr, _ = roc_curve(y, pred_proba)
  return fpr, tpr

# Passing in tuned model, vectorized X test, and Y test to the roc curve function
roc_ensemble = get_roc_curve(grid_ensemble.best_estimator_, test_cv, Y_test)
fpr, tpr = roc_ensemble
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
roc_ensemble2 = get_roc_curve(notuning, test_cv, Y_test)
fpr, tpr = roc_ensemble2
plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()