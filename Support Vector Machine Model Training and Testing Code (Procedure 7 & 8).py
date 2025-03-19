import pandas as pd
import numpy as np
import joblib
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from IPython.display import display
from sklearn import svm
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

df = pd.read_csv('trainDataNoLabel.csv', encoding='latin1')

# Getting the sentiment labels for the tweets
sentiAnalyzer = SentimentIntensityAnalyzer()
df['scores'] = df['tweets'].apply(lambda tweets: sentiAnalyzer.polarity_scores(tweets))
df['compound'] = df['scores'].apply(lambda d: d['compound'])
df['score'] = df['compound'].apply(lambda score: 'pos' if score > 0 else ('neu' if score == 0 else 'neg'))
df['label'] = df['compound'].apply(lambda score: 1 if score > 0 else (3 if score == 0 else 0))

# Saving the train data with sentiment labels
df.to_csv("trainData.csv", index=False)

# Displaying the value count for sentiment labels
display(df['label'].value_counts())

# Getting 20000 positive and negative tweets from
# the train data and combining them
data1 = df[df['label'] == 0][:20000]
data2 = df[df['label'] == 1][:20000]
data = pd.concat([data1, data2])
data = data.reset_index(drop=True)

# Splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data['tweets'], data['label'], test_size=0.30, random_state=0)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)

# Vectorization
start2 = time.process_time()
vec = CountVectorizer()
vec.fit(X_train)
train_cv = vec.transform(X_train)
test_cv = vec.transform(X_test)
timeElapsed2 = time.process_time() - start2

print("Processing time for SVM-CV:")
print(timeElapsed2, "seconds")

np.random.seed(1)

# SVM
model_svm = svm.SVC(probability=True, kernel="linear", class_weight="balanced")
# Fitting vectorized X_train and associated labels into the SVM algorithm
notuning = model_svm.fit(train_cv, Y_train)

# K-fold cross validation
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# Hyperparameter Tuning and cross validation
grid_svm = GridSearchCV(model_svm,
                        param_grid={'C': [0.01, 0.1, 1]
                                    },
                        cv=kfolds,
                        scoring="accuracy",
                        verbose=1,
                        n_jobs=-1)

# Fitting the train data into grid search cv to get tuned model
start = time.process_time()
grid_svm.fit(train_cv, Y_train)
timeElapsed = time.process_time() - start

# Printing the best parameters, mean cross validation score, and processing time
print('BEST PARAMETERS: {0}'.format(grid_svm.best_params_))
print('BEST MEAN CROSS-VALIDATION SCORE: {0:.4f}'.format(grid_svm.best_score_))
print("Processing time for hyperparameter tuning:")
print(timeElapsed, "seconds")

# Converting the results to data frame and showing the parameter results in detail
parameterResults = pd.concat([pd.DataFrame(grid_svm.cv_results_["params"]),
                              pd.DataFrame(grid_svm.cv_results_["mean_test_score"], columns=["Accuracy"]),
                              pd.DataFrame(grid_svm.cv_results_["mean_score_time"], columns=["Mean Score Time"])],
                             axis=1)
parameterResults = parameterResults.sort_values("Accuracy", ascending=False)
print(parameterResults)

# Predicting the actual class label and class probability
pred = grid_svm.best_estimator_.predict(test_cv)
pred_proba = grid_svm.best_estimator_.predict_proba(test_cv)[:, 1]
pred_notuning = notuning.predict(test_cv)
pred_proba_notuning = notuning.predict_proba(test_cv)[:, 1]

#Printing the evaluation metrics
print("The test set accuracy of the SVM classifier (without tuning) is ", accuracy_score(Y_test, pred_notuning))
print("\nThe classification report with metrics (no tuning) - \n", classification_report(Y_test, pred_notuning))
print("ROC_AUC score (NO TUNING):")
print(roc_auc_score(Y_test, pred_proba_notuning))
print("The test set accuracy of the SVM classifier is ", accuracy_score(Y_test, pred))
print("\nThe classification report with metrics - \n", classification_report(Y_test, pred))
print("ROC_AUC score:")
print(roc_auc_score(Y_test, pred_proba))

# Creating the confusion matrix for the tuned model
cm = confusion_matrix(Y_test, pred, labels=grid_svm.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_svm.best_estimator_.classes_)
disp.plot()
plt.show()

# Creating the confusion matrix for the model trained without tuning
cm2 = confusion_matrix(Y_test, pred_notuning, labels=notuning.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=notuning.classes_)
disp2.plot()
plt.show()

# Saving the SVM Models and pipelines
joblib.dump(grid_svm.best_estimator_, 'svmModel_CV.pkl', compress=1)

joblib.dump(notuning, 'svmModel_CV_notuning.pkl', compress=1)

svmPipeline = Pipeline(
    [('vect', vec), ('SVM', grid_svm.best_estimator_)]
)

joblib.dump(svmPipeline, 'svmPipeline_CV.pkl', compress=1)

svmPipeline2 = Pipeline(
    [('vect', vec), ('SVM', notuning)]
)

joblib.dump(svmPipeline2, 'svmPipeline_CV_notuning.pkl', compress=1)

# Function to create the roc curve
def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr

# Passing in tuned model, vectorized X test, and Y test to the roc curve function
roc_svm1 = get_roc_curve(grid_svm.best_estimator_, test_cv, Y_test)
fpr, tpr = roc_svm1
plt.figure(figsize=(14, 8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()

# Passing in model without tuning, vectorized X test, and Y test to the roc curve function
roc_svm2 = get_roc_curve(notuning, test_cv, Y_test)
fpr, tpr = roc_svm2
plt.figure(figsize=(14, 8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()
