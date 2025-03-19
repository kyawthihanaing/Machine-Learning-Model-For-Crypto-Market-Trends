import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


data = pd.read_csv('data.csv', encoding='latin1')
data1 = data[data['label'] == 0][:20000]
data2 = data[data['label'] == 1][:20000]
data = pd.concat([data1, data2])
data = data.reset_index(drop=True)

labels = data['label'].values
tw = data['tweets'].values
X_train, X_test, Y_train, Y_test = train_test_split(tw, labels, test_size=0.30, random_state=None, stratify=labels)

#--Feature Extraction--#
vec = CountVectorizer()
vec.fit(X_train)
#--Feature Vectorization--#
train_cv = vec.transform(X_train)
test_cv = vec.transform(X_test)


