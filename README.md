# Mini-Project--Application-of-NN


## Project Title:
Stock market prediction
## Project Description 
We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.
## Algorithm:
1.import the necessary pakages.

2.install the csv file

3.using the for loop and predict the output

4.plot the graph

5.analyze the regression bar plot
## Program:
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/content/Tesla.csv')
df.head()
df.shape
df.describe()
df.info()
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
df.head()
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1)
df.isnull().sum()
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.distplot(df[col])
plt.show()
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.boxplot(df[col])
plt.show()
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
features = df[['open-close', 'low-high']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
models = [LogisticRegression(), SVC(
kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

print(f'{models[i]} : ')
print('Training Accuracy : ', metrics.roc_auc_score(
	Y_train, models[i].predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(
	Y_valid, models[i].predict_proba(X_valid)[:,1]))
print()
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()
~~~
## Output:

![1](https://user-images.githubusercontent.com/94367917/206681082-784a27d1-0f8f-4b9c-9359-600418a557d9.png)

![2](https://user-images.githubusercontent.com/94367917/206681096-bbbbc178-1639-43f0-ad51-8aa91a1988ce.png)

![3](https://user-images.githubusercontent.com/94367917/206681109-f7970133-a767-4225-bfed-cd143dc9a181.png)

![4](https://user-images.githubusercontent.com/94367917/206681130-3091f452-0c90-4ac5-ac79-365ee003b0b1.png)

![5](https://user-images.githubusercontent.com/94367917/206681146-4c19a5f3-903e-414d-ad89-93709704ee72.png)

![6](https://user-images.githubusercontent.com/94367917/206681163-93c6952d-b4d0-4efe-8223-9059e4bf6311.png)

![7](https://user-images.githubusercontent.com/94367917/206681185-1d9f2e37-bb8f-42ce-bb8c-a18f968421b0.png)


![8](https://user-images.githubusercontent.com/94367917/206681210-df22b3cc-ee02-4b5a-adc2-19ad0ad8018c.png)

![9](https://user-images.githubusercontent.com/94367917/206681238-194dbe10-10ef-4c8b-8fec-c5484fe467b7.png)

![10](https://user-images.githubusercontent.com/94367917/206681265-7a52ff02-bc58-4e05-a1a9-9c5b571dabab.png)

![11](https://user-images.githubusercontent.com/94367917/206681289-71e6791e-a8b4-4410-a25e-ea4b24aac32c.png)


![12](https://user-images.githubusercontent.com/94367917/206681314-d7233bbd-e849-4d8a-8b13-4b03ae034c1a.png)

![13](https://user-images.githubusercontent.com/94367917/206681332-d3f7d0f7-7030-498b-8f93-7e57799280ec.png)


## Advantage :
Python is the most popular programming language in finance. Because it is an object-oriented and open-source language, it is used by many large corporations, including Google, for a variety of projects. Python can be used to import financial data such as stock quotes using the Pandas framework.
## Result:
Thus, stock market prediction is implemented successfully.

### Project by Ragul AC
