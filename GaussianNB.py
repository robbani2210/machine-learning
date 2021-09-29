import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'
df = pd.read_csv(csv_path, header=None)
headers = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
          'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']
df.replace("?", np.nan, inplace = True)
df.columns = headers

columns_del = ['AGE','BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME']
df.drop(columns_del , inplace=True, axis=1)
df = df.dropna(axis='rows') 
print(df.head(10))

model = MultinomialNB()
model.fit(df[['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
           'HISTOLOGY']], df['Class'])

perdicted = model.predict([[2,1,2,2,2,2,1,2,2,2,2,2,1]])
print(perdicted)
# print(df.dtypes)
print(df.shape)

x = df[['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
            'HISTOLOGY']]
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

gnb = GaussianNB()
gnb.fit(x_train , y_train)
y_pred = gnb.predict(x_test)


print(confusion_matrix( y_test, y_pred))
print(accuracy_score( y_test, y_pred))
