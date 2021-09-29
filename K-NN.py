import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'
df = pd.read_csv(csv_path, header=None)

headers = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
          'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']
df.columns = headers
df.replace("?", np.nan, inplace = True)
# print(df.isna().mean()) menghitung jumlah null per-kolom
df = df.dropna(axis='rows') 
# treshold = len(df) * 0.9 
# df = df.dropna(thresh = treshold ,axis='rows') menghapus dgn parameter NaN di atas 10 % 

print(df.shape)

df.isna()

scaler = StandardScaler()
scaler.fit(df[['AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
          'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']])

model_knn = KNeighborsClassifier(n_neighbors=3)
X_train, X_test , Y_train, Y_test = train_test_split(df[['AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
          'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']], 
                                                      df['Class'], test_size = 0.3, random_state=0 )
model_knn.fit(X_train, Y_train)
prediksi_knn = model_knn.predict(X_test)
print('nilai akurasi : ', end=(""))
print(accuracy_score( Y_test, prediksi_knn))
