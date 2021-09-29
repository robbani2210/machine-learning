import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.preprocessing import StandardScaler
from six import StringIO
import pydotplus
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'
df = pd.read_csv(csv_path, header=None)

#menabahkan header column
headers = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
          'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']
df.columns = headers

#merapihkan data
df.replace("?", np.nan, inplace = True)
df = df.dropna(axis='rows') 
df = df.astype(float)

#belajar data
dtree = DecisionTreeClassifier()
dtree = dtree.fit(df[['AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
          'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']], df['Class'])

#fitur = list(df.columns[1:20])
# [text]
# tree.plot_tree(dtree.fit(df[['AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
#           'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
#           'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']], df['Class']))
# r = export_text(dtree, feature_names=fitur)
# print(r)

# [picture]
# dot_data=StringIO()
# export_graphviz(dtree, out_file=dot_data, filled= True, rounded=True, 
#                 special_characters=True, feature_names=fitur, class_names=[1,2])
# graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('transport.png')

#normalisasi
scaler = StandardScaler()
scaler.fit(df[['AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
          'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']])

#uji data
X_train, X_test , Y_train, Y_test = train_test_split(df[['AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA',
          'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
          'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']], df['Class'], test_size = 0.3, random_state=0 )
dtree.fit(X_train, Y_train)
prediksi_dtree = dtree.predict(X_test)

dtree.fit(X_train, Y_train)
prediksi_dtree = dtree.predict(X_test)
print('nilai akurasi : ', end=(""))
print(accuracy_score( Y_test, prediksi_dtree))
