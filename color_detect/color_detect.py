import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

model = SVC(kernel='linear', C=1.0, random_state=1)

file='colors_rgb.csv'
df = pd.read_csv(file, sep=',')
print(df.shape)
print(df.columns.values)
print(df.iloc[0][0:3])

X = df.values[:, [0, 1, 2]]
y = df.values[:, [3]]
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)
# print(y_pred)
from sklearn.metrics import accuracy_score

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

X_test2 = [37, 49, 33]
X_test2_std = sc.transform([X_test2])
pred = model.predict(X_test2_std)
print(pred)