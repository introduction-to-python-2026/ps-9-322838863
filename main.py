import pandas as pd
df = pd.read_csv('parkinsons.csv')
df.head()
df =df.dropna()

x = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_scaled, y)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
