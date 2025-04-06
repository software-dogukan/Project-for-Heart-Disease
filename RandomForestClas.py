import pandas as pd
data=pd.read_csv("Cleveland_Synthetic_dataset.csv")
data=data.iloc[:,1:].values
data=pd.DataFrame(data)
X=data.drop(columns=13)
y=data[13]
from sklearn.preprocessing import StandardScaler
s_scaler=StandardScaler()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_train=s_scaler.fit_transform(X_train)
X_test=s_scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))


