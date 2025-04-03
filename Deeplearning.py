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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
X_train=s_scaler.fit_transform(X_train)
X_test=s_scaler.transform(X_test)
#from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
model.add(Dense(13))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
early_stopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1)
model.fit(x=X_train,y=y_train,epochs=2)
#KNeighborsClassifier(n_neighbors=5)#n_neighbors=1,metric="minkowski"

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
y_pred = (y_pred > 0.5).astype(int)  # 0.5'ten büyük değerleri 1, diğerlerini 0 yapar
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))


