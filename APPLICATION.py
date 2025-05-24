import tkinter as tk
from tkinter import ttk

from sympy import public


def get_sex_numeric_value():
    sex = sex_var.get()
    return 0 if sex == "Female" else 1




def get_cp_numeric_value():
    cp_map = {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-Anginal Pain": 3,
        "Asymptomatic": 4
    }
    return cp_map[chestPain_var.get()]




def get_fastingBloodSugar_value():
    return 1 if fastingBloodSugar_var.get() == "True" else 0



def get_restingECG_numeric_value():
    restingECG_map = {
        "Normal": 0,
        "Showing ST-T wave abnormality": 1,
        "Definite/probable left ventricular hypertrophy": 2
    }
    return restingECG_map[restingECG_var.get()]



def get_exerciseInducedAngina_var_value():
    return 1 if exerciseInducedAngina_var.get() == "Yes" else 0



def get_peakExerciseSegment_value():
    peakExerciseSegment_map = {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3
    }
    return peakExerciseSegment_map[peakExerciseSegment_var.get()]



def get_thalassemia_value():
    thalassemia_map = {
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversible Defect": 7
    }
    return thalassemia_map[thalassemia_var.get()]

def dec(X_train,y_train,X_test):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
def deep(X_train,y_train,X_test):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    model = Sequential()
    model.add(Dense(13))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1)
    model.fit(x=X_train, y=y_train, epochs=2)
    y_pred = model.predict(X_test)
    return y_pred
def gnb(X_train,y_train,X_test):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
def log_reg(X_train,y_train,X_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
def random_for(X_train,y_train,X_test):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
import pandas as pd
data = pd.read_csv("Cleveland_Synthetic_dataset.csv")
data = data.iloc[:, 1:].values
data = pd.DataFrame(data)
X = data.drop(columns=13)
y = data[13]
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = s_scaler.fit_transform(X_train)
X_test = s_scaler.transform(X_test)

root = tk.Tk()
root.title("Health Data Form")
root.geometry("650x650")



age_label = tk.Label(root, text="Age:",font=("Arial", 10, "bold")).grid(row=0, column=0, padx=50, pady=10,sticky="w")

age_spinbox = tk.Spinbox(root, from_=0, to=100, increment=1, font=("Arial", 10, "bold"))
age_spinbox.grid(row=0, column=1, pady=10,sticky="ew")



sex_var = tk.StringVar(value="Bilinmiyor")
sex_label = tk.Label(root, text="Sex:",font=("Arial", 10, "bold")).grid(row=1, column=0, padx=50, pady=10,sticky="w")
sex_combobox = ttk.Combobox(root, textvariable=sex_var, values=["Male", "Female"],font=("Arial", 10, "bold"))
sex_combobox.grid(row=1, column=1, pady=10,sticky="ew")
sex_combobox.current(0)


chestPain_var = tk.StringVar(value="Bilinmiyor")
chestPain_label = tk.Label(root, text="Chest Pain Type :",font=("Arial", 10, "bold")).grid(row=2, column=0, padx=50, pady=10,sticky="w")
chestPain_combobox = ttk.Combobox(root, textvariable=chestPain_var, values=[  "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],font=("Arial", 10, "bold"))
chestPain_combobox.grid(row=2, column=1, pady=10,sticky="ew")
chestPain_combobox.current(0)




restingBlood_label = tk.Label(root, text="Resting Blood Pressure :",font=("Arial", 10, "bold")).grid(row=3, column=0, padx=50, pady=10,sticky="w")
restingBlood_entry = tk.Entry(root,font=("Arial", 10, "bold"))
restingBlood_entry.grid(row=3, column=1, pady=10,sticky="ew")



chol_label = tk.Label(root, text="Cholesterol :",font=("Arial", 10, "bold")).grid(row=4, column=0, padx=50, pady=10,sticky="w")
chol_entry = tk.Entry(root,font=("Arial", 10, "bold"))
chol_entry.grid(row=4, column=1, pady=10,sticky="ew")



fastingBloodSugar_var = tk.StringVar(value="Bilinmiyor")
fastingBloodSugar_label = tk.Label(root, text="Fasting Blood Sugar > 120 :",font=("Arial", 10, "bold")).grid(row=5, column=0, padx=50, pady=10,sticky="w")
fastingBloodSugar_combobox = ttk.Combobox(root, textvariable=fastingBloodSugar_var, values=["False", "True"],font=("Arial", 10, "bold"))
fastingBloodSugar_combobox.grid(row=5, column=1, pady=10,sticky="ew")
fastingBloodSugar_combobox.current(0)



restingECG_var = tk.StringVar(value="Bilinmiyor")
restingECG_label = tk.Label(root, text="Resting ECG :",font=("Arial", 10, "bold")).grid(row=6, column=0, padx=50, pady=10,sticky="w")
restingECG_combobox = ttk.Combobox(root, textvariable=restingECG_var, values=["Normal","Showing ST-T wave abnormality","Definite/probable left ventricular hypertrophy"],font=("Arial", 10, "bold"))
restingECG_combobox.grid(row=6, column=1, pady=10,sticky="ew")
restingECG_combobox.current(0)



maxHeartRate_label = tk.Label(root, text="Max Heart Rate :",font=("Arial", 10, "bold")).grid(row=7, column=0, padx=50, pady=10,sticky="w")
maxHeartRate_entry = tk.Entry(root,font=("Arial", 10, "bold"))
maxHeartRate_entry.grid(row=7, column=1, pady=10,sticky="ew")



exerciseInducedAngina_var = tk.StringVar(value="Bilinmiyor")
exerciseInducedAngina_label = tk.Label(root, text="Exercise Induced Angina :",font=("Arial", 10, "bold")).grid(row=8, column=0, padx=50, pady=10,sticky="w")
exerciseInducedAngina_combobox = ttk.Combobox(root, textvariable=exerciseInducedAngina_var, values=["No", "Yes"],font=("Arial", 10, "bold"))
exerciseInducedAngina_combobox.grid(row=8, column=1, pady=10,sticky="ew")
exerciseInducedAngina_combobox.current(0)




STDepressionExercise_label = tk.Label(root, text="ST Depression Exercise :",font=("Arial", 10, "bold")).grid(row=9, column=0, padx=50, pady=10,sticky="w")
STDepressionExercise_entry = tk.Entry(root,font=("Arial", 10, "bold"))
STDepressionExercise_entry.grid(row=9, column=1, pady=10,sticky="ew")




peakExerciseSegment_var = tk.StringVar(value="Bilinmiyor")
peakExerciseSegment_label = tk.Label(root, text="Peak Exercise Segment :",font=("Arial", 10, "bold")).grid(row=10, column=0, padx=50, pady=10,sticky="w")
peakExerciseSegment_combobox = ttk.Combobox(root, textvariable=peakExerciseSegment_var, values=["Upsloping","Flat","Downsloping"],font=("Arial", 10, "bold"))
peakExerciseSegment_combobox.grid(row=10, column=1, pady=10,sticky="ew")
peakExerciseSegment_combobox.current(0)




Num_Major_Vessles_Flouro_var = tk.StringVar(value="Bilinmiyor")
Num_Major_Vessles_Flouro_label = tk.Label(root, text="Number of Major Vessels Colored by Fluoroscopy :",font=("Arial", 10, "bold")).grid(row=11, column=0, padx=50, pady=10,sticky="w")
Num_Major_Vessles_Flouro_combobox = ttk.Combobox(root, textvariable=Num_Major_Vessles_Flouro_var, values=["0", "1","2","3"],font=("Arial", 10, "bold"))
Num_Major_Vessles_Flouro_combobox.grid(row=11, column=1, pady=10,sticky="ew")
Num_Major_Vessles_Flouro_combobox.current(0)




thalassemia_var = tk.StringVar(value="Bilinmiyor")
thalassemia_label = tk.Label(root, text="Thalassemia :",font=("Arial", 10, "bold")).grid(row=12, column=0, padx=50, pady=10,sticky="w")
thalassemia_combobox = ttk.Combobox(root, textvariable=thalassemia_var, values=["Normal","Fixed Defect","Reversible Defect"],font=("Arial", 10, "bold"))
thalassemia_combobox.grid(row=12, column=1, pady=10,sticky="ew")
thalassemia_combobox.current(0)




predictionSelect_var = tk.StringVar(value="Bilinmiyor")
predictionSelect_label = tk.Label(root, text="Select PRediction Type :",font=("Arial", 10, "bold")).grid(row=13, column=0, padx=50, pady=10,sticky="w")
predictionSelect_combobox = ttk.Combobox(root, textvariable=predictionSelect_var, values=[  "Logistic Regression", "Deep Learning", "Decision Tree", "Random Forest", "Gaussian NB"],font=("Arial", 10, "bold"))
predictionSelect_combobox.grid(row=13, column=1, pady=10,sticky="ew")
predictionSelect_combobox.current(0)




def check_prediction():

    input_data = [
        int(age_spinbox.get()),
        get_sex_numeric_value(),
        get_cp_numeric_value(),
        int(restingBlood_entry.get()),
        int(chol_entry.get()),
        get_fastingBloodSugar_value(),
        get_restingECG_numeric_value(),
        int(maxHeartRate_entry.get()),
        get_exerciseInducedAngina_var_value(),
        float(STDepressionExercise_entry.get()),
        get_peakExerciseSegment_value(),
        int(Num_Major_Vessles_Flouro_var.get()),
        get_thalassemia_value()
    ]
    
    input_scaled = s_scaler.transform([input_data])
    select1 = predictionSelect_var.get()  # ComboBox'tan seçilen model

    if select1 == "Gaussian NB":
        prediction = gnb(X_train, y_train, input_scaled)
        if prediction==0:
            result = "Not Sick"
        else:
            result="Sick"
    elif select1 == "Logistic Regression":
        prediction = log_reg(X_train, y_train, input_scaled)
        if prediction==0:
            result = "Not Sick"
        else:
            result="Sick"
    elif select1 == "Random Forest":
        prediction = random_for(X_train, y_train, input_scaled)
        if prediction==0:
            result = "Not Sick"
        else:
            result="Sick"
    elif select1 == "Decision Tree":
        prediction = dec(X_train, y_train, input_scaled)
        if prediction==0:
            result= "Not Sick"
        else:
            result="Sick"
    elif select1 == "Deep Learning":
        prediction = deep(X_train, y_train, input_scaled)
        if prediction==0:
            result= "Not Sick"
        else:
            result="Sick"
    else:
        prediction = None


    result_label.config(text=f"Result: {result}")
check_button = tk.Button(root, text="CHECK", font=("Arial", 10, "bold"), bg="#888888", fg="white", command=check_prediction)
check_button.grid(row=14, column=1, pady=10, sticky="ew")
result_label = tk.Label(root, text="Result: ",font=("Arial", 10, "bold"))
result_label.grid(row=14, column=0, padx=50, pady=10,sticky="w")

root.mainloop()


