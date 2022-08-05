import pickle
from flask import Flask, render_template, request, redirect, make_response
import pandas as pd
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def pred():
    df = pd.DataFrame(columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
    
    if request.method == "POST":
        Pregnancies = request.form["Pregnancies"]
        Glucose = request.form["Glucose"]
        BloodPressure = request.form["BloodPressure"]
        SkinThickness = request.form["SkinThickness"]
        Insulin = request.form["Insulin"]
        BMI = request.form["BMI"]
        DiabetesPedigreeFunction = request.form["DiabetesPedigreeFunction"]
        Age = request.form["Age"]
    
        df = df.append({'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BloodPressure': BloodPressure, 'SkinThickness': SkinThickness,'Insulin': Insulin, 'BMI': BMI, 'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age}, ignore_index=True)
        print(df)
        cc = sc.transform(df)
        print(cc)
        with open('randomforest_old_method.pkl','rb') as file:
            rf = pickle.load(file)

        prediction = rf.predict(cc)
        df['prediction'] = prediction
        print(df['prediction'][0])
        return render_template('result.html', pred=str(df['prediction'][0]), name='name')
    
    

if __name__ == "__main__":
    app.debug=True
    app.run()
    