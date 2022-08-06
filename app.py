import pickle
from flask import Flask, render_template, request, redirect, make_response
import pandas as pd
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()

data = pd.read_csv('./dataset/Diabetes_Prediction.csv')

X = data.iloc[:, :-1].values
sc.fit(X)

app = Flask(__name__)

@app.route('/')
def index():
    print('yes')
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
        
        cc = sc.transform(df)
        
        with open('./model/randomforest_old_method.pkl','rb') as file:
            rf = pickle.load(file)

        prediction = rf.predict(cc)
        df['prediction'] = prediction
        # df['prediction'] = df['prediction'].replace(['0','1'],['Tidak ada penyakit Diabetes','Kemungkinan Terkena Diabetes. Silahkan Hubungi dokter untuk memastikan'])
        return render_template('result.html', pred=str(df['prediction'][0]), name='name')
    
    

if __name__ == "__main__":
    app.debug=True
    app.run()
    