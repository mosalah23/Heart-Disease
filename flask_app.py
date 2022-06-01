import json

from flask import Flask,jsonify,request,render_template
import sklearn.neighbors
import pickle
import numpy as np
import pandas as pd 

with open(f'clf4.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='templates')

@app.route('/home')
def home():
  return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

  if request.method == 'GET':
    return(render_template('home.html'))

  if request.method == 'POST':
      
   Smoking = request.form['Smoking']
   Stroke = request.form['Stroke']
   PhysicalHealth = request.form['PhysicalHealth']
   MentalHealth = request.form['MentalHealth']
   DiffWalking = request.form['DiffWalking']
   Sex = request.form['Sex']
   AgeCategory = request.form['AgeCategory']
   Diabetic = request.form['Diabetic']
   GenHealth = request.form['GenHealth']
   SleepTime = request.form['SleepTime']
   KidneyDisease = request.form['KidneyDisease']
   SkinCancer = request.form['SkinCancer'] 
   input_variables = pd.DataFrame([[Smoking, Stroke, PhysicalHealth,MentalHealth,DiffWalking,Sex,AgeCategory,Diabetic,GenHealth,SleepTime,KidneyDisease,SkinCancer]],
    columns=['Smoking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex','AgeCategory','Diabetic','GenHealth','SleepTime','KidneyDisease','SkinCancer'],dtype=float)
   
  prediction=model.predict(input_variables)

  if prediction == 0: prediction ='NO HEART DISEASE !Disclaimer, this is not medicinally precise!'
  else: prediction= 'MODEL SAYS HEART DISEASE !Disclaimer, this is not medicinally precise!'

   
  return render_template('home.html',original_input={'Smoking':Smoking,
                                                     'Stroke':Stroke,
                                                     'PhysicalHealth':PhysicalHealth,
                                                     'MentalHealth':MentalHealth,
                                                     'DiffWalking':DiffWalking,
                                                     'Sex':Sex,
                                                     'AgeCategory':AgeCategory,
                                                     'Diabetic':Diabetic,
                                                     'GenHealth':GenHealth,
                                                     'SleepTime':SleepTime,
                                                     'KidneyDisease':KidneyDisease,
                                                     'SkinCancer':SkinCancer},
                                     result=prediction,
                                     )





if(__name__=='__main__'):
    app.run(debug=True)