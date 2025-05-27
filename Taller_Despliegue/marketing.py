import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import Lasso
import pickle
import os

from flask import Flask, jsonify, request, abort  #imprtar librerías de flask, request para parametros abort par errores




#definimos nuestra APPI 
app = Flask(__name__)
app.config["DEBUG"] = True  #paraue nos de mñas informacion cuando estems haciendo peticiones a la API

print('El fichero se encuentra en:')
print(__file__)

#aqui hacemos que se cambiar a ese directori
print('Cambiando el directorio de trabajo a:')
os.chdir(os.path.dirname(__file__))  #cambiar el directorio de trabajo al de este archivo, para que funcione en cualquier parte

print(os.path.dirname(__file__))  #ver el directorio de trabajo

#decradores 
# End Point "/"
@app.route('/', methods=['GET'])  #direccionemos todas las peticiones a la raiz
def home():
    return "<h1>My API</h1><p>ésta e una API para predición de ventas en función de inversión en marketing.</p>"


#queremos un end point
@app.route('/v1/predict', methods=['GET'])
def predict():
    
    #cargar el modelo
    #btener los parámetros del request
    #hacer la predicción
    #devolverla 
    model = pickle.load(open('ad_model.pkl','rb'))
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"  #comprobar que tenemos tdos los parámetros 
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])  #hacer la predicción 
   # [[float(tv),float(radio),float(newspaper)]]
    #[pred1]
    return jsonify({'predictions': prediction[0]})  #devlverla , vector de 0, osea la primera predicción
    
    
@app.route('/v1/retrain', methods=['GET'])  #endpoint para ver las métricas del model
def retrain(): # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        pickle.dump(model, open('ad_model.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


app.run()