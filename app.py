import pickle
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np

# Ajuste das pastas de template e assets
app = Flask(__name__, template_folder='template', static_folder='template/assets')

# Import do modelo já treinado e salvo (essa parte foi feita no jupyter notebook)
modelo_pipeline = pickle.load(open('./models/models.pkl', 'rb'))


# Pagina principal
@app.route('/')
def home():
    return render_template("homepage.html")

# Pagina Forms que é preenchido pelo usuario
@app.route('/dados_flor')
def dados_flor():
    return render_template("form.html")


def get_data():
    sepal_length = request.form.get('sepal_length')
    sepal_width = request.form.get('sepal_width')
    petal_length = request.form.get('petal_length')
    petal_width = request.form.get('petal_width')


    d_dict = {'sepal_length': [sepal_length], 'sepal_width': [sepal_width], 'petal_length': [petal_length],
              'petal_width': [petal_width]}

    return pd.DataFrame.from_dict(d_dict, orient='columns')

## Renderiza o resultado predito pelo modelo ML na Webpage
@app.route('/send', methods=['POST'])
def show_data():

    try:
        df = get_data()
        df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

        # Faz a predição com os dados digitados pelo usuario
        prediction = modelo_pipeline.predict(df)

        if prediction == 'Iris-virginica':
            outcome = 'OPAAAA é uma Iris-virginica!'
            imagem = 'Iris_virginica.jpg'
        elif prediction == 'Iris-setosa':
            outcome = 'Quem diria, é uma Iris-setosa!'
            imagem = 'Iris_setosa.jpg'
        else:
            outcome = 'Eu jurava que não era uma Iris-versicolor!'
            imagem = 'Iris_versicolor.jpg'

    except ValueError as e:
        outcome = 'OPAAAA você digitou coisa errada! '+str(e).split('\n')[-1].strip()
        imagem = 'flor.png'
    
    return render_template('result.html', tables=[df.to_html(classes='data', header=True, col_space=10)],
                           result=outcome, imagem=imagem)


# retorna o a predição formatada em JSON para uma solicitação HTTP
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)

    try:
         prediction = modelo_pipeline.predict([np.array(list(data.values()))])
         output = {
        'status': 200,
        'prediction': prediction[0]
        }
         
    except ValueError as e:
        output = {
        'status': 500,
        'prediction': str(e).split('\n')[-1].strip()
        }
   
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
