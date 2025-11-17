from flask import Flask, render_template, request
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from keras.models import load_model
import os

app = Flask(__name__)
modelo_guardado = load_model('./models/modelo.h5')
nombres_de_clasificaciones = ['Polera', 'Pantalon', 'Pullover', 'Vestido', 'Abrigo', 
                               'Sandalia', 'Camisa', 'Zapatilla', 'Cartera', 'Bota']


if not os.path.exists('static'):
    os.makedirs('static')

def pixel(img):
    img_Ga = cv.GaussianBlur(img, (7, 7), 0)
    img_g = cv.cvtColor(img_Ga, cv.COLOR_BGR2GRAY)
    img_r = cv.resize(img_g, (28, 28), interpolation=cv.INTER_NEAREST)
    img_i = cv.bitwise_not(img_r)
    return img_i

def plot_image(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(
        nombres_de_clasificaciones[predicted_label],
        100 * np.max(predictions_array),
        nombres_de_clasificaciones[true_label]), color=color)

def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(len(nombres_de_clasificaciones)), nombres_de_clasificaciones, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(len(predictions_array)), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def predict_image(img):
    img_i = pixel(img)
    img_i = np.expand_dims(img_i, axis=0)
    img_i = img_i / 255.0  # Normalize pixel values
    predictions = modelo_guardado.predict(img_i)
    resultado = np.argmax(predictions[0])
    return predictions[0], resultado

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv.imdecode(img_bytes, cv.IMREAD_COLOR)
        
        
        predictions, resultado = predict_image(img)
        result_label = nombres_de_clasificaciones[resultado]
        
       
        plt.figure(figsize=(12, 6))
        
       
        plt.subplot(1, 2, 1)
        plot_image(predictions, resultado, pixel(img))
        
       
        plt.subplot(1, 2, 2)
        plot_value_array(predictions, resultado)
        
        plt.tight_layout()
        
        
        static_path = os.path.join(os.path.dirname(__file__), 'static', 'predicted_image.png')
        plt.savefig(static_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        print(f"Imagen guardada en: {static_path}")  
        
      
        return render_template('resultado.html', result_label=result_label)
    
    return render_template('resultado.html', result_label='Error: No se recibió ningún archivo')

if __name__ == '__main__':
    app.run(debug=True)