from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('modelo_final.h5')

# Definir las clases
class_labels = ['ectomicorrizas', 'endomicorrizas', 'ericoide']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Procesar la imagen subida
        file = request.files['image']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Preprocesar la imagen
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Realizar predicción
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_labels[predicted_class]
            predicted_probability = np.max(predictions)

            return render_template("result.html", label=predicted_label, probability=predicted_probability)

    return render_template("index.html")

@app.route('/graficos')
def graficos():
    return render_template('graficos.html')
@app.route('/prueba')
def prueba():
    return render_template('prueba.html')


if __name__ == "__main__":
    app.run(debug=True)
