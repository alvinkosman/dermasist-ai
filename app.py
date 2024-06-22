from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from os import listdir
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os
import matplotlib.pyplot as plt

# Set the matplotlib backend to 'Agg' before importing anything that might use it
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
model = load_model('C:/Users/Unknown/Documents/UAS AI/skin-inceptionv3')
NAMA_KELAS = sorted(listdir(r'D:\ISIC_2019\ISIC_2019_Test_Input'))

BARIS = 256
KOLOM = 256

def generate_prediction_plot(image_path):
    img = load_img(image_path, target_size=(BARIS, KOLOM))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Perform prediction
    predictions = model.predict(img_array)[0]
    best_score_index = np.argmax(predictions)
    classification = NAMA_KELAS[best_score_index]

    # Save plot without displaying it
    plt.imshow(img)
    plt.savefig('./static/plot.png')  # Save the plot to a static file
    plt.close()  # Close the plot to release resources

    return classification, predictions[best_score_index]

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    classification, accuracy = generate_prediction_plot(image_path)
    if classification != "Normal":
        classification = "penyakit kanker kulit berjenis " + classification
    else:
        classification = "kulit " + classification
        
    os.remove(image_path)  # Remove the uploaded image

    return render_template('index.html', prediction=classification, accuracy=f"{100 * accuracy:.2f}%", image_path='/static/plot.png')

@app.route('/manifest.json')
def serve_manifest():
    return send_file('/stamanifest.json', mimetype='application/manifest+json')

@app.route('/service-worker.js')
def serve_sw():
    return send_file('/static/service-worker.js', mimetype='application/javascript')

if __name__ == '__main__':
    app.run(port=3000)
