import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, jsonify, request
import os
from flask_cors import CORS
import pandas as pd
model = load_model('my_model.h5')

classes = ['Healthy', 'Leaf_smut', 'Red_rot', 'Sugarcane_mosaic_virus']

disease_details = {
    "Leaf_smut": {
        "symptoms": [
            "Yellowing of leaves.",
            "Drying of leaves.",
            "Constriction of internodes."
        ],
        "management": [
            "Use disease-free seeds and cuttings.",
            "Apply Bavistin or Tilt at the time of earthing up.",
            "Spray Carbendazim or Copper oxychloride."
        ],
        "spread_details": [
            "This disease can spread and the best ways to prevent it is to blah blah blah"
            
        ]
    },
    "Red_rot": {
        "symptoms": [
            "Presence of black sooty mass inside the infected shoots.",
            "Formation of brownish-black whip-like structures from the growing points.",
            "Degeneration of growing points."
        ],
        "management": [
            "Removal and destruction of infected plant debris.",
            "Treat the seeds with hot water at 50-52°C for 20-30 minutes.",
            "Spray Carbendazim or Copper oxychloride."
        ],
        "spread_details": [
            "This disease can spread and the best ways to prevent it is to blah blah blah"
            
        ]
    },
    "Sugarcane_mosaic_virus": {
        "symptoms": [
            "Formation of hard, dark, sclerotial bodies inside the infected flowers.",
            "Matured sclerotia protruding out from the affected flowers.",
            "White or yellowish pus oozing out from the affected flowers."
        ],
        "management": [
            "Avoid planting of sugarcane after ratoon crop of maize.",
            "Removal of infected plant parts.",
            "Spray Carbendazim or Copper oxychloride."
        ],
        "spread_details": [
            "This disease can spread and the best ways to prevent it is to blah blah blah"
            
        ]
    }
}


app = Flask(__name__)
CORS(app)


# Define a route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was sent
    if 'image' not in request.files:
        return jsonify({'error': 'No image file was sent.'})
       
    image = request.files['image'].read()
    npimg = np.fromstring(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Resize the image to (300, 300)
    img_size = (256, 256)
    img = cv2.resize(img, img_size)

    # Convert the image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Expand the image dimensions to match the input shape of the model
    img_input = np.expand_dims(img, axis=0)

    # Make prediction
    predicted_probs = model.predict(img_input)
    predicted_label = np.argmax(predicted_probs)
    predicted_class = classes[predicted_label]

    details = disease_details.get(predicted_class, None)
    # Return the predicted class as a JSON response
    if details:
        
        return jsonify({'prediction': predicted_class,'symptoms': details['symptoms'],
            'management':details['management'], 'spread_details': details['spread_details']})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8100, debug=True)
