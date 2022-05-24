from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import json
import requests
import os
#Load model with pretrained weights (from keras.io)
"""
NOTE: the following line downloads the model from keras each time the program is run.
In practice, you should load the pretrained model from your project directory.
"""
model = ResNet50(weights='imagenet')
img_path = 'img.jpg'
request = {}
#Read Rising Cloud Task input
with open("request.json", 'r') as f:
    request = json.load(f) 
    f.close()
    ret = []
#Downloads file from “img_url” field in request.json to img.jpg locally
with open(img_path, 'wb+') as f:
    response = requests.get(request["img_url"], stream=True)
    if not response.ok:
        print (response)
    for block in response.iter_content(1024):
        if not block:
            break
        f.write(block)
#runs the image through the classifier
with open(img_path, "rb") as f:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]
    ret = [p[1] for p in top_preds]
response = {"guesses": ret}
#puts top 3 guesses in response.json
with open("response.json", 'w+') as f:
    json.dump(response, f)
#Cleanup: deletes img.jpg
os.remove(img_path)