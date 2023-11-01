from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
# import matplotlib.pyplot as plt

app = Flask(__name__)
model_predict = load_model('custom_model95.h5')
model_predict.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')
def serve_static(filename):
    return send_from_directory('static', filename)

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
# def read_image(filename):

#     img = load_img(filename, target_size=(256,256))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x

def predict_class(image_path):
    # Read the image from the provided path
    img = image.load_img(image_path, color_mode="rgb", target_size=(256, 256), interpolation="nearest")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make a prediction
    classes = model_predict.predict(img_array, batch_size=10)

    # Get the predicted class and confidence
    max_confidence = np.max(classes[0])
    predicted_class_index = np.argmax(classes[0])

    classes_list = ['Bacterialblight', 'Brownspot', 'Leafsmut']
    predicted_class = classes_list[predicted_class_index]

    return predicted_class, max_confidence
  
@app.route('/predict',methods=['GET','POST'])


  
  
  
# image_path = "/Users/vashu/Downloads/Bennett_3rd year/CSET 225 AI /project/new custom model try/rice_leaf_disease/test/Bacterialblight/BACTERAILBLIGHT5_177.JPG"
# predicted_class, confidence = predict_class(image_path)

# Display the result


def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            # img = read_image(file_path)
            pclass, confidence = predict_class(file_path)
            # class_prediction=model.predict(img) 
            # classes_x=np.argmax(class_prediction,axis=1)
            # if classes_x == 0:
            #   fruit = "Bacterial Blight"
            # elif classes_x == 1:
            #   fruit = "Brownspot"
            # else:
            #   fruit = "Leafsmut"
            return render_template('predict.html', fruit = pclass,prob=confidence, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)