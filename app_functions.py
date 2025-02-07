import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename

# Function to load Keras model
def get_model(path):
    try:
        model = load_model(path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        raise

# Function to predict using image model (pneumonia model in this case)
def pred(path):
    try:
        # Load and preprocess the image
        data = load_img(path, target_size=(224, 224))
        data = np.asarray(data).reshape((1, 224, 224, 3))  # Shape: (1, 224, 224, 3)
        data = data / 255.0  # Normalize image

        # Load pneumonia model and make prediction
        pneumonia_model = get_model('./website/app_models/pneumonia_model.h5')
        predicted = np.round(pneumonia_model.predict(data)[0])[0]
        return predicted
    except Exception as e:
        print(f"Error in image prediction: {e}")
        raise

# Function to make predictions for other diseases based on tabular data
def ValuePredictor(to_predict_list):
    # Validate length of input list
    expected_lengths = [15, 10, 11, 9, 8]
    if len(to_predict_list) not in expected_lengths:
        raise ValueError(f"Expected input of length in {expected_lengths}, but got {len(to_predict_list)}.")
    
    try:
        # Determine which model to use based on input length
        if len(to_predict_list) == 15:
            page = 'kidney'
            with open('./website/app_models/kidney_model.pkl', 'rb') as f:
                kidney_model = pickle.load(f)
            pred = kidney_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
        
        elif len(to_predict_list) == 10:
            page = 'liver'
            with open('./website/app_models/liver_model.pkl', 'rb') as f:
                liver_model = pickle.load(f)
            pred = liver_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
        
        elif len(to_predict_list) == 11:
            page = 'heart'
            with open('./website/app_models/heart_model.pkl', 'rb') as f:
                heart_model = pickle.load(f)
            pred = heart_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
        
        elif len(to_predict_list) == 9:
            page = 'stroke'
            with open('./website/app_models/avc_scaler.pkl', 'rb') as f:
                stroke_scaler = pickle.load(f)
            l1 = np.array(to_predict_list[2:]).reshape((-1, len(to_predict_list[2:]))).tolist()[0]
            l2 = stroke_scaler.transform(np.array(to_predict_list[0:2]).reshape((-1, 2))).tolist()[0]
            l = l2 + l1
            with open('./website/app_models/avc_model.pkl', 'rb') as f:
                stroke_model = pickle.load(f)
            pred = stroke_model.predict(np.array(l).reshape(-1, len(l)))
        
        elif len(to_predict_list) == 8:
            page = 'diabetes'
            with open('./website/app_models/diabete_model.pkl', 'rb') as f:
                diabete_model = pickle.load(f)
            pred = diabete_model.predict(np.array(to_predict_list).reshape((-1, 8)))
            print(pred[0], page)
        
        else:
            raise ValueError(f"Unsupported input size: {len(to_predict_list)}")
        
        return pred[0], page
    
    except Exception as e:
        print(f"Error in prediction for {page}: {e}")
        raise
