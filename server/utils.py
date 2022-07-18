import json
import os
import pickle
import numpy as np

__model = None
__data_columns = None

def get_stroke_likelihood(age, hypertension, heart_disease, avg_glucose_level, bmi, gender, married, work_type, residence, smoking):
    print('get stroke likelihood')
    x = np.zeros(len(__data_columns))
    x[0] = age
    x[1] = hypertension
    x[2] = heart_disease
    x[3] = avg_glucose_level
    x[4] = bmi
    x[5] = age/bmi
    x[6] = age*bmi
    if gender == 1:
        x[8] = 1
    else:
        x[7] = 1
    if married == 1:
        x[10] = 1
    else:
        x[9]= 1
    if work_type == 0:
        x[11] = 1
    elif work_type == 1:
        x[12] = 1
    elif work_type == 2:
        x[13] = 1
    elif work_type == 3:
        x[14] = 1
    elif work_type == 4:
        x[15] = 1
    else:
        print("Lafda")
    if residence == 1:
        x[16] = 1
    else:
        x[17] = 1
    if smoking == 0:
        x[18] = 1
    elif smoking == 1:
        x[19] = 1
    elif smoking == 2:
        x[20] = 1
    else:
        x[21] = 1
    return np.round(__model.predict_proba([x])[0][1]*100, 2)

def load_saved_artifacts():
    print("Loading Saved Artifacts...")
    global __model
    global __data_columns

    path = os.getcwd()
    with open(path + "//artifacts//columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
    
    with open(path + "//artifacts//Stroke_Prediction.pickle", "rb") as f:
        __model = pickle.load(f)
    print("loading saved artifacts__done")

if __name__=="__main__":
    load_saved_artifacts()
    print("Load successfully")
