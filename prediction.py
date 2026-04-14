import pickle
import numpy as np

def predict(data):
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    data = np.array(data).reshape(1, -1)
    return model.predict(data)
