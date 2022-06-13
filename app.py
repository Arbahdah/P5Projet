import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)
with open('vectorizer.pickle', 'rb') as handle:
	vectorizer = pickle.load(handle)
with open('dimred.pickle', 'rb') as handle:
	svd = pickle.load(handle)
with open('normalize.pickle', 'rb') as handle:
	normalizer = pickle.load(handle)
with open('binarizer.pickle', 'rb') as handle:
	binarizer = pickle.load(handle)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    init_features = [str(x) for x in request.form.values()]
    features = [np.array(init_features)]
    dict_init_features = pd.DataFrame(data=features)
    str_features = normalizer.normalize_corpus(init_features)
    cvect_transform = vectorizer.transform(str_features)
    svd_transform = svd.transform(cvect_transform)
    prediction = model.predict(svd_transform)
    
    output = binarizer.inverse_transform(prediction)
    final_output = ''
    for o in output:
        if o:
            if final_output:
                final_output +=','
                final_output += o
    final_output = final_output.replace('(','').replace(')','')
    

    return render_template('index.html', prediction_text='Tags suggérés{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
