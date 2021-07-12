import flask
from flask import request, Flask, render_template
import pickle
from pickle import load
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')


filename = 'model.pkl'
clf = load(open(filename, 'rb'))



@app.route('/predict', methods=['POST'])
def predict():
	if request.method == "POST":
		pregnancies = int(request.form['pregnancies'])
		glucose = int(request.form['glucose'])
		blood_pressure = int(request.form['blood_pressure'])
		skin_thickness = int(request.form['skin_thickness'])
		insulin = int(request.form['insulin'])
		bmi = float(request.form['bmi'])
		diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
		age = int(request.form['age'])

		data = np.array([[pregnancies,glucose,blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function,age]])
		my_prediction = clf.predict(data)
		return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)


