import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create a Flask application
app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define a route for the default URL, which loads the index page
@app.route('/')
def index():
	return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
	# Get JSON request
	int_features = [int(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)

	output = round(prediction[0], 2)

	return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

if __name__ == '__main__':
	app.run(debug=True)
