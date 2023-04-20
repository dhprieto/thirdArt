from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import pickle
import joblib
import io
import csv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected model
    selected_model = request.form['model']

    # Get the uploaded file
    file = request.files['file']

    # Load the model
    if selected_model == 'mlp_full':
        with open('models/mlp_full_plasmAnt.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'mlp_met':
        with open('models/mlp_met_plasmAnt.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'rf_full':
        with open('models/rf_full_plasmAnt.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'rf_met':
        with open('models/rf_met_plasmAnt.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        return 'Error: Invalid model selection'

    # Make predictions
    df = pd.read_csv(file)
    predictions = model.predict(df)

    # Create a new CSV file with predictions
    output = io.StringIO()
    writer = csv.writer(output)
    for prediction in predictions:
        writer.writerows([prediction])
    output.seek(0)

    # Send the CSV file to the client
    return send_file(
        io.BytesIO(output.read().encode('utf-8')),
        mimetype='text/csv',
        attachment_filename='predictions.csv',
        as_attachment=True
    )

@app.route('/show_data',  methods=("POST", "GET"))
def showData():
    # Get the selected model
    selected_model = request.form['model']

    # Get the uploaded file
    file = request.files['file']

    # Load the model
    if selected_model == 'mlp_full':
        with open('models/mlp_full_plasmAnt.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'mlp_met':
        with open('models/mlp_met_plasmAnt.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'rf_full':
        with open('models/rf_full_plasmAnt.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selected_model == 'rf_met':
        with open('models/rf_met_plasmAnt.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        return 'Error: Invalid model selection'

    # Make predictions
    df = pd.read_csv(file)
    predictions = model.predict(df)

    # Convert pandas dataframe to html table flask
    df_html = predictions.to_html()
    return render_template('index.html', data=df_html)


if __name__ == '__main__':
    app.run(debug=True)
