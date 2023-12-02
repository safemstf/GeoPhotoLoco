from flask import Flask, render_template, Blueprint, jsonify, request
from threading import Thread
import os, subprocess, time, json
last_processed_data = None
app = Flask(__name__, static_folder='static')

# Your existing routes
@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/output')
def output():
    return render_template('output.html')

# Blueprint for sending data
send_data_blueprint = Blueprint('send_data', __name__)

@send_data_blueprint.route('/send_data', methods=['POST'], endpoint='send_data')
def send_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        upload_folder = '../Backend/TestImages'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({'message': 'File successfully uploaded'})

# Register the blueprint with the app
app.register_blueprint(send_data_blueprint)


get_data_blueprint = Blueprint('get_data', __name__)

@get_data_blueprint.route('/get_data', methods=['GET'], endpoint='get_data')
def get_data():
    # Your logic to retrieve and return data
    data = {'message': last_processed_data}
    return jsonify(data)

app.register_blueprint(get_data_blueprint)

# Function to run TestModel.py as a subprocess
def run_model():
    global last_processed_data  # Use the global variable to store the last processed data
    while True:
        try:
            result = subprocess.run(["python3", "/home/demir/Documents/GitHub/GeoPhotoLoco/Backend/TestModel.py"], stdout=subprocess.PIPE, text=True)
            captured_output = result.stdout

            # Check if the new data is different from the last processed data
            if captured_output != last_processed_data:
                # Parse the JSON string into a dictionary
                processed_data = json.loads(captured_output)

                # Update the last processed data dictionary
                last_processed_data = {
                    'processed_data': processed_data,
                    'timestamp': time.time()  # You can add more fields if needed
                }
                print("New Data:", last_processed_data)

            time.sleep(2)
            
        except Exception as e:
            print(f"Error running TestModel.py: {e}")


if __name__ == '__main__':
    test_model_thread = Thread(target=run_model)
    test_model_thread.start()
    # Start the Flask app
    app.run(debug=True, use_reloader=False)
