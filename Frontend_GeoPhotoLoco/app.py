from flask import Flask, render_template, Blueprint, jsonify, request
import os

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

        upload_folder = 'image_uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({'message': 'File successfully uploaded'})

# Register the blueprint with the app
app.register_blueprint(send_data_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
