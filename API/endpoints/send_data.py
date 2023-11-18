from flask import Blueprint, jsonify, request
import os

send_data_blueprint = Blueprint('send_data', __name__)

@send_data_blueprint.route('/send_data', methods=['POST'])
def send_data():
    # Check if the POST request has the file part
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the file to a desired location on your computer
        upload_folder = 'image_uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({'message': 'File successfully uploaded'})