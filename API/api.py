from flask import Flask
from endpoints.send_data import send_data_blueprint


app = Flask(__name__)
app.register_blueprint(send_data_blueprint)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)