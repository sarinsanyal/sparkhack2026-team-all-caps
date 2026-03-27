from flask import Flask, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app) # Allows your HTML to talk to this script

@app.route('/run-predict', methods=['POST'])
def run_predict():
    try:
        # This executes your batch file
        subprocess.Popen(["run_servers.bat"], shell=True)
        return jsonify({"status": "success", "message": "Servers starting..."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(port=5000)