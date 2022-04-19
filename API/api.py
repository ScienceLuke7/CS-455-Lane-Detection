import json
import pickle
from flask import Flask, jsonify, request, send_file
# import neural_network.model_evaluator

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify('lane-detection')

@app.route("/postRawImage", methods=["POST"])
def getProcessedImage():
    image = pickle.loads(request.data)
    print(image)
    return

@app.route("/postRawVideo", methods=["POST"])
def getProcessedVideo():
    video = request.data
    print(video)
    return




#REMOVE LATER --------------------------------
app.debug = True
#^REMOVE LATER --------------------------------
if __name__ == '__main__':
    app.run()