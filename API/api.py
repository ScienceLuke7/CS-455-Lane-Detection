import json
import pickle
import cv2
import glob
from flask import Flask, jsonify, request, send_file
# import neural_network.model_evaluator

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify('lane-detection')

@app.route("/postRawImage", methods=["POST"])
def getProcessedImage():
    image = request.data
    # print(image)
    return send_file('../_datasets/train_set/clips/0313-1/120/10.jpg', mimetype='image/jpeg')

@app.route("/postRawVideo", methods=["POST"])
def getProcessedVideo():
    video = request.data
    # print(video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output_video.mp4', fourcc, 20, (1280,720), isColor=True)
    
    for filename in glob.glob('../_datasets/train_set/clips/0313-1/120/*.jpg'):
        img = cv2.imread(filename)
        out.write(img)

    out.release()
    return send_file('./output_video.mp4', mimetype='video/mp4')




#REMOVE LATER --------------------------------
app.debug = True
#^REMOVE LATER --------------------------------
if __name__ == '__main__':
    app.run()