import json
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify('hello wooorld')

@app.route("/getSample", methods=["GET"])
def getSample():
    sampleObject = {
        'stuff': 'asdas'
    }
    return json.dumps(sampleObject.__dict__)

@app.route("/postSample", methods=["POST"])
def getWatchList():
    # doesnt do anything
    return #json.dumps()




#REMOVE LATER --------------------------------
app.debug = True
#^REMOVE LATER --------------------------------
if __name__ == '__main__':
    app.run()