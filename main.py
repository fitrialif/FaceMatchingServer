from sys import platform as _platform
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

from keras import backend as K
K.set_image_data_format('channels_first')
import tensorflow as tf
from fr_utils import *
from inception_blocks import *
#from LoadModel import *
FRmodel = True

def verify(image_path_1, image_path_2, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path_1 -- path to an image
    image_path_2 -- path to an image
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the 2 image_path_ and _2
    same_face -- True, if the faces match. False otherwise.
    """

    ### START CODE HERE ###

    # # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    # encoding_1 = img_to_encoding(image_path_1, model)
    # encoding_2 = img_to_encoding(image_path_2, model)
    #
    # # Step 2: Compute distance with identity's image (≈ 1 line)
    # dist = np.linalg.norm(encoding_1 - encoding_2)
    #
    # # Step 3: Match faces if dist < 0.7, else don't match
    # if dist < 0.7:
    #     same_face = True
    # else:
    #     same_face = False

    ### END CODE HERE ###

    #return dist, same_face
    return True

PWD = '/var/www/html/flaskapp'
MODEL_PATH = PWD + '/data/output_graph.pb'
LABELS_PATH = PWD + '/data/output_labels.txt'

# HTTP API
app = Flask(__name__)

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/photo-recognize', methods=['POST'])
def photoRecognize():
    print("coolos")
    #print(request)
    for key in request.form.keys():
        print(key)
    jsdata_1 = request.form['canvas_data_1']
    jsdata_2 = request.form['canvas_data_2']

    # convert string of image data to uint8
    nparr_1 = np.fromstring(jsdata_1, np.uint8)
    nparr_2 = np.fromstring(jsdata_2, np.uint8)

    # decode image
    img_1 = cv2.imdecode(nparr_1, cv2.IMREAD_COLOR)
    img_2 = cv2.imdecode(nparr_2, cv2.IMREAD_COLOR)

    answer = verify(img_1,img_2, FRmodel)

    ## TODO: do some logic with answer here
    return jsonify(status='OK', results=answer)

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()