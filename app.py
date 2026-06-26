from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64, cv2, os

app = Flask(__name__)
CORS(app, origins='*', allow_headers=['Content-Type', 'Authorization'])

facial_model = tf.keras.models.load_model('facial_model.keras')
ct_model = tf.keras.models.load_model('ct_model.keras')
print('Models loaded!')

def decode_img(b64):
    b64 = b64.split(',')[1] if ',' in b64 else b64
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (224, 224))

@app.route('/facial', methods=['POST', 'OPTIONS'])
def facial():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response
    try:
        img = decode_img(request.json['image'])
        t = img.astype(np.float32) / 255.0
        t = np.expand_dims(t, 0)
        p = facial_model.predict(t, verbose=0)[0]
        response = jsonify({'stroke_prob': float(p[1]), 'no_stroke_prob': float(p[0])})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ct', methods=['POST', 'OPTIONS'])
def ct():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response
    try:
        img = decode_img(request.json['image'])
        t = img.astype(np.float32) / 255.0
        t = np.expand_dims(t, 0)
        p = ct_model.predict(t, verbose=0)[0]
        response = jsonify({'hemorrhagic_prob': float(p[0]), 'ischemic_prob': float(p[1])})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    response = jsonify({'status': 'ok'})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
