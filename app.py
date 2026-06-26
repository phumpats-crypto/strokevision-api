from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64, cv2, os

app = Flask(__name__)
CORS(app)

# Load models
print('Loading models...')
facial_model = tf.keras.models.load_model('facial_model.keras')
ct_model = tf.keras.models.load_model('ct_model.keras')
print('Models loaded!')

def decode_img(b64):
    b64 = b64.split(',')[1] if ',' in b64 else b64
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (224, 224))

@app.route('/facial', methods=['POST'])
def facial():
    try:
        img = decode_img(request.json['image'])
        t   = img.astype(np.float32) / 255.0
        t   = np.expand_dims(t, 0)
        p   = facial_model.predict(t, verbose=0)[0]
        return jsonify({'stroke_prob': float(p[1]), 'no_stroke_prob': float(p[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ct', methods=['POST'])
def ct():
    try:
        img = decode_img(request.json['image'])
        t   = img.astype(np.float32) / 255.0
        t   = np.expand_dims(t, 0)
        p   = ct_model.predict(t, verbose=0)[0]
        return jsonify({'hemorrhagic_prob': float(p[0]), 'ischemic_prob': float(p[1])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
