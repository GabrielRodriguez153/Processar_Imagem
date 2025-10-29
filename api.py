from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
import asyncio
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

app = Flask(__name__)

print("Carregando modelo...")
try:
    modelo = tf.keras.models.load_model("modelo_cnn_finetuned.keras")
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    modelo = None

def analisar_planta(img_bytes, classes={0: "Planta saudável", 1: "Planta doente"}):
    if modelo is None:
        return "Erro: Modelo não carregado"
    
    try:
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predicao = modelo.predict(img_array)

        if predicao.shape[-1] == 1:
            indice = 1 if predicao[0][0] > 0.5 else 0
        else:
            indice = np.argmax(predicao[0])
        
        return classes[indice]
    except Exception as e:
        return f"Erro no processamento: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhuma imagem fornecida'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhuma imagem selecionada'}), 400

    try:
        img_bytes = file.read()
        
        resultado = analisar_planta(img_bytes)
        
        return jsonify({
            'status': 'sucesso', 
            'resultado': resultado,
            'modelo_carregado': modelo is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    status = {
        'status': 'online', 
        'message': 'API está funcionando',
        'modelo_carregado': modelo is not None,
        'timestamp': np.datetime64('now').astype(str)
    }
    return jsonify(status)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Plant Analysis API',
        'version': '1.0',
        'endpoints': {
            'health': '/health (GET)',
            'predict': '/predict (POST)'
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)