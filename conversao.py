from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
import asyncio

app = Flask(__name__)

# Carregar o modelo uma vez ao iniciar a API
modelo = tf.keras.models.load_model("modelo_cnn_finetuned.keras")

def analisar_planta(img_bytes, classes={0: "Planta saudável", 1: "Planta doente"}):
    # Carregar imagem dos bytes
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Fazer previsão
    predicao = modelo.predict(img_array)

    if predicao.shape[-1] == 1:
        indice = 1 if predicao[0][0] > 0.5 else 0
    else:
        indice = np.argmax(predicao[0])
    
    return classes[indice]

@app.route('/predict', methods=['POST'])
async def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhuma imagem fornecida'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhuma imagem selecionada'}), 400

    try:
        # Processamento assíncrono
        img_bytes = file.read()
        
        # Executar em thread separada para não bloquear
        loop = asyncio.get_event_loop()
        resultado = await loop.run_in_executor(None, analisar_planta, img_bytes)
        
        return jsonify({'status': 'sucesso', 'resultado': resultado})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online', 'message': 'API está funcionando'})

if __name__ == '__main__':
    # Para desenvolvimento apenas
    app.run(host='0.0.0.0', port=5000, debug=False)
