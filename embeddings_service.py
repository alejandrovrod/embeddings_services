from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo de embeddings (se descarga automáticamente)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "all-MiniLM-L6-v2"})

@app.route('/embed', methods=['POST'])
def embed():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Generar embedding
        embedding = model.encode(text).tolist()
        
        return jsonify({
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": "all-MiniLM-L6-v2"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/embed_batch', methods=['POST'])
def embed_batch():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
        # Generar embeddings en lote
        embeddings = model.encode(texts).tolist()
        
        return jsonify({
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "model": "all-MiniLM-L6-v2"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 
