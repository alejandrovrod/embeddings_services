from flask import Flask, request, jsonify
import os
import numpy as np

app = Flask(__name__)

# Importar sentence-transformers solo cuando sea necesario
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "loading",
        "model": "all-MiniLM-L6-v2",
        "model_loaded": MODEL_LOADED
    })

@app.route('/embed', methods=['POST'])
def embed():
    try:
        if not MODEL_LOADED:
            return jsonify({"error": "Model not loaded yet"}), 503
        
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
        if not MODEL_LOADED:
            return jsonify({"error": "Model not loaded yet"}), 503
        
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
