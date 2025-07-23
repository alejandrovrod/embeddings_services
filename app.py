from flask import Flask, request, jsonify
import os
import hashlib

app = Flask(__name__)

def simple_embedding(text):
    """Genera un embedding simple basado en hash del texto"""
    text_bytes = text.encode('utf-8')
    hash_object = hashlib.sha256(text_bytes)
    hash_hex = hash_object.hexdigest()
    
    embedding = []
    for i in range(0, len(hash_hex), 2):
        if len(embedding) >= 384:
            break
        embedding.append(int(hash_hex[i:i+2], 16) / 255.0)
    
    while len(embedding) < 384:
        embedding.append(0.0)
    
    return embedding[:384]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "simple-hash-embedding",
        "dimensions": 384
    })

@app.route('/embed', methods=['POST'])
def embed():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        embedding = simple_embedding(text)
        
        return jsonify({
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": "simple-hash-embedding"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "Simple Embeddings Service",
        "status": "running"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 
