from flask import Flask, request, Response, jsonify
import requests
import json
import os

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY') or os.environ.get('NIM_API_KEY', 'your-nvidia-api-key-here')
NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1'

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@app.route('/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-8b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        nim_payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False
        }
        
        headers = {
            'Authorization': f'Bearer {NVIDIA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{NVIDIA_BASE_URL}/chat/completions',
            headers=headers,
            json=nim_payload,
            timeout=60
        )
        
        return Response(
            response.content,
            status=response.status_code,
            content_type='application/json'
        )
        
    except Exception as e:
        error_response = {
            'error': {
                'message': str(e),
                'type': 'proxy_error',
                'code': 'internal_error'
            }
        }
        return jsonify(error_response), 500

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
@app.route('/models', methods=['GET', 'OPTIONS'])
def list_models():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        headers = {'Authorization': f'Bearer {NVIDIA_API_KEY}'}
        response = requests.get(f'{NVIDIA_BASE_URL}/models', headers=headers)
        return Response(response.content, status=response.status_code, content_type='application/json')
    except:
        fallback = {
            'data': [
                {'id': 'meta/llama-3.1-8b-instruct', 'object': 'model'},
                {'id': 'meta/llama-3.1-70b-instruct', 'object': 'model'}
            ]
        }
        return jsonify(fallback), 200

@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'nvidia-nim-proxy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
