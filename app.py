from flask import Flask, request, Response, jsonify
import requests
import json
import os

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY') or os.environ.get('NIM_API_KEY', 'your-nvidia-api-key-here')
NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1'

# Very aggressive trimming for long RPs
MAX_MESSAGES = 15  # Start with 15, adjust if needed
MAX_CHARS_PER_MESSAGE = 2000  # Limit individual message length

def trim_message_content(content, max_chars=MAX_CHARS_PER_MESSAGE):
    """Trim individual message content if too long"""
    if isinstance(content, str) and len(content) > max_chars:
        return content[:max_chars] + "..."
    return content

def trim_messages(messages, max_messages=MAX_MESSAGES):
    """
    Aggressively trims message history.
    """
    if not messages:
        return []
    
    # First, trim individual message lengths
    trimmed_content = []
    for msg in messages:
        new_msg = msg.copy()
        if 'content' in new_msg:
            new_msg['content'] = trim_message_content(new_msg['content'])
        trimmed_content.append(new_msg)
    
    # Then limit number of messages
    if len(trimmed_content) <= max_messages:
        return trimmed_content
    
    # Keep system message + recent messages
    if trimmed_content[0].get('role') == 'system':
        return [trimmed_content[0]] + trimmed_content[-(max_messages - 1):]
    else:
        return trimmed_content[-max_messages:]

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
        # Get raw data with size limit
        raw_data = request.get_data(cache=False, as_text=True)
        
        # Parse JSON
        try:
            data = json.loads(raw_data)
        except:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-8b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Aggressive trimming
        trimmed_messages = trim_messages(messages)
        
        nim_payload = {
            'model': model,
            'messages': trimmed_messages,
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
