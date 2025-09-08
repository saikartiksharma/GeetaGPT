from flask import Flask, render_template, request, jsonify
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geeta_gpt import GeetaChatGPT

app = Flask(__name__)
app.config['SECRET_KEY'] = 'geeta-gpt-secret-key'

# Initialize chatbot
chatbot = None

def init_chatbot():
    global chatbot
    try:
        chatbot = GeetaChatGPT()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if chatbot is None:
        init_chatbot()
    
    if chatbot is None:
        return jsonify({"error": "Chatbot not available"}), 500
    
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        response = chatbot.generate_response(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "chatbot": "initialized" if chatbot else "not initialized"})

if __name__ == '__main__':
    init_chatbot()
    app.run(debug=True, host='0.0.0.0', port=5000)