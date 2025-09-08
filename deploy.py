#!/usr/bin/env python3
"""
Deployment script for GeetaGPT - Bhagavad Gita AI Companion
This script sets up the complete system and provides easy deployment options.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import argparse
import codecs

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

class GeetaGPTDeployer:
    """Deployment manager for GeetaGPT."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements = [
            "torch",
            "pandas", 
            "numpy",
            "scikit-learn"
        ]
        
    def check_dependencies(self):
        """Check and install required dependencies."""
        print("🔍 Checking dependencies...")
        
        missing = []
        for package in self.requirements:
            try:
                __import__(package)
                print(f"✅ {package} - already installed")
            except ImportError:
                missing.append(package)
                print(f"❌ {package} - missing")
        
        if missing:
            print(f"\n📦 Installing missing packages: {', '.join(missing)}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                print("✅ All dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
                return False
        
        return True
    
    def create_directory_structure(self):
        """Create necessary directory structure."""
        print("📁 Creating directory structure...")
        
        directories = [
            "models",
            "data", 
            "logs",
            "config",
            "static",
            "templates"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def create_config_files(self):
        """Create configuration files."""
        print("⚙️ Creating configuration files...")
        
        # Main config
        config = {
            "model": {
                "path": "models/geeta_gpt_model.pth",
                "vocab_size": 50000,
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 6,
                "d_ff": 2048
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.0001,
                "epochs": 50,
                "max_length": 128
            },
            "inference": {
                "max_length": 100,
                "temperature": 0.8,
                "top_k": 50
            },
            "paths": {
                "dataset": "geeta_dataset.csv",
                "tokenizer": "models/tokenizer.pkl",
                "logs": "logs/"
            }
        }
        
        with open(self.project_root / "config" / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Created config/config.json")
        
        # Create .env file for environment variables
        env_content = """# GeetaGPT Configuration
MODEL_PATH=models/geeta_gpt_model.pth
TOKENIZER_PATH=models/tokenizer.pkl
DATASET_PATH=geeta_dataset.csv
LOG_LEVEL=INFO
DEVICE=auto
DEBUG=False
"""
        
        with open(self.project_root / ".env", "w") as f:
            f.write(env_content)
        
        print("✅ Created .env file")
    
    def create_web_interface(self):
        """Create simple web interface."""
        print("🌐 Creating web interface...")
        
        # Create Flask app
        app_code = '''from flask import Flask, render_template, request, jsonify
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
'''
        
        with open(self.project_root / "app.py", "w") as f:
            f.write(app_code)
        
        print("✅ Created app.py")
        
        # Create HTML template
        html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeetaGPT - Bhagavad Gita AI Companion</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 0;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 50px;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .chat-container {
            height: 400px;
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            overflow-y: auto;
            margin-bottom: 20px;
            background: #f9f9f9;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
        }
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background: #e9ecef;
            color: #333;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        .input-field {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 25px;
            outline: none;
            font-size: 16px;
        }
        .send-button {
            padding: 12px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
        }
        .send-button:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🙏 GeetaGPT - Bhagavad Gita AI Companion</h1>
        <div class="chat-container" id="chatContainer">
            <div class="message bot-message">
                Hello! I'm GeetaGPT, your AI companion for Bhagavad Gita wisdom. How can I help you today?
            </div>
        </div>
        <div class="input-container">
            <input type="text" class="input-field" id="messageInput" placeholder="Ask a question about Bhagavad Gita..." onkeypress="handleKeyPress(event)">
            <button class="send-button" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                // Remove typing indicator
                removeTypingIndicator();
                
                if (response.ok) {
                    addMessage(data.response, 'bot');
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                }
            } catch (error) {
                removeTypingIndicator();
                addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            }
        }
        
        function addMessage(text, sender) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function showTypingIndicator() {
            const chatContainer = document.getElementById('chatContainer');
            const typingDiv = document.createElement('div');
            typingDiv.id = 'typingIndicator';
            typingDiv.className = 'message bot-message';
            typingDiv.textContent = 'GeetaGPT is typing...';
            chatContainer.appendChild(typingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function removeTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            if (indicator) {
                indicator.remove();
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
'''
        
        templates_dir = self.project_root / "templates"
        with open(templates_dir / "index.html", "w") as f:
            f.write(html_template)
        
        print("✅ Created web interface")
    
    def create_docker_setup(self):
        """Create Docker setup for easy deployment."""
        print("🐳 Creating Docker setup...")
        
        # Dockerfile
        dockerfile = '''FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
'''
        
        with open(self.project_root / "Dockerfile", "w") as f:
            f.write(dockerfile)
        
        # Docker Compose
        docker_compose = '''version: '3.8'

services:
  geeta-gpt:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - DEVICE=auto
    restart: unless-stopped
'''
        
        with open(self.project_root / "docker-compose.yml", "w") as f:
            f.write(docker_compose)
        
        print("✅ Created Docker setup")
    
    def create_installation_script(self):
        """Create installation script."""
        print("📜 Creating installation script...")
        
        script = '''#!/bin/bash

echo "🚀 Starting GeetaGPT Installation..."
echo "===================================="

# Check if Python 3.7+ is installed
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python 3.7+ is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p models data logs config static templates

# Set up environment
echo "⚙️ Setting up environment..."
cp .env.example .env 2>/dev/null || echo "Creating .env file..."
touch models/tokenizer.pkl

echo "✅ Installation completed!"
echo ""
echo "Next steps:"
echo "1. Train the model: python train_geeta_gpt.py"
echo "2. Run the web interface: python app.py"
echo "3. Or run in Docker: docker-compose up"
echo ""
echo "🙏 Jai Shri Krishna!"
'''
        
        with open(self.project_root / "install.sh", "w") as f:
            f.write(script)
        
        # Make script executable
        os.chmod(self.project_root / "install.sh", 0o755)
        
        print("✅ Created installation script")
    
    def create_readme(self):
        """Create comprehensive README."""
        print("📖 Creating README...")
        
        readme_content = '''# GeetaGPT - Bhagavad Gita AI Companion

🙏 A sophisticated neural network-based AI system trained on the complete Bhagavad Gita, offering wisdom, guidance, and philosophical insights in Sanskrit, Hindi, and English.

## Features

- 🧠 **Neural Network Architecture**: GPT-2 style transformer model with multi-head attention
- 🌐 **Multilingual Support**: Processes Sanskrit, Hindi, and English text
- 🔤 **Advanced Tokenization**: Custom tokenizer optimized for multilingual text
- 🎯 **Generative AI**: Creates contextually relevant responses and verses
- 💬 **Conversational Interface**: Interactive chat interface
- 📊 **Evaluation System**: Comprehensive metrics and performance analysis
- 🌐 **Web Interface**: Flask-based web application
- 🐳 **Docker Support**: Easy deployment with Docker

## Installation

### Quick Start (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd geeta-gpt

# Run the installation script
./install.sh

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Train the model
python train_geeta_gpt.py

# Start the web interface
python app.py
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install torch pandas numpy scikit-learn flask

# Create directory structure
mkdir -p models data logs config static templates
```

## Usage

### Training the Model

```bash
# Train with default parameters
python train_geeta_gpt.py

# Custom training parameters
python train_geeta_gpt.py --epochs 100 --batch_size 64 --lr 0.0001
```

### Running the Chat Interface

```bash
# Command line interface
python geeta_gpt.py

# Web interface
python app.py
```

### Evaluation

```bash
# Evaluate model performance
python evaluate_geeta_gpt.py --model_path models/geeta_gpt_model.pth
```

## Project Structure

```
geeta-gpt/
├── geeta_gpt.py          # Main model implementation
├── train_geeta_gpt.py    # Training script
├── evaluate_geeta_gpt.py # Evaluation script
├── app.py               # Web interface
├── deploy.py            # Deployment script
├── main.py              # Basic chatbot
├── geeta_dataset.csv    # Bhagavad Gita dataset
├── models/              # Model storage
├── data/               # Data storage
├── logs/               # Log files
├── config/             # Configuration files
├── templates/          # Web templates
├── static/            # Static files
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose
└── README.md         # This file
```

## Model Architecture

### Core Components

1. **Multilingual Tokenizer**: Custom tokenizer optimized for Sanskrit, Hindi, and English
2. **Transformer Architecture**: Multi-head attention with positional encoding
3. **Neural Network**: GPT-2 style model with 6 layers, 8 attention heads
4. **Training Pipeline**: PyTorch-based training with custom dataset loader

### Key Parameters

- **Vocabulary Size**: 50,000 tokens
- **Model Dimensions**: 512 hidden units
- **Attention Heads**: 8
- **Layers**: 6 transformer blocks
- **Feed Forward**: 2048 units
- **Max Sequence Length**: 512 tokens

## Configuration

Edit `config/config.json` to customize:

```json
{
  "model": {
    "vocab_size": 50000,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 2048
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 50,
    "max_length": 128
  },
  "inference": {
    "max_length": 100,
    "temperature": 0.8,
    "top_k": 50
  }
}
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run with Docker directly
docker build -t geeta-gpt .
docker run -p 5000:5000 geeta-gpt
```

## API Usage

### Web API

```python
import requests

response = requests.post('http://localhost:5000/chat', json={
    'message': 'What is the meaning of life according to Bhagavad Gita?'
})

print(response.json()['response'])
```

### Python API

```python
from geeta_gpt import GeetaChatGPT

chatbot = GeetaChatGPT()
response = chatbot.generate_response("What is karma?")
print(response)
```

## Evaluation

The system includes comprehensive evaluation metrics:

- **Perplexity**: Language model quality metric
- **Keyword Coverage**: Relevance to expected topics
- **Quality Score**: Overall response quality
- **Diversity**: Response variety and creativity

Run evaluation:

```bash
python evaluate_geeta_gpt.py --model_path models/geeta_gpt_model.pth --num_samples 20
```

## Contributing

We welcome contributions to enhance GeetaGPT:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and spiritual purposes. Please respect the sacred nature of the Bhagavad Gita text.

## Acknowledgments

- Bhagavad Gita - The divine scripture
- Open source community for AI/ML tools
- PyTorch team for the deep learning framework

## Support

For issues and questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description

🙏 May the wisdom of the Bhagavad Gita guide all beings on the path of righteousness and peace.

---

*Built with devotion and technology for spreading ancient wisdom in modern times.*
'''
        
        with open(self.project_root / "README.md", "w") as f:
            f.write(readme_content)
        
        print("✅ Created comprehensive README")
    
    def deploy(self):
        """Complete deployment process."""
        print("🚀 Starting GeetaGPT deployment...")
        print("=" * 50)
        
        # Execute deployment steps
        steps = [
            ("Check dependencies", self.check_dependencies),
            ("Create directory structure", self.create_directory_structure),
            ("Create config files", self.create_config_files),
            ("Create web interface", self.create_web_interface),
            ("Create Docker setup", self.create_docker_setup),
            ("Create installation script", self.create_installation_script),
            ("Create README", self.create_readme)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            try:
                if step_func():
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"❌ {step_name} failed")
                    return False
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                return False
        
        print("\n" + "=" * 50)
        print("🎉 GeetaGPT deployment completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run './install.sh' to set up the environment")
        print("2. Train the model with 'python train_geeta_gpt.py'")
        print("3. Start the web interface with 'python app.py'")
        print("4. Or use Docker: 'docker-compose up'")
        print("\n🙏 Jai Shri Krishna!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Deploy GeetaGPT')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency checks')
    
    args = parser.parse_args()
    
    deployer = GeetaGPTDeployer()
    
    if args.skip_checks:
        deployer.check_dependencies = lambda: True
    
    success = deployer.deploy()
    
    if not success:
        print("\n❌ Deployment failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()