# GeetaGPT - Bhagavad Gita AI Companion

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

## Quick Start

### 1. Install Dependencies

```bash
pip install torch pandas numpy scikit-learn flask
```

### 2. Train the Model

```bash
python train_geeta_gpt.py
```

### 3. Start the Web Interface

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

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

## Usage

### Command Line Interface

```bash
python geeta_gpt.py
```

### Web Interface

```bash
python app.py
```

### Training

```bash
# Default training
python train_geeta_gpt.py

# Custom parameters
python train_geeta_gpt.py --epochs 100 --batch_size 64 --lr 0.0001
```

### Evaluation

```bash
python evaluate_geeta_gpt.py --model_path models/geeta_gpt_model.pth
```

## Configuration

Edit `config/config.json` to customize model parameters and training settings.

## API Usage

### Python API

```python
from geeta_gpt import GeetaChatGPT

chatbot = GeetaChatGPT()
response = chatbot.generate_response("What is karma?")
print(response)
```

### Web API

```python
import requests

response = requests.post('http://localhost:5000/chat', json={
    'message': 'What is the meaning of life according to Bhagavad Gita?'
})

print(response.json()['response'])
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run with Docker directly
docker build -t geeta-gpt .
docker run -p 5000:5000 geeta-gpt
```

## Evaluation Metrics

The system includes comprehensive evaluation:

- **Perplexity**: Language model quality metric
- **Keyword Coverage**: Relevance to expected topics
- **Quality Score**: Overall response quality
- **Diversity**: Response variety and creativity

## Contributing

We welcome contributions to enhance GeetaGPT:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and spiritual purposes. Please respect the sacred nature of the Bhagavad Gita text.

## Support

For issues and questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description

🙏 May the wisdom of the Bhagavad Gita guide all beings on the path of righteousness and peace.

---

*Built with devotion and technology for spreading ancient wisdom in modern times.*