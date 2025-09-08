import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import json
import pickle
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math
import random

class MultilingualTokenizer:
    """Multilingual tokenizer for Sanskrit, Hindi, and English text."""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.vocab_inv = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for tokenization."""
        if pd.isna(text):
            return ""
        
        # Convert to string and normalize
        text = str(text).strip()
        
        # Add spaces between characters and punctuation for better tokenization
        text = re.sub(r'([!?.])', r' \1 ', text)
        text = re.sub(r'([^a-zA-Z0-9\s!?.])', r' \1 ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_counts = Counter()
        
        for text in texts:
            processed = self._preprocess_text(text)
            words = processed.split()
            word_counts.update(words)
        
        # Create vocabulary with most common words
        most_common = word_counts.most_common(self.vocab_size - 4)  # Reserve space for special tokens
        vocab = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3}
        
        for i, (word, _) in enumerate(most_common, 4):
            vocab[word] = i
        
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
    
    def train(self, dataset_path: str):
        """Train tokenizer on dataset."""
        print("🔤 Training multilingual tokenizer...")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Combine all text for vocabulary building
        all_texts = []
        for _, row in df.iterrows():
            all_texts.extend([str(row['sanskrit']), str(row['hindi']), str(row['english'])])
        
        # Build vocabulary
        self._build_vocab(all_texts)
        
        print(f"✅ Vocabulary built with {len(self.vocab)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        processed = self._preprocess_text(text)
        tokens = processed.split()
        
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ' '.join([self.vocab_inv.get(token_id, self.unk_token) for token_id in token_ids])
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    
    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Scaled dot product attention."""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, v)
        self.attention_weights = attn_probs.detach()
        
        return output
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = q.size(0)
        
        # Linear transformations
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(output)

class FeedForward(nn.Module):
    """Position-wise feed forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class GeetaGPT(nn.Module):
    """GPT-style model for Bhagavad Gita text generation."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, n_layers: int = 6, 
                 d_ff: int = 2048, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Embeddings and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Output
        logits = self.fc_out(x)
        
        return logits
    
    def generate(self, start_token: int, max_length: int = 100, temperature: float = 1.0, 
                 top_k: Optional[int] = None, device: str = 'cpu'):
        """Generate text using trained model."""
        self.eval()
        
        with torch.no_grad():
            current_token = torch.tensor([[start_token]], device=device)
            generated = [start_token]
            
            for _ in range(max_length - 1):
                # Forward pass
                logits = self.forward(current_token)
                
                # Get logits for last position
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k sampling
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[indices] = values
                
                # Sample next token
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).item()
                
                generated.append(next_token)
                
                # Update current token
                current_token = torch.tensor([[next_token]], device=device)
                
                # Stop if end token is generated
                if next_token == 3:  # Assuming <EOS> token is 3
                    break
        
        return generated

class GeetaDataset(torch.utils.data.Dataset):
    """Dataset for Bhagavad Gita verses."""
    
    def __init__(self, df: pd.DataFrame, tokenizer: MultilingualTokenizer, max_length: int = 128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare all sequences
        self.sequences = []
        for _, row in df.iterrows():
            # Combine sanskrit, hindi, and english text
            combined_text = f"{row['sanskrit']} {row['hindi']} {row['english']}"
            tokens = self.tokenizer.encode(combined_text)
            
            # Create sequences for training
            for i in range(len(tokens) - max_length + 1):
                input_seq = tokens[i:i + max_length]
                target_seq = tokens[i + 1:i + max_length + 1]
                
                self.sequences.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        
        # Pad sequences
        input_seq = input_seq + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (self.max_length - len(input_seq))
        target_seq = target_seq + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (self.max_length - len(target_seq))
        
        return torch.tensor(input_seq), torch.tensor(target_seq)

class GeetaTrainer:
    """Trainer for GeetaGPT model."""
    
    def __init__(self, model: GeetaGPT, tokenizer: MultilingualTokenizer, dataset: GeetaDataset,
                 device: str = 'cpu', learning_rate: float = 0.0001):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab[tokenizer.pad_token])
        
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            
            # Compute loss
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        return avg_loss
    
    def forward(self, inputs: torch.Tensor):
        """Forward pass with attention mask."""
        # Create attention mask
        mask = (inputs != self.tokenizer.vocab[self.tokenizer.pad_token]).unsqueeze(1).unsqueeze(2)
        
        return self.model(inputs, mask)
    
    def train(self, num_epochs: int, save_path: str = "geeta_gpt_model.pth"):
        """Train the model."""
        print(f"🚀 Starting training on {self.device}")
        
        for epoch in range(num_epochs):
            loss = self.train_epoch(epoch)
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, save_path)
                print(f"💾 Model saved at epoch {epoch}")
        
        print("✅ Training completed!")
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.tokenizer.vocab,
            'vocab_size': self.tokenizer.get_vocab_size(),
        }, path)
        print(f"💾 Model saved to {path}")

class GeetaChatGPT:
    """Main chat interface for GeetaGPT."""
    
    def __init__(self, model_path: str = "geeta_gpt_model.pth", tokenizer_path: str = "tokenizer.pkl"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        
        # Initialize model
        self.model = self.load_model(model_path)
        
        # Conversation history
        self.conversation_history = []
        
        print("🎉 GeetaGPT loaded successfully!")
    
    def load_tokenizer(self, path: str) -> MultilingualTokenizer:
        """Load trained tokenizer."""
        try:
            with open(path, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"🔤 Tokenizer loaded from {path}")
            return tokenizer
        except FileNotFoundError:
            print(f"🔤 Tokenizer not found, creating new one...")
            tokenizer = MultilingualTokenizer()
            tokenizer.train("geeta_dataset.csv")
            
            # Save tokenizer
            with open(path, 'wb') as f:
                pickle.dump(tokenizer, f)
            return tokenizer
    
    def load_model(self, path: str) -> GeetaGPT:
        """Load trained model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            model = GeetaGPT(
                vocab_size=checkpoint['vocab_size'],
                d_model=512,
                n_heads=8,
                n_layers=6,
                d_ff=2048,
                max_len=512
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print(f"🧠 Model loaded from {path}")
            return model
        except FileNotFoundError:
            print("🧠 No pre-trained model found. Please train the model first.")
            return None
    
    def generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.8):
        """Generate response to user prompt."""
        if self.model is None:
            return "❌ Model not trained yet. Please train the model first."
        
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_tokens = prompt_tokens[-50:]  # Use last 50 tokens to fit context
        
        # Generate response
        response_tokens = self.model.generate(
            start_token=self.tokenizer.vocab[self.tokenizer.bos_token],
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            device=self.device
        )
        
        # Decode response
        response = self.tokenizer.decode(response_tokens)
        
        # Clean up special tokens
        response = response.replace(f" {self.tokenizer.vocab_inv.get(self.tokenizer.bos_token, '')}", "")
        response = response.replace(f" {self.tokenizer.vocab_inv.get(self.tokenizer.eos_token, '')}", "")
        
        return response
    
    def chat(self):
        """Interactive chat interface."""
        print("🙏 Welcome to GeetaGPT - Your Bhagavad Gita AI Companion!")
        print("💬 Ask questions about the Bhagavad Gita or request wisdom.")
        print("📝 Type 'quit' to exit, 'history' to see conversation history")
        print("=" * 80)
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🙏 Thank you for chatting with GeetaGPT!")
                print("🌅 May the wisdom of the Bhagavad Gita guide you.")
                break
            
            elif user_input.lower() == 'history':
                self.show_history()
                continue
            
            elif user_input.lower() == 'help':
                self.show_help()
                continue
            
            elif not user_input:
                continue
            
            # Add to conversation history
            self.conversation_history.append(f"User: {user_input}")
            
            # Generate response
            print("\n🤖 GeetaGPT: ", end="")
            response = self.generate_response(user_input)
            print(response)
            
            # Add to conversation history
            self.conversation_history.append(f"GeetaGPT: {response}")
    
    def show_history(self):
        """Show conversation history."""
        print("\n📜 Conversation History:")
        print("-" * 40)
        for i, entry in enumerate(self.conversation_history[-10:], 1):  # Show last 10 entries
            print(f"{i}. {entry}")
        print("-" * 40)
    
    def show_help(self):
        """Show help information."""
        print("\n🔧 GeetaGPT Help:")
        print("- Ask questions about Bhagavad Gita, philosophy, or life")
        print("- Request verses, explanations, or wisdom")
        print("- Type 'history' to see conversation history")
        print("- Type 'quit' to exit")
        print("- Be respectful and seek knowledge with devotion")

def main():
    """Main function."""
    print("🚀 Starting GeetaGPT - Bhagavad Gita AI Companion")
    print("=" * 60)
    
    # Initialize chatbot
    chatbot = GeetaChatGPT()
    
    # Start chat
    chatbot.chat()

if __name__ == "__main__":
    main()