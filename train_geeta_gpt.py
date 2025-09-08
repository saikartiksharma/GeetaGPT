import torch
import pandas as pd
from geeta_gpt import GeetaGPT, MultilingualTokenizer, GeetaDataset, GeetaTrainer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Train GeetaGPT model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='geeta_gpt_model.pth', help='Path to save trained model')
    parser.add_argument('--tokenizer_save_path', type=str, default='tokenizer.pkl', help='Path to save tokenizer')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 Training GeetaGPT on {device}")
    print(f"📊 Parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    
    # Load dataset
    print("📖 Loading Bhagavad Gita dataset...")
    try:
        df = pd.read_csv('geeta_dataset.csv')
        print(f"✅ Loaded {len(df)} verses")
    except FileNotFoundError:
        print("❌ Error: geeta_dataset.csv not found")
        return
    
    # Initialize tokenizer
    tokenizer = MultilingualTokenizer()
    tokenizer.train('geeta_dataset.csv')
    
    # Save tokenizer
    with open(args.tokenizer_save_path, 'wb') as f:
        import pickle
        pickle.dump(tokenizer, f)
    print(f"💾 Tokenizer saved to {args.tokenizer_save_path}")
    
    # Initialize model
    model = GeetaGPT(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=512
    )
    
    # Create dataset
    dataset = GeetaDataset(df, tokenizer, max_length=128)
    print(f"📚 Dataset created with {len(dataset)} training samples")
    
    # Initialize trainer
    trainer = GeetaTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        learning_rate=args.lr
    )
    
    # Train model
    print("🏋️ Starting training...")
    trainer.train(num_epochs=args.epochs, save_path=args.model_save_path)
    
    # Save final model
    trainer.save_model(args.model_save_path)
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()