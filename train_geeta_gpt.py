import torch
import pandas as pd
from geeta_gpt import GeetaGPT, MultilingualTokenizer, GeetaDataset, GeetaTrainer
import argparse
import os
import time
import json
import datetime
import psutil
import gc

def print_system_info():
    """Print system information and available resources."""
    print("=" * 80)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 80)
    
    # CPU Information
    cpu_count = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"🔧 CPU: {cpu_count} physical cores, {cpu_logical} logical cores")
    print(f"📊 CPU Usage: {cpu_percent}%")
    
    # Memory Information
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024**3)  # GB
    available_memory = memory.available / (1024**3)  # GB
    print(f"💾 Total Memory: {total_memory:.2f} GB")
    print(f"📉 Available Memory: {available_memory:.2f} GB ({memory.percent}% used)")
    
    # GPU Information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        print(f"🎮 GPU: {gpu_name}")
        print(f"🎮 GPU Memory Total: {gpu_memory:.2f} GB")
        print(f"🎮 GPU Memory Allocated: {gpu_allocated:.2f} GB")
        print(f"🎮 GPU Memory Cached: {gpu_cached:.2f} GB")
    else:
        print("🎮 GPU: Not available")
    
    print("=" * 80)

def load_dataset_with_progress():
    """Load dataset with progress tracking."""
    print("📖 Loading Bhagavad Gita dataset...")
    start_time = time.time()
    
    try:
        # Check file size first
        file_size = os.path.getsize('geeta_dataset.csv')
        print(f"📄 Dataset file size: {file_size / (1024*1024):.2f} MB")
        
        # Load dataset
        df = pd.read_csv('geeta_dataset.csv')
        load_time = time.time() - start_time
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total rows: {len(df):,}")
        print(f"⏱️  Loading time: {load_time:.2f} seconds")
        
        # Display column information
        print("📋 Dataset columns:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"   • {col}: {non_null_count:,} non-null values")
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: geeta_dataset.csv not found")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def train_tokenizer_with_progress(tokenizer, dataset_path):
    """Train tokenizer with progress tracking."""
    print("🔤 Training multilingual tokenizer...")
    start_time = time.time()
    
    try:
        # Train tokenizer
        tokenizer.train(dataset_path)
        
        training_time = time.time() - start_time
        vocab_size = tokenizer.get_vocab_size()
        
        print(f"✅ Tokenizer training completed!")
        print(f"🔤 Vocabulary size: {vocab_size:,} tokens")
        print(f"⏱️  Training time: {training_time:.2f} seconds")
        
        # Show sample vocabulary
        print("📝 Sample vocabulary:")
        sample_tokens = list(tokenizer.vocab.keys())[:10]
        for token in sample_tokens:
            print(f"   • '{token}' -> {tokenizer.vocab[token]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training tokenizer: {str(e)}")
        return False

def create_model_with_progress(tokenizer):
    """Create model with progress tracking."""
    print("🧠 Creating GeetaGPT model...")
    start_time = time.time()
    
    try:
        vocab_size = tokenizer.get_vocab_size()
        
        # Model configuration
        model_config = {
            'vocab_size': vocab_size,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'max_len': 512
        }
        
        print(f"📊 Model configuration:")
        for key, value in model_config.items():
            print(f"   • {key}: {value}")
        
        # Initialize model
        model = GeetaGPT(**model_config)
        
        creation_time = time.time() - start_time
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully!")
        print(f"🔢 Total parameters: {total_params:,}")
        print(f"🔢 Trainable parameters: {trainable_params:,}")
        print(f"⏱️  Creation time: {creation_time:.2f} seconds")
        
        return model, model_config
        
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return None, None

def create_dataset_with_progress(df, tokenizer, max_length=128):
    """Create dataset with progress tracking."""
    print("📚 Creating training dataset...")
    start_time = time.time()
    
    try:
        # Create dataset
        dataset = GeetaDataset(df, tokenizer, max_length=max_length)
        
        creation_time = time.time() - start_time
        
        print(f"✅ Dataset created successfully!")
        print(f"📊 Total training samples: {len(dataset):,}")
        print(f"⏱️  Creation time: {creation_time:.2f} seconds")
        
        # Show sample data
        print("📝 Sample training data:")
        sample_input, sample_target = dataset[0]
        print(f"   • Input sequence shape: {sample_input.shape}")
        print(f"   • Target sequence shape: {sample_target.shape}")
        print(f"   • Input tokens: {sample_input[:10].tolist()}...")
        print(f"   • Target tokens: {sample_target[:10].tolist()}...")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error creating dataset: {str(e)}")
        return None

def monitor_memory_usage():
    """Monitor memory usage during training."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"📊 Memory Usage - RAM: {memory_mb:.2f} MB, GPU: {gpu_memory:.2f} MB")
    else:
        print(f"📊 Memory Usage - RAM: {memory_mb:.2f} MB")
    
    return memory_mb

def create_training_log(args):
    """Create training log file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_log_{timestamp}.json"
    
    os.makedirs("logs", exist_ok=True)
    
    log_data = {
        'timestamp': timestamp,
        'start_time': datetime.datetime.now().isoformat(),
        'parameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'device': args.device,
            'model_save_path': args.model_save_path,
            'tokenizer_save_path': args.tokenizer_save_path
        },
        'system_info': {
            'cpu_cores': psutil.cpu_count(),
            'total_memory': psutil.virtual_memory().total / (1024**3),
            'cuda_available': torch.cuda.is_available()
        }
    }
    
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📝 Training log created: {log_filename}")
    return log_filename

def update_training_log(log_filename, epoch_data):
    """Update training log with epoch data."""
    try:
        with open(log_filename, 'r') as f:
            log_data = json.load(f)
        
        if 'epochs' not in log_data:
            log_data['epochs'] = []
        
        log_data['epochs'].append(epoch_data)
        log_data['end_time'] = datetime.datetime.now().isoformat()
        
        with open(log_filename, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Error updating training log: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Train GeetaGPT model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='geeta_gpt_model.pth', help='Path to save trained model')
    parser.add_argument('--tokenizer_save_path', type=str, default='tokenizer.pkl', help='Path to save tokenizer')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for training logs')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume_training', type=str, default=None, help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 GeetaGPT Training Script - Optimized Version")
    print("=" * 80)
    
    # Print system information
    print_system_info()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 Training GeetaGPT on {device}")
    print(f"📊 Training Parameters:")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Batch Size: {args.batch_size}")
    print(f"   • Learning Rate: {args.lr}")
    print(f"   • Checkpoint Interval: {args.checkpoint_interval}")
    print(f"   • Model Save Path: {args.model_save_path}")
    print(f"   • Tokenizer Save Path: {args.tokenizer_save_path}")
    print(f"   • Resume Training: {args.resume_training if args.resume_training else 'No'}")
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create training log
    log_filename = create_training_log(args)
    
    # Monitor initial memory
    initial_memory = monitor_memory_usage()
    
    # Load dataset
    print("\n📖 Loading and preparing dataset...")
    df = load_dataset_with_progress()
    if df is None:
        return
    
    # Initialize tokenizer
    print("\n🔤 Preparing tokenizer...")
    tokenizer = MultilingualTokenizer()
    if not train_tokenizer_with_progress(tokenizer, 'geeta_dataset.csv'):
        return
    
    # Save tokenizer
    print("\n💾 Saving tokenizer...")
    tokenizer_start_time = time.time()
    with open(args.tokenizer_save_path, 'wb') as f:
        import pickle
        pickle.dump(tokenizer, f)
    tokenizer_save_time = time.time() - tokenizer_start_time
    print(f"✅ Tokenizer saved to {args.tokenizer_save_path}")
    print(f"⏱️  Save time: {tokenizer_save_time:.2f} seconds")
    
    # Create model
    print("\n🧠 Creating model...")
    model, model_config = create_model_with_progress(tokenizer)
    if model is None:
        return
    
    # Create dataset
    print("\n📚 Preparing training dataset...")
    dataset = create_dataset_with_progress(df, tokenizer, max_length=128)
    if dataset is None:
        return
    
    # Initialize trainer
    print("\n🏋️ Initializing trainer...")
    trainer_start_time = time.time()
    trainer = GeetaTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        learning_rate=args.lr
    )
    trainer_init_time = time.time() - trainer_start_time
    print(f"✅ Trainer initialized in {trainer_init_time:.2f} seconds")
    
    # Monitor memory after setup
    setup_memory = monitor_memory_usage()
    print(f"📊 Memory increase after setup: {setup_memory - initial_memory:.2f} MB")
    
    # Check if resuming training
    start_epoch = 0
    if args.resume_training:
        print(f"\n🔄 Attempting to resume training from {args.resume_training}")
        try:
            checkpoint = torch.load(args.resume_training, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✅ Training resumed from epoch {start_epoch}")
            
            # Update training log with resume information
            resume_data = {
                'action': 'resume',
                'from_checkpoint': args.resume_training,
                'resumed_epoch': start_epoch,
                'timestamp': datetime.datetime.now().isoformat()
            }
            update_training_log(log_filename, resume_data)
        except Exception as e:
            print(f"⚠️  Failed to resume training: {str(e)}")
            print("🔄 Starting fresh training...")
    
    # Train model
    print("\n🏋️ Starting training...")
    print("=" * 80)
    
    total_start_time = time.time()
    
    try:
        total_batches = len(trainer.train_loader)
        print(f"📊 Total batches per epoch: {total_batches:,}")
        
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            print(f"🔥 Starting Epoch {epoch + 1}/{args.epochs}")
            
            # Train epoch with enhanced progress tracking
            epoch_loss = trainer.train_epoch(epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            # Monitor memory
            epoch_memory = monitor_memory_usage()
            
            # Calculate training metrics
            batches_per_second = total_batches / epoch_time if epoch_time > 0 else 0
            samples_per_second = total_batches * args.batch_size / epoch_time if epoch_time > 0 else 0
            
            # Print epoch summary
            print(f"📊 Epoch Summary:")
            print(f"   • Loss: {epoch_loss:.4f}")
            print(f"   • Time: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)")
            print(f"   • Memory: {epoch_memory:.2f} MB")
            print(f"   • Batches/Second: {batches_per_second:.2f}")
            print(f"   • Samples/Second: {samples_per_second:.0f}")
            print(f"   • Throughput: {samples_per_second * args.batch_size:.0f} samples/sec")
            
            # Save checkpoint
            if epoch % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
                checkpoint_start_time = time.time()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': epoch_loss,
                    'config': model_config,
                    'epoch_time': epoch_time,
                    'memory_usage': epoch_memory,
                    'training_start_time': total_start_time,
                    'args': vars(args)
                }
                
                # Save multiple checkpoint versions
                torch.save(checkpoint, args.model_save_path)
                torch.save(checkpoint, f"{args.model_save_path}.epoch_{epoch}")
                
                # Save best model if loss is improved
                if epoch_loss < getattr(trainer, 'best_loss', float('inf')):
                    trainer.best_loss = epoch_loss
                    torch.save(checkpoint, f"{args.model_save_path}.best")
                    print(f"🏆 New best model saved with loss: {epoch_loss:.4f}")
                
                checkpoint_time = time.time() - checkpoint_start_time
                print(f"💾 Model checkpoint saved at epoch {epoch} (took {checkpoint_time:.2f} seconds)")
                print(f"   • Main checkpoint: {args.model_save_path}")
                print(f"   • Epoch checkpoint: {args.model_save_path}.epoch_{epoch}")
                if epoch_loss < getattr(trainer, 'best_loss', float('inf')):
                    print(f"   • Best model: {args.model_save_path}.best")
            
            # Update training log
            epoch_data = {
                'epoch': epoch,
                'loss': epoch_loss,
                'time': epoch_time,
                'memory': epoch_memory,
                'batches_per_second': batches_per_second,
                'samples_per_second': samples_per_second,
                'timestamp': datetime.datetime.now().isoformat()
            }
            update_training_log(log_filename, epoch_data)
            print(f"📝 Epoch data logged to {log_filename}")
        
        print(f"✅ Epoch {epoch + 1} completed!")
        print("-" * 50)
        
        # Enhanced garbage collection
        if epoch % 5 == 0:
            print("🗑️  Performing garbage collection...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   • CUDA cache cleared")
            print("✅ Garbage collection completed")
        
        # Print cumulative statistics
        cumulative_time = time.time() - total_start_time
        avg_epoch_time = cumulative_time / (epoch + 1)
        estimated_remaining = avg_epoch_time * (args.epochs - epoch - 1)
        
        print(f"⏱️  Cumulative Time: {cumulative_time:.2f}s ({cumulative_time/60:.2f}m)")
        print(f"⏱️  Estimated Remaining: {estimated_remaining:.2f}s ({estimated_remaining/60:.2f}m)")
        
        # Print loss trend analysis
        if len(trainer.epoch_losses) > 1:
            loss_change = trainer.epoch_losses[-1] - trainer.epoch_losses[-2]
            loss_change_pct = (loss_change / trainer.epoch_losses[-2]) * 100 if trainer.epoch_losses[-2] != 0 else 0
            trend = "📉" if loss_change > 0 else "📈" if loss_change < 0 else "➡️"
            print(f"{trend} Loss Trend: {loss_change:+.4f} ({loss_change_pct:+.2f}%)")
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user!")
        print("💾 Saving current state...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': epoch_loss,
            'config': model_config,
            'interrupted': True
        }, args.model_save_path.replace('.pth', '_interrupted.pth'))
        print("✅ Model saved with '_interrupted' suffix")
    
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("💾 Saving current state...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': epoch_loss,
            'config': model_config,
            'error': str(e)
        }, args.model_save_path.replace('.pth', '_error.pth'))
        print("✅ Model saved with '_error' suffix")
        return
    
    # Save final model
    print("💾 Saving final model...")
    final_save_start_time = time.time()
    trainer.save_model(args.model_save_path)
    final_save_time = time.time() - final_save_start_time
    
    total_time = time.time() - total_start_time
    
    # Final memory check
    final_memory = monitor_memory_usage()
    
    print("=" * 80)
    print("🎉 Training completed successfully!")
    print(f"📊 Final Summary:")
    print(f"   • Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   • Final Memory Usage: {final_memory:.2f} MB")
    print(f"   • Memory Increase: {final_memory - initial_memory:.2f} MB")
    print(f"   • Final Model Save Time: {final_save_time:.2f} seconds")
    
    # Calculate final statistics
    avg_epoch_time = total_time / args.epochs if args.epochs > 0 else 0
    total_samples = args.epochs * total_batches * args.batch_size
    total_samples_per_second = total_samples / total_time if total_time > 0 else 0
    
    print(f"📈 Performance Statistics:")
    print(f"   • Average Epoch Time: {avg_epoch_time:.2f} seconds")
    print(f"   • Total Samples Processed: {total_samples:,}")
    print(f"   • Average Samples/Second: {total_samples_per_second:.0f}")
    print(f"   • Total Batches Processed: {args.epochs * total_batches:,}")
    
    # Print memory efficiency
    memory_efficiency = (total_samples / (final_memory - initial_memory)) if (final_memory - initial_memory) > 0 else 0
    print(f"🧠 Memory Efficiency: {memory_efficiency:.0f} samples per MB")
    
    print("=" * 80)
    print("📝 Training logs saved to the logs directory")
    print("💾 Model saved to the specified path")
    print("🚀 Training session completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()