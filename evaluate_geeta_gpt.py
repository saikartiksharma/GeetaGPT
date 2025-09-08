import torch
import pandas as pd
import numpy as np
from geeta_gpt import GeetaGPT, MultilingualTokenizer
import argparse
import json
from typing import List, Dict
import random

class GeetaEvaluator:
    """Evaluator for GeetaGPT model."""
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'auto'):
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load tokenizer
        import pickle
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load model
        self.model = self.load_model(model_path)
        
        print(f"🧠 Evaluator initialized on {self.device}")
    
    def load_model(self, model_path: str) -> GeetaGPT:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
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
        return model
    
    def perplexity(self, text: str) -> float:
        """Calculate perplexity of text."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2:
            return float('inf')
        
        # Prepare input
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        # Pad sequences
        max_len = 128
        input_tokens = input_tokens + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (max_len - len(input_tokens))
        target_tokens = target_tokens + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (max_len - len(target_tokens))
        
        input_tensor = torch.tensor([input_tokens], device=self.device)
        target_tensor = torch.tensor([target_tokens], device=self.device)
        
        # Calculate loss
        with torch.no_grad():
            mask = (input_tensor != self.tokenizer.vocab[self.tokenizer.pad_token]).unsqueeze(1).unsqueeze(2)
            logits = self.model(input_tensor, mask)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_tensor.view(-1),
                ignore_index=self.tokenizer.vocab[self.tokenizer.pad_token]
            )
        
        return torch.exp(loss).item()
    
    def generate_test_prompts(self) -> List[Dict]:
        """Generate test prompts for evaluation."""
        test_prompts = [
            {
                "prompt": "What is the meaning of life according to Bhagavad Gita?",
                "expected_keywords": ["dharma", "karma", "moksha", "yoga"]
            },
            {
                "prompt": "Explain the concept of karma",
                "expected_keywords": ["action", "consequence", "fruit", "desire"]
            },
            {
                "prompt": "How can one achieve inner peace?",
                "expected_keywords": ["stability", "mind", "senses", "detachment"]
            },
            {
                "prompt": "What is the nature of the self?",
                "expected_keywords": ["atman", "soul", "eternal", "unchanging"]
            },
            {
                "prompt": "How should one perform their duties?",
                "expected_keywords": ["selfless", "dedication", "attachment", "result"]
            }
        ]
        return test_prompts
    
    def evaluate_response(self, response: str, expected_keywords: List[str]) -> Dict:
        """Evaluate generated response."""
        response_lower = response.lower()
        
        keyword_matches = sum(1 for keyword in expected_keywords if keyword in response_lower)
        match_ratio = keyword_matches / len(expected_keywords)
        
        # Calculate perplexity
        ppl = self.perplexity(response)
        
        # Basic quality metrics
        word_count = len(response.split())
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        return {
            "keyword_matches": keyword_matches,
            "match_ratio": match_ratio,
            "perplexity": ppl,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": word_count / max(sentence_count, 1)
        }
    
    def run_evaluation(self, num_samples: int = 10) -> Dict:
        """Run comprehensive evaluation."""
        print("🧪 Starting evaluation...")
        
        test_prompts = self.generate_test_prompts()
        
        results = {
            "perplexity_scores": [],
            "quality_scores": [],
            "keyword_coverage": [],
            "total_samples": 0
        }
        
        for i, prompt_data in enumerate(test_prompts):
            print(f"\n📝 Testing prompt {i+1}/{len(test_prompts)}: {prompt_data['prompt']}")
            
            # Generate response
            response_tokens = self.model.generate(
                start_token=self.tokenizer.vocab[self.tokenizer.bos_token],
                max_length=100,
                temperature=0.8,
                top_k=50,
                device=self.device
            )
            
            response = self.tokenizer.decode(response_tokens)
            response = response.replace(f" {self.tokenizer.vocab_inv.get(self.tokenizer.bos_token, '')}", "")
            response = response.replace(f" {self.tokenizer.vocab_inv.get(self.tokenizer.eos_token, '')}", "")
            
            print(f"🤖 Generated response: {response[:200]}...")
            
            # Evaluate response
            metrics = self.evaluate_response(response, prompt_data['expected_keywords'])
            
            results["perplexity_scores"].append(metrics["perplexity"])
            results["quality_scores"].append(metrics["match_ratio"])
            results["keyword_coverage"].append(metrics["keyword_matches"])
            results["total_samples"] += 1
            
            print(f"📊 Metrics: PPL={metrics['perplexity']:.2f}, "
                  f"Keyword Match={metrics['keyword_matches']}/{len(prompt_data['expected_keywords'])}, "
                  f"Score={metrics['match_ratio']:.2f}")
        
        # Calculate summary statistics
        if results["total_samples"] > 0:
            results["avg_perplexity"] = np.mean(results["perplexity_scores"])
            results["avg_quality_score"] = np.mean(results["quality_scores"])
            results["avg_keyword_coverage"] = np.mean(results["keyword_coverage"])
            
            print(f"\n📈 Evaluation Summary:")
            print(f"   Average Perplexity: {results['avg_perplexity']:.2f}")
            print(f"   Average Quality Score: {results['avg_quality_score']:.2f}")
            print(f"   Average Keyword Coverage: {results['avg_keyword_coverage']:.2f}")
            print(f"   Total Samples: {results['total_samples']}")
        
        return results
    
    def save_evaluation_report(self, results: Dict, output_path: str):
        """Save evaluation report."""
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_info": {
                "vocab_size": self.tokenizer.get_vocab_size(),
                "device": str(self.device)
            },
            "results": results,
            "recommendations": self.generate_recommendations(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Evaluation report saved to {output_path}")
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if "avg_perplexity" in results and results["avg_perplexity"] > 200:
            recommendations.append("Consider increasing training epochs or adjusting model architecture")
        
        if "avg_quality_score" in results and results["avg_quality_score"] < 0.5:
            recommendations.append("Improve training data quality or increase model complexity")
        
        if "avg_keyword_coverage" in results and results["avg_keyword_coverage"] < 2:
            recommendations.append("Enhance prompt engineering or fine-tune on domain-specific data")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Evaluate GeetaGPT model')
    parser.add_argument('--model_path', type=str, default='geeta_gpt_model.pth', help='Path to trained model')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.pkl', help='Path to tokenizer')
    parser.add_argument('--output_path', type=str, default='evaluation_report.json', help='Path to save evaluation report')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = GeetaEvaluator(args.model_path, args.tokenizer_path, args.device)
    
    # Run evaluation
    results = evaluator.run_evaluation(num_samples=args.num_samples)
    
    # Save report
    evaluator.save_evaluation_report(results, args.output_path)

if __name__ == "__main__":
    main()