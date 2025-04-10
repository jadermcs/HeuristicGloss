#!/usr/bin/env python3
"""
Example script demonstrating the constrained decoder.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from .constrained_decoder import ConstrainedDecoder
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate text with constrained decoder")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--prompt", type=str, default="Today I feel", help="Prompt for generation")
    parser.add_argument("--max-length", type=int, default=30, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num-sequences", type=int, default=3, help="Number of sequences to generate")
    parser.add_argument("--allowed-words", type=str, nargs="+", 
                        default=["good", "great", "excellent", "amazing", "wonderful", 
                                "fantastic", "happy", "joy", "excited", "positive", 
                                "beautiful", "love"],
                        help="List of allowed words")
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Create constrained decoder
    decoder = ConstrainedDecoder(tokenizer)
    
    # Convert allowed words to token IDs
    print(f"Constraining to allowed words: {args.allowed_words}")
    allowed_tokens = decoder.allowed_tokens_from_strings(args.allowed_words)
    
    # Add necessary tokens like spaces, punctuation
    for token in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if token is not None:
            allowed_tokens.append(token)
    
    # Also add space/punctuation tokens if needed
    for s in [" ", ".", ",", "!", "?"]:
        allowed_tokens.extend(tokenizer.encode(s, add_special_tokens=False))
    
    print(f"Total allowed tokens: {len(allowed_tokens)}")
    
    # Generate text with constraints
    print(f"Generating from prompt: '{args.prompt}'")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    
    outputs = decoder.generate_with_constraints(
        model,
        input_ids,
        allowed_tokens,
        max_length=args.max_length,
        temperature=args.temperature,
        num_return_sequences=args.num_sequences
    )
    
    print("\nGenerated sequences:")
    for i, output in enumerate(outputs, 1):
        print(f"{i}. {output}")

if __name__ == "__main__":
    main() 