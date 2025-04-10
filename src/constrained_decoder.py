import torch
import numpy as np
from typing import List, Optional, Union, Tuple

class ConstrainedDecoder:
    """
    A decoder that constrains language model outputs to only allowed tokens.
    This can be used with any transformer-based language model that uses logits for token prediction.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the constrained decoder.
        
        Args:
            tokenizer: The tokenizer used by the language model
        """
        self.tokenizer = tokenizer
        
    def constrain_logits(
        self, 
        logits: torch.Tensor,
        allowed_tokens: List[int],
        disallowed_tokens: Optional[List[int]] = None,
        temperature: float = 1.0,
        filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """
        Constrains the logits to only allow specific tokens.
        
        Args:
            logits: The model's output logits of shape (batch_size, vocab_size)
            allowed_tokens: List of token IDs that are allowed for generation
            disallowed_tokens: Optional list of token IDs that should be explicitly disallowed
            temperature: Temperature for softmax sampling
            filter_value: Value to use for disallowed tokens (default: -Inf)
            
        Returns:
            Modified logits with only allowed tokens having non-filter values
        """
        if temperature != 1.0:
            logits = logits / temperature
            
        # Create a mask tensor for allowed tokens
        mask = torch.ones_like(logits) * filter_value
        
        # Set allowed tokens in the mask to their original logit values
        for token_id in allowed_tokens:
            mask[:, token_id] = logits[:, token_id]
            
        # Further filter out explicitly disallowed tokens
        if disallowed_tokens is not None:
            for token_id in disallowed_tokens:
                mask[:, token_id] = filter_value
                
        return mask
    
    def allowed_tokens_from_strings(self, allowed_strings: List[str]) -> List[int]:
        """
        Convert a list of allowed strings to their token IDs.
        
        Args:
            allowed_strings: List of strings that are allowed for generation
            
        Returns:
            List of token IDs corresponding to the allowed strings
        """
        token_ids = []
        for string in allowed_strings:
            ids = self.tokenizer.encode(string, add_special_tokens=False)
            token_ids.extend(ids)
        return list(set(token_ids))  # Remove duplicates
    
    def generate_with_constraints(
        self,
        model,
        input_ids: torch.Tensor,
        allowed_tokens: List[int],
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate text using the model with token constraints.
        
        Args:
            model: The language model
            input_ids: Input token IDs
            allowed_tokens: List of token IDs that are allowed
            max_length: Maximum length of generated sequence
            temperature: Temperature for sampling
            top_k: If specified, only sample from the top k most likely tokens
            top_p: If specified, sample from the smallest set of tokens whose cumulative probability exceeds p
            num_return_sequences: Number of sequences to return
            
        Returns:
            List of generated text sequences
        """
        # Prepare logits processor function for constrained decoding
        def logits_processor(logits):
            constrained_logits = self.constrain_logits(
                logits, allowed_tokens, temperature=temperature
            )
            
            if top_k is not None:
                # Keep only top k tokens
                values, _ = torch.topk(constrained_logits, top_k)
                min_values = values[:, -1].unsqueeze(-1).expand_as(constrained_logits)
                constrained_logits = torch.where(
                    constrained_logits < min_values,
                    torch.ones_like(constrained_logits) * -float("Inf"),
                    constrained_logits
                )
                
            if top_p is not None:
                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(constrained_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create the mask
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                constrained_logits = constrained_logits.masked_fill(indices_to_remove, -float("Inf"))
                
            return constrained_logits
        
        # Use the model's generate function with our custom logits processor
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            logits_processor=logits_processor,
            **kwargs
        )
        
        # Decode the outputs
        outputs = []
        for ids in output_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            outputs.append(text)
            
        return outputs

# Example usage
def example():
    """
    Example usage of the constrained decoder.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer
    model_name = "gpt2"  # Can be replaced with any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create constrained decoder
    decoder = ConstrainedDecoder(tokenizer)
    
    # Define allowed vocabulary (e.g., only positive words)
    allowed_strings = [
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "happy", "joy", "excited", "positive", "beautiful", "love"
    ]
    
    # Convert to token IDs
    allowed_tokens = decoder.allowed_tokens_from_strings(allowed_strings)
    
    # Add necessary tokens like spaces, punctuation
    for token in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if token is not None:
            allowed_tokens.append(token)
    
    # Also add space/punctuation tokens if needed
    for s in [" ", ".", ",", "!", "?"]:
        allowed_tokens.extend(tokenizer.encode(s, add_special_tokens=False))
    
    # Generate text with constraints
    prompt = "Today I feel"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    outputs = decoder.generate_with_constraints(
        model,
        input_ids,
        allowed_tokens,
        max_length=30,
        temperature=0.7,
        num_return_sequences=3
    )
    
    print("Generated sequences:")
    for output in outputs:
        print(f"- {output}")

if __name__ == "__main__":
    example() 