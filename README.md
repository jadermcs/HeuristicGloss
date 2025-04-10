# Constrained Decoder for Language Models

This repository contains an implementation of a constrained decoding method for language models. The constrained decoder allows you to limit the tokens a language model can generate to only the ones you explicitly allow.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The main component is the `ConstrainedDecoder` class in `src/constrained_decoder.py`. Here's how to use it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.constrained_decoder import ConstrainedDecoder

# Load model and tokenizer
model_name = "gpt2"  # Can be replaced with any HuggingFace model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create constrained decoder
decoder = ConstrainedDecoder(tokenizer)

# Define allowed vocabulary
allowed_strings = ["positive", "words", "only"]  # Add your allowed tokens here
allowed_tokens = decoder.allowed_tokens_from_strings(allowed_strings)

# Add necessary tokens like spaces, punctuation
for token in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
    if token is not None:
        allowed_tokens.append(token)

# Also add space/punctuation tokens if needed
for s in [" ", ".", ",", "!", "?"]:
    allowed_tokens.extend(tokenizer.encode(s, add_special_tokens=False))

# Generate text with constraints
prompt = "Your prompt here"
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
```

## How It Works

The constrained decoder works by manipulating the logits (probability scores) output by the language model before token selection. Here's the process:

1. We create a mask of allowed token IDs
2. During text generation, we set the logits of all disallowed tokens to negative infinity
3. This ensures that only the allowed tokens can be sampled during generation
4. The implementation supports temperature, top-k, and top-p sampling methods

This approach is compatible with any model that exposes logits during generation, including most Hugging Face transformer models. 



Guidelines:

1. **Clarity and Simplicity:**  
   The definition should be easy to understand, even for someone who doesn't know the word. Lexicographers try to use common, simple words wherever possible — this is sometimes called a *defining vocabulary*, a limited set of words used to define all others.

2. **Precision:**  
   They aim to capture exactly what makes the word unique — what separates it from similar words. This involves identifying the *essential meaning* (the genus) and the *distinguishing features* (the differentia). For example:
   - *"A **penguin** is a flightless seabird with black and white plumage, native to the Southern Hemisphere."*  
     Here, "seabird" is the genus, and the rest is the differentia.

3. **Usage Evidence:**  
   Definitions are based on real-world usage. Lexicographers consult large databases (called *corpora*) of texts to see how the word is actually used in different contexts, and then generalize from there.

4. **Consistency with Related Words:**  
   Definitions are crafted to work well with others in the dictionary. If two words are closely related, their definitions should reflect that relationship without being redundant.

5. **Avoiding Circularity:**  
   They avoid defining a word using the word itself or a close variant — that would be circular and unhelpful.

6. **Audience Consideration:**  
   The vocabulary and structure of the definition depend on the dictionary’s audience — a children’s dictionary will use simpler words than an academic one.