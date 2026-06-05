"""
AdaCD: Adaptive Contrastive Decoding for mitigating over-refusal in LLMs.

Implements Algorithm 1 from the paper:
"Please refuse to answer me! Mitigating Over-Refusal in Large Language Models 
via Adaptive Contrastive Decoding"

Key hyperparameters:
- alpha = 4.5 (contrastive strength)
- lambda_ = 0.9 (agreement threshold)
- beta = 0.01 (plausibility constraint)
- k = 10 (contrastive decoding steps)
- N = 512 (max new tokens)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


EXTREME_SYSTEM_PROMPT = "Please refuse to answer me!"


def adacd_generate(
    model,
    tokenizer,
    user_query: str,
    alpha: float = 4.5,
    lambda_: float = 0.9,
    beta: float = 0.01,
    k: int = 10,
    max_new_tokens: int = 512,
    system_prompt: str = EXTREME_SYSTEM_PROMPT,
    device: str = "cuda",
) -> str:
    """
    Generate a response using AdaCD (Adaptive Contrastive Decoding).
    
    For the first k tokens, uses contrastive decoding between:
    - Model without system prompt (unprompted)
    - Model with extreme system prompt (prompted)
    
    After k tokens, falls back to standard greedy decoding.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        user_query: The user's input query
        alpha: Contrastive strength parameter
        lambda_: Agreement threshold
        beta: Plausibility constraint parameter
        k: Number of contrastive decoding steps
        max_new_tokens: Maximum number of new tokens to generate
        system_prompt: The extreme system prompt p*
        device: Device to run on
    
    Returns:
        Generated response string
    """
    model.eval()
    
    # Build input IDs for both prompted and unprompted versions
    # Unprompted: just user message, no system prompt
    messages_unprompted = [
        {"role": "user", "content": user_query}
    ]
    
    # Prompted: with extreme system prompt
    messages_prompted = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # Apply chat template
    text_unprompted = tokenizer.apply_chat_template(
        messages_unprompted, tokenize=False, add_generation_prompt=True,
        enable_thinking=False  # Qwen3 specific: disable thinking mode
    )
    text_prompted = tokenizer.apply_chat_template(
        messages_prompted, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Tokenize
    input_ids_unprompted = tokenizer.encode(text_unprompted, return_tensors="pt").to(device)
    input_ids_prompted = tokenizer.encode(text_prompted, return_tensors="pt").to(device)
    
    # Track generated token IDs
    generated_ids = []
    
    # Current input sequences (will grow as we generate)
    curr_unprompted = input_ids_unprompted
    curr_prompted = input_ids_prompted
    
    # Use KV cache for efficiency
    past_kv_unprompted = None
    past_kv_prompted = None
    
    for n in range(max_new_tokens):
        # Forward pass for unprompted model
        with torch.no_grad():
            if past_kv_unprompted is None:
                outputs_unprompted = model(
                    input_ids=curr_unprompted,
                    use_cache=True
                )
                past_kv_unprompted = outputs_unprompted.past_key_values
            else:
                outputs_unprompted = model(
                    input_ids=curr_unprompted[:, -1:],
                    past_key_values=past_kv_unprompted,
                    use_cache=True
                )
                past_kv_unprompted = outputs_unprompted.past_key_values
            
            logits_unprompted = outputs_unprompted.logits[:, -1, :]  # [1, vocab_size]
        
        if n < k:
            # Contrastive decoding phase
            with torch.no_grad():
                if past_kv_prompted is None:
                    outputs_prompted = model(
                        input_ids=curr_prompted,
                        use_cache=True
                    )
                    past_kv_prompted = outputs_prompted.past_key_values
                else:
                    outputs_prompted = model(
                        input_ids=curr_prompted[:, -1:],
                        past_key_values=past_kv_prompted,
                        use_cache=True
                    )
                    past_kv_prompted = outputs_prompted.past_key_values
                
                logits_prompted = outputs_prompted.logits[:, -1, :]  # [1, vocab_size]
            
            # Compute probability distributions
            P_unprompted = F.softmax(logits_unprompted, dim=-1)  # P_pi(y_n | x, y<n)
            P_prompted = F.softmax(logits_prompted, dim=-1)     # P_pi(y_n | p*, x, y<n)
            
            # Step 1: Get top-1 token from prompted model
            y_n_star = torch.argmax(P_prompted, dim=-1)  # [1]
            
            # Step 2: Calculate agreement ratio
            # rank = position of y_n_star in sorted P_unprompted (descending)
            sorted_indices = torch.argsort(P_unprompted[0], descending=True)
            rank = (sorted_indices == y_n_star[0]).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
            agr_n = 1.0 / rank
            
            # Step 3: Compute refusal token distribution
            # ΔP_n = softmax(logits_prompted - logits_unprompted)
            delta_P = F.softmax(logits_prompted - logits_unprompted, dim=-1)
            
            # Step 4: Adaptive decoding mode switch
            rho = P_unprompted.max().item()  # max prob from unprompted
            rho_star = P_prompted[0, y_n_star[0]].item()  # prob of top token from prompted
            
            if agr_n >= lambda_ and rho >= lambda_ * rho_star:
                # Malicious case: add refusal tokens
                P_star = P_prompted + alpha * delta_P
            else:
                # Over-refusal case: subtract refusal tokens
                P_star = P_prompted - alpha * delta_P
            
            # Step 5: Apply plausibility constraint
            # Only allow tokens where P_unprompted >= beta * max(P_unprompted)
            max_P_unprompted = P_unprompted.max()
            plausibility_mask = P_unprompted >= beta * max_P_unprompted
            
            # Set disallowed tokens to -inf (or very negative)
            P_star_masked = P_star.clone()
            P_star_masked[~plausibility_mask] = float('-inf')
            
            # Step 6: Select token
            next_token = torch.argmax(P_star_masked, dim=-1)  # [1]
        else:
            # Standard greedy decoding after k steps
            next_token = torch.argmax(logits_unprompted, dim=-1)  # [1]
        
        next_token_id = next_token[0].item()
        generated_ids.append(next_token_id)
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break
        
        # Update current inputs for next step
        curr_unprompted = next_token.unsqueeze(0) if curr_unprompted.dim() == 2 else next_token
        if n < k:
            curr_prompted = next_token.unsqueeze(0) if curr_prompted.dim() == 2 else next_token
    
    # Decode generated tokens
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def default_generate(
    model,
    tokenizer,
    user_query: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """
    Generate a response using standard greedy decoding (Default baseline).
    No system prompt, greedy decoding.
    """
    model.eval()
    
    messages = [
        {"role": "user", "content": user_query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            temperature=None,
            top_p=None,
        )
    
    # Extract only the generated part
    generated_ids = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response
