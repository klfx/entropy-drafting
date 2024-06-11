#!/usr/bin/env python
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_utils import CustomGenerationMixin
import torch
import time

parser = argparse.ArgumentParser(description='Transformers Text-generation with Entropy Drafting')
parser.add_argument('--target', type=str, help='Target Model')
parser.add_argument('--draft', type=str, help='Draft Model')
parser.add_argument('--input', type=str, help='Input Text')
parser.add_argument('--max_new_tokens', type=int, default=100, help='Max New Tokens to generate')
parser.add_argument('--entropy_drafting', type=bool, default=True, help='Use Entropy Drafting')
parser.add_argument('--entropy_drafting_type', type=str, default='static', help='Entropy Drafting Type')

args = parser.parse_args()

def _extend(instance, new_class):
    instance.__class__ = type(
        '%s_extended_with_%s' % (instance.__class__.__name__, new_class.__name__), 
        (instance.__class__, new_class), 
        {},
        ) 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    class CustomAutoModelForCausalLM(AutoModelForCausalLM, CustomGenerationMixin):
        pass

    model = AutoModelForCausalLM.from_pretrained(args.target).to(device)
    model_draft = AutoModelForCausalLM.from_pretrained(args.draft).to(device)
    _extend(model, CustomGenerationMixin)

    early_termination_params = {
    "static_threshold_val" : 2.25,
    "ma_m" : 0.5,
    "ma_last_n" : 7,
    "cum_threshold_val" : 10,
    "cum_last_n" : 7,
    }

    if args.entropy_drafting_type == 'static':
        early_termination_params['type'] = 'static'
    elif args.entropy_drafting_type == 'ma':
        early_termination_params['type'] = 'ma'
    elif args.entropy_drafting_type == 'cum':
        early_termination_params['type'] = 'cum'
    else:
        raise ValueError("Invalid Entropy Drafting Type")

    inputs = tokenizer(args.input, return_tensors="pt").to(device)

    start_time = time.time()
    if args.entropy_drafting:
        tokens, dml, tml = model.generate(
            **inputs,
            assistant_model=model_draft,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=6,
        )

    else:
        tokens, dml, tml = model.generate(
            **inputs,
            assistant_model=model_draft,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=6,
            num_k=10,
            early_termination=True,
            early_termination_params=early_termination_params
        )
    end_time = time.time()

    gen_time = end_time - start_time

    print(f"Generated Text: {tokenizer.batch_decode(tokens, skip_special_tokens=True)}")
    print(f"\nGeneration Time: {gen_time:.2f} seconds")



if __name__ == "__main__":
    main()