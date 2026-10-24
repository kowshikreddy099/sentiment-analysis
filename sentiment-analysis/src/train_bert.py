"""Fine-tune BERT / DistilBERT / RoBERTa on reviews.
Calls the updated multi-domain transformer training engine.
"""
import os, sys

if __name__ == "__main__":
    from train_transformer import main
    main()
