from collections import Counter
from typing import List

def extract_ngrams(text: str, n: int) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def calculate_ngram_overlap(reference: str, generated: str, n: int):
    reference_ngrams = extract_ngrams(reference, n)
    generated_ngrams = extract_ngrams(generated, n)
    
    reference_counter = Counter(reference_ngrams)
    generated_counter = Counter(generated_ngrams)
    
    overlapping_ngrams = set(reference_counter.keys()) & set(generated_counter.keys())
    
    overlap_count = sum(min(reference_counter[ngram], generated_counter[ngram]) for ngram in overlapping_ngrams)
    precision = overlap_count / len(generated_ngrams) if generated_ngrams else 0
    recall = overlap_count / len(reference_ngrams) if reference_ngrams else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    return precision, recall, f1_score

reference_text = "Natural language processing is very interesting."
generated_text = "Natural language processing is quite interesting."

# 计算Unigram（1-gram）
precision_unigram, recall_unigram, f1_unigram = calculate_ngram_overlap(reference_text, generated_text, 1)
print(f"Unigram - 精确率: {precision_unigram:.2f}, 召回率: {recall_unigram:.2f}, F1分数: {f1_unigram:.2f}")

# 计算Bigram（2-gram）
precision_bigram, recall_bigram, f1_bigram = calculate_ngram_overlap(reference_text, generated_text, 2)
print(f"Bigram - 精确率: {precision_bigram:.2f}, 召回率: {recall_bigram:.2f}, F1分数: {f1_bigram:.2f}")
