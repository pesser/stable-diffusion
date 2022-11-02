

import os
import sys
import evaluate
from tqdm import tqdm
from multiprocessing import Pool
import nltk
from nltk.translate.bleu_score import SmoothingFunction

class SelfBleu:
    def __init__(self, test_text='', gram=3):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for i, hypothesis in tqdm(enumerate(test_data)):
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference[:i] + reference[i+1:], hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

def calc_div(lines, n=4):
    num_ngrams, num_words, score = 0, 0, 0
    for line in lines:
        ngrams = []
        line = nltk.word_tokenize(line)
        for i in range(len(line)-n+1):
            ngram = line[i:i+n]
            if not ngram in ngrams:
                ngrams.append(ngram)
        num_ngrams += len(ngrams)
        num_words += len(line)
        score += len(ngrams) / len(line)
    score /= len(lines)
    return num_ngrams / num_words, score


if __name__ == "__main__":
    metric = sys.argv[1]
    src, dst = sys.argv[2], sys.argv[3]
    
    with open(src, 'r') as src_f:
        src_lines = [line.strip() for line in src_f][:10]
    with open(dst, 'r') as dst_f:
        dst_lines = [line.strip() for line in dst_f][:10]
    
    metric = metric.lower()
    print(f"testing with {metric}")
    if metric == 'bleu':
        bleu = evaluate.load('bleu')
        results = bleu.compute(predictions=dst_lines, references=src_lines)
    elif metric == 'rouge':
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=dst_lines, references=src_lines)
    elif metric == 'bertscore':
        bertscore = evaluate.load('bertscore')
        results = bertscore.compute(predictions=dst_lines, references=src_lines, lang='en')
        # results = bertscore.compute(predictions=dst_lines, references=src_lines, model_type='bert-base-uncased')
    elif metric == 'selfbleu':
        selfbleu = SelfBleu(src, gram=5)
        results = selfbleu.get_bleu()
    elif metric == 'div-4':
        results = calc_div(src_lines, n=4)
    print(results)