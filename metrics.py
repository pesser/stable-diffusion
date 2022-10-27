import sys
import evaluate

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
    print(results)