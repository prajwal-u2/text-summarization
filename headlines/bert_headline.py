import pandas as pd
from summarizer import Summarizer
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download necessary NLTK data
nltk.download('punkt')

# Load your dataset
df = pd.read_csv("text-summarization/datasets/business_data.csv")
#df = df.head(10)  # For testing, limit to the first 10 rows

# Create a BERT extractive summarizer
summarizer = Summarizer()

# Function to compute ROUGE scores
def compute_rouge(reference, hypothesis):
    """Computes ROUGE scores for a given reference and hypothesis."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

# Function to compute BLEU score
def compute_bleu(reference, hypothesis):
    """Computes BLEU score for a given reference and hypothesis."""
    reference = reference.split()  # Tokenize
    hypothesis = hypothesis.split()  # Tokenize
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu([reference], hypothesis, smoothing_function=smoothing_function)

# Apply summarization on the content column
df['generated_summary'] = df['content'].apply(lambda x: summarizer(x, min_length=10, max_length=30))

# Calculate ROUGE and BLEU scores for each generated summary
df['rouge_scores'] = df.apply(lambda x: compute_rouge(x['headlines'], x['generated_summary']), axis=1)
df['bleu_scores'] = df.apply(lambda x: compute_bleu(x['headlines'], x['generated_summary']), axis=1)

# Extract ROUGE and BLEU scores
df['rouge_1'] = df['rouge_scores'].apply(lambda x: x[0])
df['rouge_2'] = df['rouge_scores'].apply(lambda x: x[1])
df['rouge_L'] = df['rouge_scores'].apply(lambda x: x[2])

df['bleu'] = df['bleu_scores']

# Calculate average ROUGE and BLEU scores for the entire dataset
rouge_1_avg = df['rouge_1'].mean()
rouge_2_avg = df['rouge_2'].mean()
rouge_L_avg = df['rouge_L'].mean()

bleu_avg = df['bleu'].mean()

# Output the evaluation metrics
print(f'Average ROUGE-1 score: {rouge_1_avg}')
print(f'Average ROUGE-2 score: {rouge_2_avg}')
print(f'Average ROUGE-L score: {rouge_L_avg}')
print(f'Average BLEU score: {bleu_avg}')

# Add average scores to a new row for overall metrics
df_avg = pd.DataFrame({
    'rouge_1': [rouge_1_avg],
    'rouge_2': [rouge_2_avg],
    'rouge_L': [rouge_L_avg],
    'bleu': [bleu_avg]
})

# Save the results (dataframe + average metrics) into a CSV file
df.to_csv('headline_results_bert.csv', index=False)
df_avg.to_csv('headline_average_metrics_bert.csv', index=False)

print("Results have been saved to 'summarized_results.csv' and 'summarized_average_metrics.csv'.")
