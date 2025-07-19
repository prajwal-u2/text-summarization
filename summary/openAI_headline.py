import pandas as pd
import openai
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download necessary NLTK data
nltk.download('punkt')

# Set your OpenAI API key (make sure to set your own key here)
openai.api_key = ""
# Load your dataset
df = pd.read_csv("text-summarization/datasets/business_data.csv")
df = df.head(10)  # For testing, limit to the first 10 rows

# Function to get GPT summary (new API)
def gpt_summary(text):
    """Generates a summary using GPT."""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ],
        max_tokens=200,  # Adjust max_tokens based on your requirement
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

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

# Apply summarization using GPT on the content column
df['generated_summary'] = df['content'].apply(lambda x: gpt_summary(x))

# Calculate ROUGE and BLEU scores for each generated summary
df['rouge_scores'] = df.apply(lambda x: compute_rouge(x['description'], x['generated_summary']), axis=1)
df['bleu_scores'] = df.apply(lambda x: compute_bleu(x['description'], x['generated_summary']), axis=1)

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

# Save the results (dataframe + average metrics) into CSV files
df.to_csv('summarized_results_gpt.csv', index=False)
df_avg.to_csv('summarized_average_metrics_gpt.csv', index=False)

print("Results have been saved to 'summarized_results_gpt.csv' and 'summarized_average_metrics_gpt.csv'.")
