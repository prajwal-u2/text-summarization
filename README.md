# Text Summarization Project

A comprehensive comparison and analysis of different text summarization algorithms including BERT, T5, and TextRank. This project evaluates the performance of extractive and abstractive summarization techniques on various datasets.

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements and compares three popular text summarization algorithms:

- **BERT (Bidirectional Encoder Representations from Transformers)**: Extractive summarization using transformer-based embeddings
- **T5 (Text-to-Text Transfer Transformer)**: Abstractive summarization using sequence-to-sequence models
- **TextRank**: Graph-based extractive summarization using PageRank algorithm

The project includes both summary generation and headline generation capabilities, with comprehensive evaluation metrics and analysis.

## 🤖 Algorithms

### 1. BERT Summarization
- **Type**: Extractive
- **Approach**: Uses BERT embeddings to identify and extract the most important sentences
- **Advantages**: Contextual understanding, high accuracy
- **Use Case**: Document summarization, content extraction

### 2. T5 Summarization
- **Type**: Abstractive
- **Approach**: Sequence-to-sequence model that generates new text
- **Advantages**: Can create novel summaries, better coherence
- **Use Case**: Creative summarization, headline generation

### 3. TextRank
- **Type**: Extractive
- **Approach**: Graph-based algorithm using sentence similarity and PageRank
- **Advantages**: No training required, interpretable
- **Use Case**: Quick summarization, keyword extraction

## 📁 Project Structure

```
text-summarization/
├── code/                    # Core algorithm implementations
│   ├── bert.py             # BERT-based summarization
│   ├── T5.py               # T5-based summarization
│   └── TextRank.py         # TextRank algorithm
├── datasets/               # Data files and download scripts
│   ├── business_data.csv   # Business articles dataset
│   ├── train.csv          # Training data
│   ├── test.csv           # Test data
│   ├── validation.csv     # Validation data
│   └── download-dataset.py # Dataset download script
├── summary/               # Summary generation scripts
│   ├── bert_summary.py    # BERT summary evaluation
│   ├── openAI_headline.py # OpenAI headline generation
│   └── output/            # Summary results and metrics
├── headlines/             # Headline generation scripts
│   ├── bert_headline.py   # BERT headline generation
│   └── output/            # Headline results and metrics
└── README.md              # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies

Install the required packages:

```bash
pip install transformers
pip install torch
pip install sentence-transformers
pip install summarizer
pip install nltk
pip install gensim
pip install networkx
pip install scipy
pip install numpy
pip install pandas
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd text-summarization
```

2. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 💻 Usage

### Basic Usage

#### BERT Summarization
```python
from code.bert import Summarizer

summarizer = Summarizer()
summary = summarizer(input_text, min_length=50, max_length=1500)
```

#### T5 Summarization
```python
from code.T5 import summarize_text

summary = summarize_text(input_text, max_length=1000, min_length=500)
```

#### TextRank Summarization
```python
# Run the TextRank script directly
python code/TextRank.py
```

### Advanced Usage

#### Running Summary Evaluation
```bash
cd summary
python bert_summary.py
```

#### Running Headline Generation
```bash
cd headlines
python bert_headline.py
```

## 📊 Datasets

The project uses several datasets for evaluation:

- **Business Data**: Articles from business domain
- **Training Data**: Large-scale training dataset (1.2GB)
- **Test Data**: Evaluation dataset (48MB)
- **Validation Data**: Validation dataset (55MB)

### Downloading Datasets

Use the provided download script:
```bash
cd datasets
python download-dataset.py
```

## 📈 Results

The project generates comprehensive evaluation metrics including:

- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU Scores**: Bilingual Evaluation Understudy
- **METEOR Scores**: Metric for Evaluation of Translation with Explicit ORdering
- **BERTScore**: Contextual embeddings-based evaluation

Results are stored in the respective `output/` directories with detailed analysis and comparisons.

## 🔧 Customization

### Adjusting Summary Length
```python
# For BERT
summary = summarizer(text, min_length=100, max_length=500)

# For T5
summary = summarize_text(text, max_length=800, min_length=200)
```

### Model Selection
```python
# For T5, choose different model sizes
model_name = "t5-small"    # Fast, smaller
model_name = "t5-base"     # Balanced
model_name = "t5-large"    # High quality, slower
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the transformers library
- Google Research for T5 model
- Stanford NLP for TextRank algorithm
- The summarizer library contributors

---

**Note**: This project is for educational and research purposes. Please ensure you have the necessary permissions when using copyrighted content for summarization.