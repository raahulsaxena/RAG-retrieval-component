# RAG Powered Patent Analysis Tool

This repository contains the code for a **RAG (Retrieval-Augmented Generation)** powered patent analysis tool. The tool is designed to leverage document retrieval and language generation techniques to process and analyze patent documents effectively. By combining the power of Retrieval-Augmented Generation (RAG), it can extract insights, summarize patents, and perform various analyses using the content of patents from a large database.

## Features

- **Patent Document Retrieval**: Retrieve relevant patent documents based on user queries.
- **Automated Patent Summarization**: Generate concise summaries of patents.
- **Patent Comparison**: Compare multiple patents to highlight key differences and similarities.
- **Semantic Search**: Perform searches based on the meaning of the queries, not just keywords.
- **Patent Classification**: Classify patents into different categories based on their content.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/raahulsaxena/RAG-patent-analysis-tool.git
    ```

2. Navigate to the project directory:
    ```bash
    cd RAG-patent-analysis-tool
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Patent Retrieval

To retrieve relevant patents based on a query, run the following:

```bash
python retriever.py --query "Method for enhancing semiconductor performance"
```

### 2. Summarizing Patent Documents

```bash
python summarize.py --document "path_to_patent_file.pdf"
```

### 3. Patent Comparison

```bash
python compare.py --document1 "path_to_patent1.pdf" --document2 "path_to_patent2.pdf"
```

## Files
	•	retriever.py: Handles the retrieval of patent documents based on a query.
	•	summarize.py: Generates summaries of patent documents.
	•	compare.py: Compares two patent documents and highlights differences.
	•	queries.json: Contains example queries used for testing.
	•	chunks_array.npz: Stores preprocessed patent chunks for efficient retrieval.
	•	jaccard.py: Implements the Jaccard similarity for patent document comparison.
	•	sample_input.py: Provides a sample input for testing the system.

## Requirements
	•	Python 3.x
	•	transformers library for language models
	•	torch for deep learning
	•	numpy for numerical operations
	•	scikit-learn for machine learning tools
	•	Other dependencies listed in requirements.txt
