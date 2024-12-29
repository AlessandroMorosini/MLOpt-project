# Context-Aware LLMs for Financial Data: Bankruptcy Risk Prediction

### Problem Statement
Early bankruptcy prediction is vital for business valuation and impacts investors, regulators, and other stakeholders. Traditional methods relying on tabular data can be labor-intensive and overlook contextual insights. Here, we explore the use of TabText framework to address these limitations by processing the tabular data into LLM's food and running our risk-prediction analysis. 

### Key Experiments
- **Embedding Models:** Comparing FinBERT and BERT for financial embeddings.
- **Model Comparisons:** Benchmarking traditional machine learning models against contextual embedding models.
- **Handling Missing Data:** Evaluating performance with increasing levels of missing values.
- **Multi-Modality:** Evaluating performance when combining tabular data with embeddings.
- **Ensembling:** Combining the predictions from tabular and embeddings-based models to create ensembles. 

## Repository Structure

There are two main notebooks in the folder named `code`:
- **`Embeddings/`**: Generates contextual embeddings using the LLMs (Python).
- **`Main/`**: Contains the core analysis, including model comparisons and evaluations (Python).
Moreover, there are two other notebooks:
- **`Subsampling/`**: Implements problem-specific data subsampling techniques for class imbalance (Julia).
- **`OCT/`**: Explores optional classification trees (Julia).

## Results Summary

- FinBERT outperforms BERT, highlighting the value of domain-specific fine-tuning.
- XGBoost achieves higher overall accuracy, with embeddings showing limited added value in this dataset.
- TabText demonstrates robustness to missing data, halving the performance gap when data quality degrades.
- Ensemble methods further improve predictions by leveraging complementary strengths.
