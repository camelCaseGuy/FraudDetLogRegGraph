# Fraud Detection Using Logistic Regression and Graph Analytics
## Project Goal
Build a machine learning pipeline to detect fraudulent transactions using logistic regression.
Leverage graph analytics to identify relationships between transactions and users (e.g., fraud rings).
Present findings with clear code, visualizations, and a structured GitHub repository.
## Dataset
### Credit Card Fraud Detection Dataset
- Transaction data with features and labels (fraudulent or not).
- Enrich data by creating a synthetic graph structure that connects users, transactions, and merchants.
## Tools and Libraries
Neo4j: To model transaction relationships and perform graph queries.
Apache Spark & GraphFrames: For distributed graph computation.
Python:
- Pandas: Data manipulation.
- Scikit-learn: Logistic regression implementation.
- Matplotlib/Seaborn: Data visualization.
- Py2Neo: To interact with Neo4j from Python.
## Workflow
### Data Preparation
Load Dataset:
- Use Python to load the Kaggle dataset or synthetic data.
- Create Graph Structure:
    - Add relationships between transactions, users, and merchants (e.g., User A → Transaction 1 → Merchant X).
    - Export graph data to Neo4j and Apache Spark.
### Neo4j Graph Analysis
- Import graph data into Neo4j.
Nodes: Users, Transactions, Merchants.
Relationships: User-Transaction, Transaction-Merchant.
- Run graph queries to detect suspicious patterns:
- Identify densely connected nodes (potential fraud rings).
- Calculate centrality or community detection for suspicious users.
- Export graph features (e.g., degree centrality, clustering coefficient) to use in the logistic regression model.
### GraphFrames with Spark
Use GraphFrames to:
- Compute additional graph metrics (e.g., PageRank, shortest paths).
- Identify fraudulent clusters in the graph.
- Join these features with transaction data.
### Logistic Regression Model
Preprocess Data:
- Combine graph features and transaction features.
- Normalize/scale features as needed.
Train Logistic Regression:
- Use Scikit-learn to train the model on labeled data.
- Perform hyperparameter tuning using GridSearchCV.
Evaluate Model:
- Metrics: Accuracy, Precision, Recall, F1-score.
- Visualize the confusion matrix and feature importance.
### Results Visualization
Use Matplotlib/Seaborn to:
- Plot graph structures with fraudulent nodes highlighted.
- Display feature importance and model evaluation metrics.
## Repository Structure
fraud-detection-graph-logistic-regression/
├── data/
│   ├── raw_data.csv           # Original dataset
│   ├── graph_data.csv         # Enriched graph data
│   ├── processed_data.csv     # Preprocessed data for ML
├── notebooks/
│   ├── data_preprocessing.ipynb  # Data preparation and graph creation
│   ├── neo4j_analysis.ipynb      # Graph analysis with Neo4j
│   ├── spark_graphframes.ipynb   # Spark GraphFrames analysis
│   ├── logistic_regression.ipynb # Logistic regression and evaluation
├── src/
│   ├── neo4j_queries.py       # Scripts for interacting with Neo4j
│   ├── spark_graph_analysis.py # Spark GraphFrames operations
│   ├── model_training.py      # Logistic regression model training
├── results/
│   ├── visualizations/        # Graphs and evaluation metrics
├── README.md                  # Project description
├── requirements.txt           # Dependencies
└── LICENSE

Project Title and Description:
"Fraud Detection Using Logistic Regression and Graph Analytics."
Brief description of the project and its goals.
Tech Stack:
Python, Neo4j, GraphFrames for Spark, Scikit-learn.
Dataset:
Mention the dataset source and how it's used.
Instructions:
Step-by-step guide to run the project.
Results:
Add screenshots of visualizations and performance metrics.
Future Work:
Ideas to extend the project (e.g., deep learning models, real-time analysis).
## GitHub
Notebooks:
Jupyter notebooks with clear annotations.
Data preparation, Neo4j queries, graph analysis, and ML modeling.
Visualizations:
- Include images of graph analysis (e.g., fraud rings) and ML evaluation metrics in the results folder.
Interactive Demo:
- Interactive visualization (Streamlit or Flask)
