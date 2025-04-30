# Fake-News-Detection
Supervised learning project created computationally classify news articles as either "Real" or "Fake" based on their text content using Natural Language Processing (NLP) and Machine Learning techniques.

**Classifies news articles as real or fake using NLP (TF-IDF) and Machine Learning models (Neural Network, XGBoost).**

## Description

The proliferation of misinformation ("fake news") online, particularly through social media, poses a significant challenge. This project explores computational methods to automatically classify news articles based on their textual content. By analyzing patterns in the language used, we train models to distinguish between legitimate and fabricated news reports. The project utilizes TF-IDF for feature extraction and compares the performance of a Sequential Neural Network (built with Keras/TensorFlow) and an XGBoost classifier.

## Key Features

*   **Data Loading & Preprocessing:** Loads data from `True.csv` and `Fake.csv` files. Preprocesses text data by converting to lowercase, removing punctuation, and filtering out common English stopwords.
*   **Feature Extraction:** Uses Term Frequency-Inverse Document Frequency (TF-IDF) to convert text into numerical feature vectors suitable for machine learning models (`TfidfVectorizer` from Scikit-learn).
*   **Model Training:** Implements and trains two distinct classification models:
    *   A Sequential Neural Network (using Keras/TensorFlow).
    *   An XGBoost Classifier.
*   **Model Evaluation:** Assesses model performance using:
    *   Accuracy Score
    *   Precision, Recall, F1-Score
    *   Classification Report
    *   Confusion Matrix (visualized using Seaborn/Matplotlib).
*   **Model & Vectorizer Persistence:** Includes code to save the trained models and the TF-IDF vectorizer using `pickle` and `joblib` for potential future use or deployment.
*   **Visualization:** Includes a WordCloud visualization to show prominent words in the dataset.

## Technologies Used

*   Python 3.x
*   Jupyter Notebook
*   **Libraries:**
    *   Pandas (Data manipulation)
    *   NLTK (NLP tasks like tokenization, stopwords)
    *   Scikit-learn (TF-IDF, train/test split, metrics)
    *   TensorFlow / Keras (Neural Network implementation)
    *   XGBoost (Gradient Boosting classifier)
    *   Seaborn / Matplotlib (Data visualization)
    *   WordCloud (Word frequency visualization)
    *   Joblib / Pickle (Model and vectorizer saving/loading)
    *   BeautifulSoup4 (Potentially for advanced text cleaning, though primary use not shown in snippet)

## Dataset

This project uses two CSV files: `True.csv` and `Fake.csv`.
*   `True.csv`: Contains text from legitimate news articles.
*   `Fake.csv`: Contains text from fake or fabricated news articles.
Each file typically contains columns like 'title', 'text', 'subject', and 'date'. For modeling, this project primarily uses the 'text' column and assigns a 'category' label (1 for True, 0 for Fake).


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/amr-x5/Fake-News-Detection.git>
    cd <Materials>
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    Make sure you have a `requirements.txt` file (you'll need to create this, see below) and run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data:**
    The notebook includes `nltk.download('stopwords')`. Run this in a Python interpreter or ensure the notebook executes this cell successfully.
    ```bash
    python -m nltk.downloader stopwords
    ```
5.  **Obtain Dataset:** Download `True.csv` and `Fake.csv` and place them in the root project directory (or modify the loading paths in the notebook).
