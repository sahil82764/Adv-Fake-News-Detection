import pandas as pd
import os
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

# --- Configuration ---
# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
KAGGLE_DIR = os.path.join(RAW_DATA_DIR, 'kaggle_fake_news')
LIAR_DIR = os.path.join(RAW_DATA_DIR, 'liar')

# --- Download NLTK data (if not already present) ---
try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK punkt tokenizer not found. Downloading...")
    nltk.download('punkt')

# --- Advanced Text Processing Functions ---
# Note: For transformer models like DistilBERT/ALBERT, heavy preprocessing like
# stopword removal and stemming is often NOT recommended as it can remove valuable context.
# These functions are included for training traditional ML models later.

def clean_text(text):
    """Basic text cleaning: lowercase, remove special characters."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    return text

def process_text_advanced(text, do_stemming=False, remove_stopwords=False):
    """Advanced text processing pipeline."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        
    if do_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]
        
    return " ".join(tokens)


def load_and_process_kaggle():
    """Loads and processes the Kaggle dataset."""
    logging.info("Processing Kaggle dataset...")
    try:
        fake_df = pd.read_csv(os.path.join(KAGGLE_DIR, 'Fake.csv'))
        true_df = pd.read_csv(os.path.join(KAGGLE_DIR, 'True.csv'))
        
        fake_df['label'] = 0
        true_df['label'] = 1
        
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        # Combine title and text for a complete context
        combined_df['text'] = combined_df['title'] + '. ' + combined_df['text']
        
        return combined_df[['text', 'label']]
    except FileNotFoundError as e:
        logging.error(f"Error loading Kaggle files: {e}. Please check the path.")
        return pd.DataFrame()

def load_and_process_liar():
    """Loads and processes the LIAR dataset."""
    logging.info("Processing LIAR dataset...")
    liar_columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
        'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
    ]
    try:
        train_df = pd.read_csv(os.path.join(LIAR_DIR, 'train.tsv'), sep='\t', names=liar_columns)
        test_df = pd.read_csv(os.path.join(LIAR_DIR, 'test.tsv'), sep='\t', names=liar_columns)
        valid_df = pd.read_csv(os.path.join(LIAR_DIR, 'valid.tsv'), sep='\t', names=liar_columns)
        
        combined_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
        
        label_mapping = {
            'true': 1, 'mostly-true': 1,
            'false': 0, 'barely-true': 0, 'pants-on-fire': 0
        }
        
        combined_df = combined_df[combined_df['label'].isin(label_mapping.keys())]
        combined_df['label'] = combined_df['label'].map(label_mapping)
        combined_df.rename(columns={'statement': 'text'}, inplace=True)
        
        return combined_df[['text', 'label']]
    except FileNotFoundError as e:
        logging.error(f"Error loading LIAR files: {e}. Please check the path.")
        return pd.DataFrame()

def main():
    """Main function to run the enhanced data preprocessing pipeline."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    log_file_path = os.path.join(LOGS_DIR, 'preprocess.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()]
    )

    logging.info("--- Starting Enhanced Data Harmonization ---")
    
    kaggle_data = load_and_process_kaggle()
    liar_data = load_and_process_liar()

    if kaggle_data.empty and liar_data.empty:
        logging.warning("No data could be loaded. Exiting.")
        return

    final_data = pd.concat([kaggle_data, liar_data], ignore_index=True)
    logging.info(f"Combined dataset shape before cleaning: {final_data.shape}")

    # --- Basic Cleaning ---
    final_data.dropna(subset=['text', 'label'], inplace=True)
    final_data.drop_duplicates(subset=['text'], inplace=True)
    final_data['label'] = final_data['label'].astype(int)
    logging.info(f"Shape after basic cleaning and deduplication: {final_data.shape}")

    # --- Create two versions of the text data ---
    # 1. 'text_raw': Minimally cleaned text, ideal for Transformer models
    # 2. 'text_processed': Heavily processed text for traditional ML models
    logging.info("Creating raw text version for Transformers...")
    final_data['text_raw'] = final_data['text'].apply(clean_text)

    logging.info("Creating processed text version for traditional ML models (with stemming)...")
    final_data['text_processed'] = final_data['text'].apply(
        lambda x: process_text_advanced(x, do_stemming=True, remove_stopwords=True)
    )
    
    # Shuffle the dataset
    final_data = final_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select final columns to save
    output_df = final_data[['text_raw', 'text_processed', 'label']]

    # Save to a single, efficient file format
    output_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.parquet')
    output_df.to_parquet(output_path, index=False)
    
    logging.info(f"Final dataset shape: {output_df.shape}")
    logging.info(f"Label distribution:\n{output_df['label'].value_counts(normalize=True)}")
    logging.info(f"Successfully saved processed data to {output_path}")

if __name__ == '__main__':
    main()
