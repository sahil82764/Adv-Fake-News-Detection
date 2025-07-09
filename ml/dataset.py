import torch
from torch.utils.data import Dataset
import pandas as pd

class FakeNewsDataset(Dataset):
    """
    Custom PyTorch Dataset for loading fake news data.

    This class is designed to be flexible for different model types.
    For transformer models, it tokenizes raw text on-the-fly.
    For traditional ML models, it can be used to simply load the processed text.
    """
    def __init__(self, data_path: str, tokenizer=None, max_length: int = 512, text_column: str = 'text_raw'):
        """
        Args:
            data_path (str): Path to the parquet file (e.g., train.parquet).
            tokenizer: A Hugging Face tokenizer. Required for transformer models.
            max_length (int): The maximum sequence length for tokenization.
            text_column (str): The name of the text column to use ('text_raw' or 'text_processed').
        """
        try:
            self.data = pd.read_parquet(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}. Please run preprocessing scripts.")

        # Ensure the specified text column exists
        if text_column not in self.data.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset at {data_path}.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # Pre-convert columns to lists for faster access with .iloc
        self.texts = self.data[self.text_column].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetches a single data sample. If a tokenizer is provided, it tokenizes the text.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # If no tokenizer, just return text and label (for traditional ML)
        if self.tokenizer is None:
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }

        # If tokenizer is present, perform tokenization (for transformers)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
