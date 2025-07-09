import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional
# For precise type hinting of the tokenizer object
from transformers import PreTrainedTokenizerBase

class FakeNewsDataset(Dataset):
    """
    Custom PyTorch Dataset for loading fake news data.

    This class is designed to be flexible for different model types.
    For transformer models, it tokenizes raw text on-the-fly.
    For traditional ML models, it can be used to simply load the processed text.
    """
    def __init__(self, data_path: str, tokenizer: Optional[PreTrainedTokenizerBase] = None, max_length: int = 512, text_column: str = 'text_raw'):
        """
        Args:
            data_path (str): Path to the parquet file (e.g., train.parquet).
            tokenizer (Optional[PreTrainedTokenizerBase]): A Hugging Face tokenizer.
                If provided, text is tokenized for transformer models.
                If None, raw text is returned, suitable for sklearn pipelines.
            max_length (int): The maximum sequence length for tokenization.
            text_column (str): The name of the text column to use
                ('text_raw' or 'text_processed').
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
        
        # Pre-convert columns to lists for faster __getitem__ access.
        # This improves speed but increases memory usage as data is duplicated.
        # For extremely large datasets, consider accessing the dataframe directly.
        self.texts = self.data[self.text_column].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetches a single data sample.

        If a tokenizer is provided, it tokenizes the text and returns tensors
        suitable for a PyTorch DataLoader.

        If no tokenizer is provided, it returns the raw text and a label tensor.
        This is intended for use with vectorizers like sklearn's TfidfVectorizer,
        not for direct use in a standard PyTorch DataLoader which cannot batch raw strings.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # If no tokenizer, return raw text and label (for traditional ML pipelines)
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
