import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# This ensures the script can be run from anywhere by adding the project root to the path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.dataset import FakeNewsDataset
from ml import config
from ml.utils import compute_metrics, save_json_results

def train_transformer(model_key: str):
    """
    Trains, evaluates, and saves a transformer model based on the project config.
    """
    # --- 1. Setup ---
    model_name = config.TRANSFORMER_CONFIG['model_names'][model_key]
    logging.info(f"--- Starting Transformer Training for: {model_key} ({model_name}) ---")
    logging.info(f"Using device: {config.DEVICE}")

    # --- 2. Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(config.DEVICE)

    # --- 3. Datasets and DataLoaders ---
    train_dataset = FakeNewsDataset(
        data_path=os.path.join(config.DATA_DIR, 'train.parquet'),
        tokenizer=tokenizer,
        max_length=config.TRANSFORMER_CONFIG['max_length'],
        text_column=config.TRANSFORMER_CONFIG['text_column']
    )
    val_dataset = FakeNewsDataset(
        data_path=os.path.join(config.DATA_DIR, 'validation.parquet'),
        tokenizer=tokenizer,
        max_length=config.TRANSFORMER_CONFIG['max_length'],
        text_column=config.TRANSFORMER_CONFIG['text_column']
    )
    train_loader = DataLoader(train_dataset, batch_size=config.TRANSFORMER_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TRANSFORMER_CONFIG['batch_size'])

    # --- 4. Optimizer, Scheduler, and Scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TRANSFORMER_CONFIG['learning_rate'], weight_decay=config.TRANSFORMER_CONFIG['weight_decay'])
    num_training_steps = len(train_loader) * config.TRANSFORMER_CONFIG['epochs'] // config.TRANSFORMER_CONFIG['gradient_accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.TRANSFORMER_CONFIG['warmup_steps'], num_training_steps=num_training_steps)
    use_amp = config.DEVICE == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- 5. Training & Evaluation Loop ---
    best_f1 = 0.0
    training_stats = []

    for epoch in range(config.TRANSFORMER_CONFIG['epochs']):
        logging.info(f"--- Epoch {epoch+1}/{config.TRANSFORMER_CONFIG['epochs']} ---")
        
        # Training phase
        model.train()
        total_train_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for i, batch in enumerate(progress_bar_train):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / config.TRANSFORMER_CONFIG['gradient_accumulation_steps']

            scaler.scale(loss).backward()
            
            if (i + 1) % config.TRANSFORMER_CONFIG['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * config.TRANSFORMER_CONFIG['gradient_accumulation_steps']
            progress_bar_train.set_postfix({'loss': total_train_loss / (i + 1)})
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1} Evaluating")
            for batch in progress_bar_val:
                input_ids = batch['input_ids'].to(config.DEVICE)
                attention_mask = batch['attention_mask'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                total_val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = compute_metrics(all_preds, all_labels)
        
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}")
        logging.info(f"Average Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Validation Metrics (Epoch {epoch+1}): {val_metrics['f1_score']:.4f} F1-Score")

        training_stats.append({
            'epoch': epoch + 1,
            'training_loss': avg_train_loss,
            'validation_loss': avg_val_loss,
            'validation_metrics': val_metrics
        })

        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            logging.info(f"New best F1 score: {best_f1:.4f}. Saving model...")
            model_save_path = os.path.join(config.MODELS_DIR, f'{model_key}_model')
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)

    # --- 6. Save Final Results ---
    results_filename = f'transformer_{model_key}_training_stats.json'
    save_json_results(training_stats, results_filename, config.RESULTS_DIR)
    logging.info(f"--- Training for {model_key} complete. ---")

if __name__ == '__main__':
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(config.LOGS_DIR, 'train_transformers.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()]
    )
    for model_key in config.TRANSFORMER_CONFIG['model_names']:
        train_transformer(model_key)