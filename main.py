import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm


# check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# init tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

NUM_EPOCHS = 10 #num of epochs, change for better performance vs computationally expensive


# clean the data by removing NaN values and ensuring numeric values for complexity

def clean_data(df):
    df = df.dropna(subset=['sentence', 'token', 'complexity'])
    df = df[(df['sentence'] != '') & (df['token'] != '')]
    df['complexity'] = pd.to_numeric(df['complexity'], errors='coerce')
    return df.dropna(subset=['complexity'])

# load and preprocess data from a file path

def load_and_preprocess_data(file_path, is_test=False):
    try:
        df = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='warn')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        print("Attempting to read file with different encoding...")
        df = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='warn', encoding='latin1')
    
    if not is_test:
        df = clean_data(df)
        y = df['complexity'].values
    else:
        y = np.zeros(len(df))  # Assume zero complexity for placeholder
    X = df['sentence'] + " [SEP] " + df['token']
    return X, y, df

# dataset class for handling complexity data

class ComplexityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx] if self.labels is not None else 0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# evaluate prediction performance using various metrics

def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Median Absolute Error: {med_ae:.4f}")


    
    return mse, rmse, mae, med_ae

# plot learning curves for training and validation loss


def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(min(min(train_losses), min(val_losses)) * 0.9, 
             max(max(train_losses), max(val_losses)) * 1.1)  # Adjust y-axis limits
    plt.savefig('learning_curves.png')
    plt.close()

# model definition for complexity prediction using BERT

class ComplexityBERTModel(nn.Module):
    def __init__(self, bert_model):
        super(ComplexityBERTModel, self).__init__()
        self.bert = bert_model # BERT model
        self.dropout = nn.Dropout(0.3) # droput layer
        self.fc1 = nn.Linear(768, 256) # full connected layer
        self.fc2 = nn.Linear(256, 1) # output layer

# feed forward function

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# train the model with early stopping

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=NUM_EPOCHS, patience=2):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.mse_loss(outputs.squeeze(), labels)
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = F.mse_loss(outputs.squeeze(), labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt') # save the model
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered")
                break

    plot_learning_curves(train_losses, val_losses)
    return model

#  error analysis


def error_analysis(y_true, y_pred, X, n=10):
    errors = np.abs(y_true - y_pred)
    worst_indices = np.argsort(errors)[-n:]
    best_indices = np.argsort(errors)[:n]
    
    print("Worst Predictions:")
    for idx in worst_indices:
        print(f"Input: {X.iloc[idx]}")
        print(f"Predicted: {y_pred[idx]:.4f}")
        print()
    
    print("Best Predictions:")
    for idx in best_indices:
        print(f"Input: {X.iloc[idx]}")
        print(f"Predicted: {y_pred[idx]:.4f}")
        print()

# train and evaluate the model with hyperparameter tuning


def train_and_evaluate(X_train, y_train, X_test, y_test, is_single_word):
    # Scale the target variable
    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten() if y_test is not None else None

    train_dataset = ComplexityDataset(X_train, y_train_scaled, tokenizer)
    test_dataset = ComplexityDataset(X_test, y_test_scaled if y_test_scaled is not None else np.zeros(len(X_test)), tokenizer)

    # hyperparameter tuning
    learning_rates = [1e-6, 5e-6, 1e-5]
    batch_sizes = [8, 16]
    best_val_loss = float('inf')
    best_lr = None
    best_batch_size = None

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with lr={lr}, batch_size={batch_size}")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            model = ComplexityBERTModel(BertModel.from_pretrained('bert-base-uncased'))
            model.to(device)
            optimizer = AdamW(model.parameters(), lr=lr)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * NUM_EPOCHS)
            
            model = train_model(model, train_loader, val_loader, optimizer, scheduler, device)
            
            # Evaluate on validation set
            model.eval()
            val_predictions = []
            val_true = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_true.extend(labels.cpu().numpy())
            
            val_mse = mean_squared_error(val_true, val_predictions)
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_lr = lr
                best_batch_size = batch_size

    print(f"Best hyperparameters: lr={best_lr}, batch_size={best_batch_size}")

    # train final model with best hyperparameters
    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size)

    final_model = ComplexityBERTModel(BertModel.from_pretrained('bert-base-uncased'))
    final_model.to(device)
    optimizer = AdamW(final_model.parameters(), lr=best_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * NUM_EPOCHS)

    final_model = train_model(final_model, train_loader, test_loader, optimizer, scheduler, device)

    # evaluate on test set
    final_model.eval()
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = final_model(input_ids, attention_mask=attention_mask)
            test_predictions.extend(outputs.cpu().numpy().flatten())

    # inverse transform the predictions
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()

    if y_test is not None and len(y_test) > 0:
        print(f"{'Single' if is_single_word else 'Multi'} Word Complexity Prediction Results:")
        evaluate_predictions(y_test, test_predictions)       # error analysis
        error_analysis(y_test, test_predictions, X_test)

        # baseline model (always predict mean complexity)
        mean_complexity = np.mean(y_train)
        baseline_predictions = np.full_like(y_test, mean_complexity)
        print(f"{'Single' if is_single_word else 'Multi'} Word Complexity Baseline Model Performance:")
        evaluate_predictions(y_test, baseline_predictions)
    else:
        print("Test set labels not available. Skipping evaluation and error analysis.")

    return test_predictions

print("Processing single word complexity...")
X_train, y_train, df_train = load_and_preprocess_data('lcp_single_train.tsv')
X_test, y_test, df_test = load_and_preprocess_data('lcp_single_test.tsv', is_test=True)

single_predictions = train_and_evaluate(X_train, y_train, X_test, y_test, is_single_word=True)

# create a DataFrame for single word predictions
single_predictions_df = pd.DataFrame({
    'sentence': X_test,
    'token': df_test['token'],
    'predicted_complexity': single_predictions
})

# save single word predictions
single_predictions_df.to_csv('single_word_predictions.csv', index=False)
print("Single word predictions saved to single_word_predictions.csv")

print("\nProcessing multiple word complexity...")
X_train, y_train, df_train = load_and_preprocess_data('lcp_multi_train.tsv')
X_test, y_test, df_test = load_and_preprocess_data('lcp_multi_test.tsv', is_test=True)

multiple_predictions = train_and_evaluate(X_train, y_train, X_test, y_test, is_single_word=False)

# create a DataFrame for multiple word predictions
multiple_predictions_df = pd.DataFrame({
    'sentence': X_test,
    'token': df_test['token'],
    'predicted_complexity': multiple_predictions
})

# save multiple word predictions
multiple_predictions_df.to_csv('multiple_word_predictions.csv', index=False)
print("Multiple word predictions saved to multiple_word_predictions.csv")