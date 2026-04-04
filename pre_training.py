from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Đọc bộ dữ liệu từ file CSV
dataset = load_dataset("csv", data_files="data.csv")

# Tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['query'], truncation=True, padding='max_length', max_length=128)

# Áp dụng tokenizer vào bộ dữ liệu
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Chia bộ dữ liệu thành train và test
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# In ra kích thước của bộ dữ liệu sau khi chia
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
