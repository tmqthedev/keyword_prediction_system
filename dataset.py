from datasets import load_dataset
import os

# Lấy đường dẫn tuyệt đối đến thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến file CSV
train_file = os.path.join(current_dir, 'data.csv')
test_file = os.path.join(current_dir, 'data.csv')  # Sử dụng cùng file cho test

# Đọc bộ dữ liệu từ file CSV
try:
    dataset = load_dataset('csv', data_files={
        'train': train_file,
        'test': test_file
    })
    print("Dataset loaded successfully!")
    
    # Kiểm tra dữ liệu
    print("\nDataset Info:")
    print(dataset)
    
    # Chọn dataset train và test
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # In thông tin chi tiết
    print("\nTrain dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    print("\nSample data from train dataset:")
    print(train_dataset[:5])
    
except FileNotFoundError as e:
    print(f"Error: Could not find the CSV file.\nPlease ensure 'data.csv' exists in: {current_dir}")
except Exception as e:
    print(f"Error loading dataset: {e}")