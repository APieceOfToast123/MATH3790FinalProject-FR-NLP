import pandas as pd

# 检查和清理训练集
train_data = pd.read_csv('cnn_train.csv', header=None)
valid_train_data = train_data[(train_data[0] >= 0) & (train_data[0] < 7)]
valid_train_data.to_csv('cnn_train_clean.csv', header=False, index=False)

# 检查和清理验证集
val_data = pd.read_csv('cnn_val.csv', header=None)
valid_val_data = val_data[(val_data[0] >= 0) & (val_data[0] < 7)]
valid_val_data.to_csv('cnn_val_clean.csv', header=False, index=False)

print(f"Training data cleaned: {len(train_data) - len(valid_train_data)} invalid rows removed.")
print(f"Validation data cleaned: {len(val_data) - len(valid_val_data)} invalid rows removed.")
