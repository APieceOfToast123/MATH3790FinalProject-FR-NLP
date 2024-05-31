# 导入必要的库
import collections

def calculate_vocab_size(file_path):
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # 使用 collections.Counter 来计算每个字符的频率
    counter = collections.Counter(data)

    # 获取唯一字符的数量
    unique_chars = len(counter)

    # 添加特殊字符的数量（如 <start>, <end>, <unk>）
    vocab_size = unique_chars + 3  # 根据需要修改这里的数字以匹配你的特殊字符需求

    return vocab_size

if __name__ == "__main__":
    file_path = 'poems.txt'  # 这里填写你的数据文件路径
    vocab_size = calculate_vocab_size(file_path)
    print("Estimated Vocabulary Size:", vocab_size)
