import os
import pandas as pd
import numpy as np
import cv2

def process_data(root_dir, output_file):
    data = []
    labels = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    for emotion in emotions:
        emotion_dir = os.path.join(root_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
        for file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            data.append(img.flatten())
            labels.append(emotions.index(emotion))
    
    data = np.array(data)
    labels = np.array(labels).reshape(-1, 1)
    dataset = np.hstack((data, labels))
    
    pd.DataFrame(dataset).to_csv(output_file, header=None, index=False)
    print(f'Saved {len(data)} images to {output_file}')

# 处理训练集和验证集
print('Processing train dataset...')
process_data('fer2013/train', 'cnn_train.csv')
print('Processing test dataset...')
process_data('fer2013/test', 'cnn_val.csv')
