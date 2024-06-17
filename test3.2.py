import os

import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# 建立类别标签，不同类别对应不同的数字。
label = ['aloe', 'burger', 'cabbage', 'candied_fruits',
         'carrots', 'chips', 'chocolate', 'drinks', 'fries',
         'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
         'pizza', 'ribs', 'salmon', 'soup', 'wings']
label_dict = dict(zip(label, range(len(label))))


def extract_features(path, rates=(1.0,)):
    """
    提取特征
    :param path: 音频文件路径
    :param rates: 拉伸系数，对音频进行缩放，音调不变。这里为了做数据增强
    :return: 不同的rate提取的特征
    """
    y_0, sr = librosa.load(path)
    # 缩放后的y
    y_list = [librosa.effects.time_stretch(y_0, rate=rate) for rate in rates]
    features = []
    for y in y_list:
        # 这里使用mfcc
        mel = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128).T
        features.append(np.mean(mel, axis=0))
    return np.array(features)


def extract_features_train(parent_dir, max_file=10):
    """
    训练集提取特征
    :param parent_dir: 训练集文件夹的路径
    :param max_file: 每个类别最多处理的文件数
    :return: 训练集的X和Y
    """
    X, Y = [], []
    for sub_dir in label:
        _, _, filenames = next(os.walk(os.path.join(parent_dir, sub_dir)))
        for filename in tqdm(filenames[:max_file]):
            # 这里做了数据增强，拉伸系数0.5, 0.7, 1.0, 1.4, 2.0
            features = extract_features(os.path.join(parent_dir, sub_dir, filename), (0.5, 0.7, 1.0, 1.4, 2.0))
            for feature in features:
                X.append(feature)
                Y.append(label_dict[sub_dir])
    return [np.array(X), np.array(Y)]


def extract_features_test(parent_dir):
    """
    测试集提取特征
    :param parent_dir: 测试集路径
    :return: 测试集的X
    """
    X = []
    _, _, filenames = next(os.walk(parent_dir))
    for filename in tqdm(filenames):
        # 测试集不需要数据增强，所以没有传rates
        X.append(extract_features(os.path.join(parent_dir, filename))[0])
    return np.array(X)


def save_name(test_dir):
    """
    将测试集的所有文件名保存为一个文件。
    因为数据集都为音频文件，占用空间过大，所以先预处理好，需要使用的时候，直接读取文件
    :return:
    """
    _, _, filenames = next(os.walk(test_dir))
    with open('path', 'w') as f:
        f.writelines([filename + '\n' for filename in filenames])


def save_features(train_dir, test_dir):
    """
    将训练集和测试集的特征提取之后，保存为npy文件。
    原因同save_name方法
    :return:
    """
    save_name(test_dir)
    X, Y = extract_features_train(train_dir, 1000)
    print(X.shape)
    print(Y.shape)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    X_ = extract_features_test(test_dir)
    print(X_.shape)
    np.save('X_.npy', X_)


# 实现train_test_split方法
def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test


def load_features():
    """
    从文件加载保存的特征
    :return: 训练集X和Y，测试集X_
    """
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    X_ = np.load('X_.npy')
    return X, Y, X_


def classifier():

    model = Sequential()
    # 由于n_mfcc=128，所以这里的Dense输入也为128维，激活函数使用的relu，经过尝试，效果好于tanh
    model.add(Dense(1024, input_dim=128, activation="relu"))
    # Dropout主要为了防止过拟合，这里随机去掉一半的特征进行预测
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    # 音频的分类为20类
    model.add(Dense(20, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def to_csv(model, X_, save_path='submit.csv'):
    """
    将结果保存为csv文件
    :param model: 训练好的模型
    :param X_: 测试集的特征
    :param save_path: 文件保存的路径
    :return:
    """
    predictions = model.predict(X_)
    preds = np.argmax(predictions, axis=1)
    preds = [label[x] for x in preds]
    path = []
    # 此处的path文件是save_name方法保存的
    with open("path") as f:
        for line in f:
            path.append(line.strip())
    result = pd.DataFrame({'name': path, 'label': preds})
    result.to_csv(save_path, index=False)

# 训练集文件位置
train_dir = r'D:\2345Downloads\桌面\深度学习\数据集\train_sample\train_sample'
# 测试集文件位置
test_dir = r'D:\2345Downloads\桌面\深度学习\数据集\test_b\test_b'
# 处理数据集，并保存为文件
save_features(train_dir, test_dir)
# 从文件中加载特征
X, Y, X_ = load_features()
# 对特征进行标准化处理
train_mean = np.mean(X, axis=0)
train_std = np.std(X, axis=0)
X = (X - train_mean) / train_std
X_ = (X_ - train_mean) / train_std
# 将类别转换为one-hot
Y = to_categorical(Y)
# 训练测试数据分离
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_ratio=0.1, seed=666)
# 建立模型
model = classifier()
model.fit(X_train, y_train, epochs=1000, batch_size=5000, validation_data=(X_test, y_test))

to_csv(model, X_, 'submit3.csv')


