import logging

import keras
import pandas as pd
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import MinMaxScaler
from app.base import BaseModel
import numpy as np
import matplotlib.pyplot as plt


class LSTMModel(BaseModel):
    def __init__(self, data_file_path, batch_size=32, data_split=0.8, time_step=60, epoch=100, learning_rate=0.001):
        df = pd.read_csv(data_file_path, encoding='gbk', sep='\t')
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df.index = df['date']
        df['change_rate'] = (df['close'] - df['open']) * 100 / df['close']
        df['amplitude'] = (df['high'] - df['low']) * 100 / df['close']
        df['average'] = df['amount'] / df['volume']
        self.batch_size = batch_size
        self.data_split = data_split
        self.time_step = time_step
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.table_index = df['date']
        self.model = None
        self.dataset = df.values[:, 1:]
        self.train_data, self.test_data = self.prepare()

    def load(self, model_file_name):
        logging.info("加载LSTM模型......")
        model = keras.models.load_model(model_file_name)
        return model

    def split_data(self, dataArray, time_step):
        x_data, y_data = [], []
        for i in range(len(dataArray) - time_step):
            x_data.append(dataArray[i:i + time_step])
            y_data.append(dataArray[i + time_step])
        return np.array(x_data), np.array(y_data)

    def prepare(self):

        # 生成训练集和测试集
        train_data = self.dataset[:int(len(self.dataset) * self.data_split)]
        test_data = self.dataset[int(len(self.dataset) * self.data_split):]
        return train_data, test_data

    def create(self, shape):
        logging.info("创建LSTM模型......")
        # 创建LSTM模型
        model = Sequential()
        # 添加第一个LSTM层和一些Dropout正则化
        model.add(LSTM(
            units=64,
            return_sequences=True,
            input_shape=(shape[1], shape[2]),
            dropout=0.2,
            recurrent_dropout=0.2)
        )

        # 添加第二个LSTM层和一些Dropout正则化
        model.add(LSTM(units=32, return_sequences=True))
        model.add(LSTM(units=32, return_sequences=True))

        # 添加第三个LSTM层和一些Dropout正则化
        model.add(LSTM(units=16))

        # 添加输出层
        model.add(Dense(units=shape[2], activation='linear'))

        logging.info("开始编译模型......")
        # 8.编译RNN
        model.compile(optimizer=Adam(self.learning_rate), loss='mean_squared_error')
        return model

    def createAndTrain(self, train_data, test_data, saved_model_path):
        logging.info("开始训练LSTM模型......")
        # 归一化
        train_data = self.scaler.fit_transform(train_data)
        test_data = self.scaler.fit_transform(test_data)
        # 生成训练集和测试集的X和Y
        train_x, train_y = self.split_data(train_data, self.time_step)
        test_x, test_y = self.split_data(test_data, self.time_step)
        print(f"训练集的维度为：{train_x.shape}, {train_y.shape}")
        print(f"测试集的维度为：{test_x.shape}, {test_y.shape}")
        # 创建模型
        model = self.create(train_x.shape)
        # 训练模型
        result =model.fit(
            train_x,    # 训练集
            train_y,    # 训练集标签
            epochs=self.epoch,  # 迭代次数
            batch_size=self.batch_size,     # 每次训练的样本数
            verbose=2,  # 0:不输出日志信息 1:输出进度条记录 2:每个epoch输出一行记录
            shuffle=False,  # 是否打乱数据
            # sample_weight=weight, # 样本权重
            validation_data=(test_x, test_y),   # 验证集
            callbacks=[ModelCheckpoint(f"{saved_model_path}best_model.h5", monitor='val_loss', save_best_only=True)]
        )
        # 保存模型
        model.save(saved_model_path)
        logging.info(f"模型保存成功！{saved_model_path}")
        return model, result

    def trainResult(self, result):
        logging.info("开始绘制训练图......")
        plt.plot(result.history['loss'], marker='o')
        plt.plot(result.history['val_loss'], marker='o')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.grid()
        plt.show()

    def test(self, model, test_data):
        logging.info("开始预测股票......")
        # 归一化
        test_data = self.scaler.fit_transform(test_data)
        # 生成测试集的X和Y
        test_x, test_y = self.split_data(test_data, self.time_step)
        logging.info(f"测试集的维度为：{test_x.shape}, {test_y.shape}")
        # 预测
        predicted_data = model.predict(test_x)
        logging.info(f"预测集的维度为：{predicted_data.shape}, {predicted_data.shape}")
        # 反归一化
        predicted_data_array = self.scaler.inverse_transform(predicted_data)
        logging.info(f"预测结果sharp为：{predicted_data_array.shape}")
        logging.debug(f"预测结果为：{predicted_data_array}")
        # 反归一化
        real_data_array = self.scaler.inverse_transform(test_y)
        logging.info(f"实际结果sharp为：{real_data_array.shape}")
        logging.debug(f"实际结果为：{real_data_array}")
        return predicted_data_array, real_data_array

    def evalResult(self, predicted_data):
        logging.info("开始绘制评估对比图......")

        predicted_data = predicted_data.astype('object')
        predicted_data = np.insert(predicted_data, 0, self.table_index.iloc[int(len(self.table_index)* self.data_split  + self.time_step):], axis=1)
        predicted_data = pd.DataFrame(predicted_data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change_rate', 'amplitude', 'average'])
        predicted_data.set_index('date', inplace=True)

        real_data = self.test_data.astype('object')
        real_data = real_data[self.time_step:]
        real_data = np.insert(real_data, 0, self.table_index.iloc[int(len(self.table_index)* self.data_split  + self.time_step):], axis=1)
        real_data = pd.DataFrame(real_data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change_rate', 'amplitude', 'average'])
        real_data.set_index('date', inplace=True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(100))
        plt.plot(real_data['close'], color='red', label='Real Stock Price', marker='o')
        plt.plot(predicted_data['close'], color='blue', label='Predicted Stock Price', marker='o')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.grid()
        plt.legend()
        plt.show()

    def future(self, model, days):
        logging.info("开始预测未来股票......")
        print("self.dataset.sharp = ", self.dataset.shape)
        dataset = self.scaler.fit_transform(self.dataset)
        features = dataset.shape[1]
        predicted_data = dataset[len(dataset) - self.time_step:len(dataset)]
        print("predicted_data.sharp = ", predicted_data.shape)
        # lastOne = (self.dataset[len(self.dataset) - self.time_step:len(self.dataset)]).reshape(-1, features)
        backData = predicted_data.tolist()
        next_predicted_list = []
        for i in range(days):
            one_next = backData[len(backData) - self.time_step:]
            one_next = np.array(one_next).reshape(1, self.time_step, features)
            print("one_next.sharp = ", one_next.shape)
            # print("one_next = ", one_next)
            next_predicted = model.predict([one_next])
            print("next_predicted.sharp = ", next_predicted.shape)
            print("next_predicted = ", next_predicted)
            next_predicted = self.scaler.inverse_transform(next_predicted)
            next_predicted_list.append(next_predicted[0].tolist())
            backData.append(next_predicted[0])

        next_predicted_list = np.array(next_predicted_list)

        return next_predicted_list

    def predictResult(self, predicted_data):
        logging.info("开始绘制预测图......")

        next_predicted_list = predicted_data.astype('object')

        given_date = self.table_index.iloc[len(self.table_index) - 1:].values[0]
        date_range = pd.date_range(start=given_date, periods=len(predicted_data) + 1, freq=pd.offsets.BDay())
        business_days = date_range.strftime('%Y-%m-%d').tolist()
        business_days = business_days[1:]

        next_predicted_list = np.insert(next_predicted_list, 0, business_days, axis=1)
        next_predicted_list = pd.DataFrame(next_predicted_list,
                                           columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                                                    'change_rate', 'amplitude', 'average'])
        next_predicted_list.set_index('date', inplace=True)

        ax = plt.gca()
        # ax.xaxis.set_major_locator(MultipleLocator(100))
        plt.plot(next_predicted_list['close'], color='blue', label='Predicted Stock Price', marker='o')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.grid()
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def predictTest(self, model, train_data, test_data):
        logging.info("开始预测对比股票......")
        train_data = self.scaler.fit_transform(train_data)
        features = test_data.shape[1]
        predicted_data = train_data[len(train_data) - self.time_step:len(train_data)]
        print("predicted_data.sharp = ", predicted_data.shape)
        # lastOne = (self.dataset[len(self.dataset) - self.time_step:len(self.dataset)]).reshape(-1, features)
        backData = predicted_data.tolist()
        next_predicted_list = []
        for i in range(len(test_data)):
            one_next = backData[len(backData) - self.time_step:]
            one_next = np.array(one_next).reshape(1, self.time_step, features)
            print("one_next.sharp = ", one_next.shape)
            # print("one_next = ", one_next)
            next_predicted = model.predict([one_next])
            print("next_predicted.sharp = ", next_predicted.shape)
            print("next_predicted = ", next_predicted)
            next_predicted = self.scaler.inverse_transform(next_predicted)
            next_predicted_list.append(next_predicted[0].tolist())
            backData.append(next_predicted[0])

        next_predicted_list = np.array(next_predicted_list)

        return next_predicted_list
