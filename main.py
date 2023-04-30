from app.model.lstm import LSTMModel
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    NEED_RETRAIN = True
    file_name = 'SH#600030'
    MODEL_FILE_PATH = './models/'
    lstm_model = LSTMModel(data_file_path=f"./data/{file_name}.txt", data_split=0.9)
    train_data, test_data = lstm_model.prepare()
    model_file_name = MODEL_FILE_PATH + file_name + 'rnn_model.pd'

    if NEED_RETRAIN or os.path.exists(model_file_name) == False:
        model, train_result = lstm_model.createAndTrain(train_data, test_data, f"./models/{file_name}.h5")
        lstm_model.trainResult(train_result)
    else:
        model = lstm_model.load(f"./models/{file_name}.h5")

    model.summary()
    predicted_data, real_data = lstm_model.test(model, test_data)
    lstm_model.evalResult(predicted_data)
    #
    print(f"预测数据的维度为：{predicted_data.shape}")
    print(f"真实数据的维度为：{real_data.shape}")
    future_list = lstm_model.future(model, 10)
    print(f"未来10天的预测数据为：{future_list}")
    lstm_model.predictResult(future_list)