# stock_lstm 基于LSTM深度学习的股票趋势预测算法
从通达信导出个股的历史数据放置到data目录下，直接运行main.py

python >3.8

keras版本：2.2.4

pandas版本：0.24.2

tensorflow版本：1.13.1

numpy版本：1.16.2

scikit-learn版本：0.20.3

matplotlib.pyplot版本：3.0.3

实现了完整的LSTM模型的创建、测试、滚动预测的整个流程，步骤如下：
 
 1、将数据集分成训练集和测试集

 2、预处理（归一化等）数据集

 3、创建lstm模型并训练
 
 4、测试模型
 
 5、滚动预测
 
 本代码只做研究LSTM模型，没有实际运用到实际股票预测中。

![myplot.png](asset%2Fmyplot.png)
 ![myplot1.png](asset%2Fmyplot1.png)
 ![myplot2.png](asset%2Fmyplot2.png)
