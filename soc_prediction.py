# -*- coding:utf-8 -*-
'''
作者：ych
日期：2023年9月15日
'''
import numpy as np
import paddle
from paddle.static import InputSpec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = r'./'  ##数据位置

data = np.load(path)
x = data['x']
y = data['y']

##划分训练集和测试集    数据归一化
split_radio=0.8  ##8:2划分
train_split=int(split_radio*x.shape[0])
input_windows = x.shape[1]
fea_num = x.shape[2]

def normalize(data1,train_split):   ##只利用trainset 计算均值和方差，然后给所有数据进行归一化
    mean = np.mean(data1[:train_split],axis=0)
    var = np.std(data1[:train_split], axis=0)
    # for i in range(data.shape[0])
    return (data1 - mean)/var , mean , var

x_norm , mean_x, var_x = normalize(x, train_split)
y_norm , mean_y , var_y = normalize(y, train_split)

x_norm[:,-1,1]=0   ##单独令padding部分为0 ;  universe的时候，记得处理这个为0的地方

##划分数据集
train_x = x_norm[:train_split]
train_y = y_norm[:train_split]
test_x = x_norm[train_split:]
test_y = y_norm[train_split:]


##制作数据集
class MyDataset(paddle.io.Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.x = paddle.to_tensor(x, dtype='float32')
        self.y = paddle.to_tensor(y, dtype='float32')

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]


train_dataset = MyDataset(train_x, train_y)
test_dataset = MyDataset(test_x, test_y)

##搭建模型 LSTM  MLP+LSTM+MLP
class LSTM(paddle.nn.Layer):
    def __init__(self,input_windows,fea_num):   ##时间序列长度    特征数量
        super().__init__()
        self.fc1 = paddle.nn.Linear(fea_num,64)
        self.relu1 = paddle.nn.ReLU()
        self.lstm = paddle.nn.LSTM(64, 128, num_layers=1)
        self.fc2 = paddle.nn.Linear(128, 256)
        self.relu2 = paddle.nn.ReLU()
        self.fc3 = paddle.nn.Linear(256,1)

    def forward(self, inputs):
        inputs = self.relu1(self.fc1(inputs))
        inputs, _ = self.lstm(inputs)
        inputs = inputs[:,-1,:]
        inputs = self.relu2(self.fc2(inputs))
        inputs = self.fc3(inputs)
        return inputs

##打印网络结构
model = LSTM(input_windows,fea_num)
paddle.summary(model,(128,13,3))


# 自定义Callback 需要继承基类 Callback
class LossCallback(paddle.callbacks.Callback):

    def __init__(self):
        self.losses = []

    def on_train_begin(self, logs={}):
        # 在fit前 初始化losses，用于保存每个batch的loss结果
        self.losses = []

    def on_train_batch_end(self, step, logs={}):
        # 每个batch训练完成后调用，把当前loss添加到losses中
        self.losses.append(logs.get('loss'))


loss_log = LossCallback()


##参数设置
epoch_num = 10
batch_size = 256
lr = 0.001

inputs = InputSpec([None,13,3], 'float32', 'x')
labels = InputSpec([None,1], 'float32', 'y')
model = paddle.Model(LSTM(input_windows, fea_num),inputs, labels)

lr_schedual = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr,T_max=epoch_num,verbose=False)
opt = paddle.optimizer.Adam(learning_rate=lr_schedual, parameters=model.parameters(), beta1=0.9, beta2=0.999)

model.prepare(opt,paddle.nn.MSELoss(), paddle.metric.Accuracy())
##模型训练
model.fit(train_dataset,test_dataset,batch_size,eval_freq=10,save_freq=5, save_dir='lstm_checkpoint', verbose=1, drop_last=True, shuffle=True, callbacks=[loss_log],epochs=epoch_num)

##可视化Loss
log_loss = [ loss_log.losses[i] for i in range(len(loss_log.losses))]
plt.figure()
plt.plot(log_loss)

##模型评估
path_pre=r'lstm_checkpoint/final'
model.load(path_pre)
eval_result = model.evaluate(test_dataset, batch_size, verbose=1)
print(eval_result)

#制作pre_dataset
path_task=r'data/1_102_车组.csv'
model.load(path_pre)
task = pd.read_csv(path_task)
x1_list=[]
y1_list=[]
for j in range(task.shape[0]-input_windows+1):
    x1 = task[j:(j+input_windows)].values
    x1[-1,1] = 0
    x1_list.append(x1)
    # print(task[(j+input_windows-1):(j+input_windows)])
    y1_list.append(task[(j+input_windows-1):(j+input_windows)].values[0,1])
x1 = np.array(x1_list)
y1 = np.array(y1_list)
##归一化
x1 = (x1 - mean_x)/var_x
y1 = (y1 - mean_y)/var_y

##去除NAN
x1[:,-1,1]=0

##定义pre_dataset
pre_dataset = MyDataset(x1,y1)

# ##模型预测
test_value = model.predict(pre_dataset)

