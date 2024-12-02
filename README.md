# SparseTSF-in-mindspore

SparseTSF在MindSpore上的代码迁移

使用代码和介绍文字、实验结果请遵循CC0 1.0 通用知识共享公共许可协议，开源是为了大家交流讨论，希望大家遵循开源精神！

## 模型启动前准备

创建3.8的Python环境，目前不支持其他版本的Python，受限于MindSpore的版本变化。

```bash
conda create -n SparseTSF-MS python=3.8
```

然后我们启动SparseTSF-MS环境并进行相应的配置

```bash
conda activate SparseTSF-MS
pip install -r requirements.txt
```

在仓库中的数据集上的模型训练测试的指令：

```bash
python run.py --config_name configs/GPU_ETTh1.yaml
```

### 主要修改部分

#### 1.data_factory.py

由于在原来的data_factory.py文件主要作用是构建Dataset、DataLoader类用于模型的数据读取

Dataset类是所有数据集的基类，提供了数据处理方法来帮忙预处理数据

Dataloader类则是数据集上的python可迭代对象，方便我们进行批处理

下面是在torch中与mindspore中代码的对比

```pytho
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
```

```pyt
import mindspore.dataset as ds	
from mindspore.dataset import GeneratorDataset
data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        inverse=args.inverse, #new
        timeenc=timeenc,
        freq=freq,
        cols=args.cols #new
    )
data_loader = GeneratorDataset(source=data_set, column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"],shuffle = shuffle_flag)
data_loader = data_loader.batch(batch_size = batch_size, drop_remainder=drop_last)
```

发现主要区别在于：在mindspore中，batch_size需要特别地设置

#### 2.SparseTSF的修改

首先最明显的不同在于，模型继承的类不同，torch中继承的是torch.nn.Module,而mindspore的在mindspore.nn.Cell

```py
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding

class Model(nn.Module):
```

```python
import mindspore 
import mindspore.nn as nn
from mindspore import ops

class SparseTSF(nn.Cell):
```

其次，在我们在mindspore中设计神经网络时，一方面，神经网络模块必须放入nn.SequentialCell容器，这个容器允许将多个网络层按顺序组合起来，例如我的神经网络是经过一层卷积层然后经过激活函数最后进行一次卷积的话，就往这个容器中append进自己定义的卷积、激活函数、卷积，然后将输入传入这个容器就可以自动计算出经过我们自己定义的网络的结果。其次，mindspore中的全连接层是封装在Dense模块中

```python
self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)
self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
```



```python
self.conv1d = nn.SequentialCell([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                      stride=1, pad_mode="pad", padding=self.period_len // 2, has_bias=False)
        ])

        # 自定义全连接操作
        self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)
```

相较于torch中的forward函数进行前向传播操作，mindspore则是通过函数construct来进行的

```python
 def forward(self, x):
```

```python
def construct(self, x):
```

其余对张量的操作，二者大致是一致的，因此不做介绍

#### 3.Exp_Main()的修改

首先一个小的修改部分在于优化器的选择时一点小小的不同,即选取参数的不同

```python
def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
```

```python
def _select_optimizer(self):
        model_optim = nn.Adam(self.model.trainable_params(),learning_rate=self.args.learning_rate)
        return model_optim
```

其次是训练时数据的迭代的在名字上不同

```python
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
```



```python
 for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_source.create_tuple_iterator()):
```

其次在前向传播和反向传播的二者存在较大的不同

对于torch而言就是简单的计算出模型计算的结果然后计算loss值，进行反向传播，参数调整

```python
batch_x = batch_x.float().to(self.device)
batch_y = batch_y.float().to(self.device)
outputs = self.model(batch_x)
f_dim = -1 if self.args.features == 'MS' else 0
outputs = outputs[:, -self.args.pred_len:,f_dim:]
batch_y = batch_y[:, -self.args.pred_len:,f_dim:].to(self.device)
loss = criterion(outputs, batch_y)
train_loss.append(loss.item())
loss.backward()
model_optim.step()
```

而mindspore则较为复杂，需要先写一个forward_fn函数，用作计算结果和损失值

```python
def forward_fn(batch_x,batch_y,batch_x_mark,batch_y_mark,label_len,pred_len):
            cast = ops.Cast()
            batch_x = cast(batch_x,ms.float32)
            #print("Batch_x:\n{0}".format(batch_x.shape))
            batch_y = cast(batch_y,ms.float32)
            
            batch_x_mark = cast(batch_x_mark,ms.float32)
            batch_y_mark = cast(batch_y_mark,ms.float32)
            ouputs = model(batch_x)
            f_dim = 0
            ouputs = ouputs[:,-self.args.pred_len:,f_dim:]
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
            #print("Outputs:{0},Batch_y:{1}".format(ouputs.shape,batch_y.shape))
            loss = criterion(ouputs,batch_y)
            return loss,ouputs
```

然后定义一个grad_fn函数用于在前向传播时计算梯度并得到损失值，然后将计算出来的梯度传入优化器进行参数的更新

```python
grad_fn = ms.ops.value_and_grad(forward_fn,None,model_optim.parameters,has_aux=True)
(loss,_),grads = grad_fn(batch_x,batch_y,batch_x_mark,batch_y_mark,self.args.label_len,self.args.pred_len)
loss = ms.ops.depend(loss,model_optim(grads))
train_loss.append(loss.asnumpy().item())
```

对于学习率的调整和早停机制的修改会放在后面进行介绍

#### 4.utils.tools的修改

学习率调整策略的修改

在torch中可以直接对学习率进行修改，因为每个参数都有一个['lr']的超参数；而对于mindspore的学习率调整只能新生成一个optimizer来进行学习率的更新

```python
def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
     if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if printout: print('Updating learning rate to {}'.format(lr))
```

```python
def adjust_learning_rate(optimizer, parameters, epoch, args):
    lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        optimizer = ms.nn.Adam(parameters, learning_rate=lr)
        print('Updating learning rate to {}'.format(lr))
    return optimizer
```

早停策略大致一致，不同之处在于参数的存储调用的函数的名称不同

标准化的调整同样也只是名字修改层面

#### 5.run.py的调整

run.py的主要作用是参数的设置以及设备的选择

设备的选择上,mindspore是采用的context来设置，设置之后，不需要向torch一样通过to_device将数据传入GPU

```python
args = dict_to_namedtuple(args)
print("DEVICE:", args.device)
ms.set_context(device_target=args.device)
ms.set_context(mode=ms.PYNATIVE_MODE)
```

其余一致

**此外我们在论文的基础上进一步实现了在mindspore上的模型的等效参数可视化**

下面po上我们的理解与代码实现

在《SparseTSF》论文中，采用了一种参数可视化的方法来展现参数的周期性从而展现其模型提取周期性的强大能力：

<img src="https://github.com/July-h5kf3/SparseTSF-in-mindspore/blob/main/figure/image_in_papers.png" style="width:400px">

对于线性层的权重可视化很简单，因为一个线性层的参数就是一个大小为[I,O]的矩阵，其中I为输入的向量大小，O为输出的向量大小，那么可以直接plot出来。

然而对于SparseTSF，由于其可训练参数构成为：大小为 $2 * \frac{\omega}{2} + 1$的一维卷积核以及大小为$\frac{H}{\omega} * \frac{L}{\omega}$ 的线性层，显然无法直接用矩阵表现出来，那么我们应该如何对这样的参数进行可视化呢？论文采取了等效参数的方法。

我们首先可以思考一下一个简单的线性层中矩阵中的一个元 的意义。很显然这是**第i个输入对于第j个输出的贡献**那么再回到我们的时间序列预测问题，这同样是i个输入与，无论模型多么复杂，我们同样借鉴上面的思路将其等效为一个二维参数矩阵，矩阵的意义同上。

为什么呢？可以这样想

假设我们输入的用来预测时间序列是 $[1,0,\dots,0]$,那我们得到的输出$[a_{11},a_{12},\dots,a_{1n}]$那么就时间序列的第一个时间步对预测时间序列的贡献（经过这个模型）就一目了然了！

接下来我们来讲解一下在代码层面的实现

```python
import torch
from models.SparseTSF import Model as SparseTSF
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
```

首先导入库，特别地需要导入matplotlib库以及模型

```python
class config:
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 7
        self.period_len = 24
        self.d_model = 128
        self.model_type = 'linear'
configs = config()
```

定义参数

```python
path =
"./checkpoints/ETTh1_96_96_SparseTSF_ETTh1_ftM_sl96_pl96_linear_test_0_seed2023/
checkpoint.pth"
model = SparseTSF(configs)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint)
```

加载我们训练好的模型的参数，并导入模型

```python
eye_matrix = torch.eye(96,96)
eye_matrix = eye_matrix.unsqueeze(-1)
eye_matrix = eye_matrix.repeat(1,1, 7)
```

定义单位矩阵，这里最后之所以repeat（1,1,7）是因为ETTH1数据集有7个通道，每个通道的输入输出贡献不一致，我们分别求出来，最后求均值归一化。

```python
output = model(final_tensor)
output = output.mean(dim = 2)
weights_list = output.detach().numpy()
weights_min = np.min(weights_list)
weights_max = np.max(weights_list)
weights_list = (weights_list - weights_min) / (weights_max - weights_min)
```

求出模型的结果，并将结果求均值归一化，并且转化为numpy方便进行二维可视化

```python
save_root = ''
if os.path.exists('weights_plot'):
os.mkdir('weights_plot')
fig,ax = plt.subplots()
w_name = 'SparseTSF'
im = ax.imshow(weights_list,cmap='plasma_r')
fig.colorbar(im,pad = 0.1)
plt.savefig(os.path.join(save_root,w_name+'.pdf'),dpi=500)
plt.close
print(output)
```

保存模型，并绘制图像.
mindspore对应的代码在仓库中有，在导入模型参数以及设置单位矩阵中有一些不同

最后展示一下我们的结果：

这是torch的版本训练出来的参数在我们的代码上呈现的可视化结果

<img src="https://github.com/July-h5kf3/SparseTSF-in-mindspore/blob/main/figure/torch_in_our_code.png" style="width:400px">

这是我们迁移到mindspore上训练出来的参数在我们的代码上呈现的可视化结果

<img src="https://github.com/July-h5kf3/SparseTSF-in-mindspore/blob/main/figure/mindspore.png" style="width:400px">

## Aknowledge 致谢参考

SparseTSF：https://github.com/lss-1138/SparseTSF
Informer：https://github.com/zhouhaoyi/Informer2020

