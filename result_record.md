# 简单的四层BiGRU，将social feature通过一层dense后直接相连接，输出层前面有一个dense(num_units=128,activation=tanh)
结果为：
![1](https://github.com/xhsun1997/rumor_project/blob/main/1.png)
# 经过随机打乱数据后，取出来75%作为训练，得到的结果为：
![2](https://github.com/xhsun1997/rumor_project/blob/main/2.png)
