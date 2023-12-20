###pytorch_basic_structure###
1.文件结构如下：
├──How_to_use_tensor.py (介绍torch基本用法)

├──train.py (训练网络的主循环脚本)

├──data
│   ├─gender_classification_eval_set.csv
│   ├─gender_classification_test.csv
│   ├─gender_classification_train.csv
│   ├─gender_classification_training_set.csv
│   └─split_dataset.py (将训练集分为train_set和eval_set)

├──network
│   ├─MyLoss.py (自定义损失函数的脚本)
│   └─MyNetwork.py (构建神经网络结构的脚本)

├──utils
│   └─MyDataset.py (数据加载模块,用来加载数据)

├──result.csv (保存的结果文件)

2.更多来源：
https://pytorch.org/docs/stable/tensors.html
