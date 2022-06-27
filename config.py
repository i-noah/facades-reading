# dataset config

train_dir ='./train_data' # 训练数据文件夹
train_split, validation_split, test_split = 0.8, 0.1, 0.1 # 数据集划分比例
batch_size = 16 # 每次批量从数据集加载的数量
data_loading_workers = 2 # 数据加载的线程
random_seed = 330 # 随机种子
image_size = 224 # 图像尺寸，最小224


# train config

hub_repo = 'pytorch/vision:v0.10.0' # torch hub 储存库版本
model_arch = 'resnext50_32x4d' # 模型架构
out_dir = './out' # 存档储存位置
momentum = 0.9 # 梯度下降动量
initial_learning_rate = 5e-5 # 初始学习率
weight_decay = 1e-3 # 梯度下降权值衰减
start_epoch = 0 # 起始代数 方便做继续训练
epochs = 6 # 终止代数
print_freq = 10 # 每隔多少组数据输出一次日志