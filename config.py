# dataset config

train_dir ='./train_data'
train_split, validation_split, test_split = 0.7, 0.2, 0.1
batch_size = 16
data_loading_workers = 2
random_seed = 330
image_size = 224


# train config

hub_repo = 'pytorch/vision:v0.10.0'
model_arch = 'resnext101_32x8d'
out_dir = './out'
momentum = 0.9
initial_learning_rate = 5e-5
weight_decay = 1e-3
start_epoch = 0
epochs = 20
print_freq = 20