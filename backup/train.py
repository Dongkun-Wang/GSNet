import argparse
import json
import os
import random
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import normal_and_generate_dataset_time, get_mask, get_adjacent, get_grid_node_map_maxtrix,load_or_create_dataset,create_dataloaders
from lib.early_stop import EarlyStopping
from model.TDNet import TDNet
from lib.utils import mask_loss, compute_loss, predict_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file', default='./config/nyc/TDNet_NYC_Config.json')
parser.add_argument("--gpus", type=str, help="test program", default='4')
parser.add_argument("--test", action="store_true", help="test program")

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']

all_data_filename = config['all_data_filename']
mask_filename = config['mask_filename']
road_adj_filename = config['road_adj_filename']
risk_adj_filename = config['risk_adj_filename']
grid_node_filename = config['grid_node_filename']

spatial_temporal_embedding_dim = 128

grid_node_map = get_grid_node_map_maxtrix(grid_node_filename)  # TODO：400x243
num_of_vertices = grid_node_map.shape[1]

patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

train_rate = config['train_rate']
valid_rate = config['valid_rate']

recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior

training_epoch = config['training_epoch']

# 添加学习率调度器的配置
scheduler_step_size = config.get('scheduler_step_size', 10)
scheduler_gamma = config.get('scheduler_gamma', 0.5)

# 添加 num_workers 配置，默认为 4
#num_workers = config.get('num_workers', 4)

# 添加日志记录器，例如 TensorBoard（可选）
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='./logs')

def training(net,
             training_epoch,
             train_loader,
             val_loader,
             test_loader,
             road_adj,
             risk_adj,
             risk_mask,
             trainer,
             scheduler,  # 添加调度器
             early_stop,
             device,
             scaler,
             data_type='nyc'):
    global_step = 1
    for epoch in range(1, training_epoch + 1):
        start_epoch_time = time()
        net.train()
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{training_epoch}", unit="batch")
        for batch, (train_feature, target_time, gragh_feature, train_label) in enumerate(train_loader_tqdm, 1):
            # 将数据移动到设备上
            train_feature = train_feature.to(device)
            target_time = target_time.to(device)
            gragh_feature = gragh_feature.to(device)
            train_label = train_label.to(device)

            # 计算模型输出
            output = net(train_feature, target_time, gragh_feature, road_adj, risk_adj)
            # 计算损失
            l = mask_loss(output, train_label, risk_mask, data_type=data_type)
            trainer.zero_grad()
            l.backward()
            trainer.step()

            training_loss = l.item()
            # 更新进度条的后缀信息
            train_loader_tqdm.set_postfix({'Loss': f"{training_loss:.6f}"})

            # 可选：记录到 TensorBoard
            # writer.add_scalar('Loss/train', training_loss, global_step)

            global_step += 1

        # 调整学习率
        scheduler.step()

        epoch_time = time() - start_epoch_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s", flush=True)

        # 计算验证损失
        val_loss = compute_loss(net, val_loader, risk_mask, road_adj, risk_adj, grid_node_map, global_step - 1, device, data_type)
        print(f'Global step: {global_step - 1}, Epoch: {epoch}, Validation loss: {val_loss:.6f}', flush=True)

        # 可选：记录到 TensorBoard
        # writer.add_scalar('Loss/val', val_loss, epoch)

        # 每当验证损失有所改进时
        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
                predict_and_evaluate(net, test_loader, risk_mask, road_adj, risk_adj, grid_node_map, global_step - 1, scaler, device)

            print(f'Global step: {global_step - 1}, Epoch: {epoch}, Test RMSE: {test_rmse:.4f}, '
                  f'Test Recall: {test_recall:.2f}%, Test MAP: {test_map:.4f}', flush=True)

            # 可选：记录到 TensorBoard
            # writer.add_scalar('Metrics/RMSE', test_rmse, epoch)
            # writer.add_scalar('Metrics/Recall', test_recall, epoch)
            # writer.add_scalar('Metrics/MAP', test_map, epoch)

        # 早停机制
        early_stop(val_loss, test_rmse, test_recall, test_map,
                   test_inverse_trans_pre, test_inverse_trans_label)
        if early_stop.early_stop:
            print(f"Early Stopping at global step: {global_step}, epoch: {epoch}", flush=True)

            print(f'Best Test RMSE: {early_stop.best_rmse:.4f}, Best Test Recall: {early_stop.best_recall:.2f}%, '
                  f'Best Test MAP: {early_stop.best_map:.4f}', flush=True)

            break

    # 关闭 TensorBoard 记录器（如果使用）
    # writer.close()

    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map





def create_model(config, grid_node_map):
    """
    创建模型的函数。

    参数：
        config: 配置字典。
        grid_node_map: 网格节点映射矩阵。

    返回：
        model: 创建的模型。
    """
    seq_len = config['recent_prior'] + config['week_prior']
    pre_len = config['pre_len']
    spatial_temporal_embedding_dim = 128
    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(config['gcn_num_filter'])

    TDNet_Model = TDNet(seq_len, pre_len, num_of_vertices,
                        config['num_of_input_feature'], spatial_temporal_embedding_dim,
                        north_south_map, west_east_map, grid_node_map)
    return TDNet_Model

def create_optimizer(model, learning_rate):
    """
    创建优化器和学习率调度器的函数。

    参数：
        model: 模型。
        learning_rate: 学习率。

    返回：
        trainer: 优化器。
        scheduler: 学习率调度器。
    """
    trainer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(trainer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    return trainer, scheduler

def main(config):
    # 加载或创建数据集
    saved_data = load_or_create_dataset(config, args)

    scaler = saved_data['scaler']


    # 创建 DataLoader
    batch_size = config['batch_size']
    train_loader, val_loader, test_loader = create_dataloaders(saved_data, batch_size)

    # 创建模型
    TDNet_Model = create_model(config, grid_node_map)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
        TDNet_Model = nn.DataParallel(TDNet_Model)
    TDNet_Model.to(device)
    print(TDNet_Model)

    # 统计参数量
    num_of_parameters = sum(p.numel() for p in TDNet_Model.parameters())
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    # 创建优化器和学习率调度器
    learning_rate = config['learning_rate']
    trainer, scheduler = create_optimizer(TDNet_Model, learning_rate)

    early_stop = EarlyStopping(patience=patience, delta=delta)

    risk_mask = get_mask(mask_filename)
    road_adj = torch.from_numpy(get_adjacent(road_adj_filename)).to(device)
    risk_adj = torch.from_numpy(get_adjacent(risk_adj_filename)).to(device)

    best_rmse, best_recall, best_map = training(
        TDNet_Model,
        training_epoch,
        train_loader,
        val_loader,
        test_loader,
        road_adj,
        risk_adj,
        risk_mask,
        trainer,
        scheduler,  # 添加调度器
        early_stop,
        device,
        scaler,
        data_type=config['data_type']
    )
    return best_rmse, best_recall, best_map

if __name__ == "__main__":
    main(config)