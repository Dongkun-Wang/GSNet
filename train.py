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
from tqdm import tqdm  # 添加导入

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import normal_and_generate_dataset_time,get_mask,get_adjacent,get_grid_node_map_maxtrix
from lib.early_stop import EarlyStopping
from model.TDNet import TDNet
from lib.utils import mask_loss,compute_loss,predict_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file',default='./config/nyc/TDNet_NYC_Config.json')
parser.add_argument("--gpus", type=str,help="test program",default='6')
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
spatial_temporal_embedding_dim = 64

grid_node_map = get_grid_node_map_maxtrix(grid_node_filename)   #TODO：400x243
num_of_vertices = grid_node_map.shape[1]


patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic=True
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

def training(net,
             training_epoch,
             train_loader,
             val_loader,
             test_loader,
             road_adj,
             risk_adj,
             risk_mask,
             trainer,
             early_stop,
             device,
             scaler,
             data_type='nyc'):
    global_step = 1
    for epoch in range(1, training_epoch + 1):
        net.train()
        # 使用 tqdm 包装训练数据加载器
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{training_epoch}", unit="batch")
        for batch, (train_feature, target_time, gragh_feature, train_label) in enumerate(train_loader_tqdm, 1):
            start_time = time()
            train_feature, target_time = train_feature.to(device), target_time.to(device)
            gragh_feature, train_label = gragh_feature.to(device), train_label.to(device)
            # 计算损失
            l = mask_loss(net(train_feature, target_time, gragh_feature, road_adj, risk_adj, grid_node_map),
                          train_label, risk_mask, data_type=data_type)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            training_loss = l.cpu().item()
            # 更新进度条的后缀信息
            train_loader_tqdm.set_postfix({
                'Loss': f"{training_loss:.6f}",
                'Time': f"{time() - start_time:.2f}s"
            })
            global_step += 1

        # 计算验证损失
        val_loss = compute_loss(net, val_loader, risk_mask, road_adj, risk_adj,  grid_node_map, global_step - 1, device, data_type)
        print(f'Global step: {global_step - 1}, Epoch: {epoch}, Validation loss: {val_loss:.6f}', flush=True)

        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
                predict_and_evaluate(net, test_loader, risk_mask, road_adj, risk_adj, grid_node_map, global_step - 1, scaler, device)

            print(f'Global step: {global_step - 1}, Epoch: {epoch}, Test RMSE: {test_rmse:.4f}, '
                  f'Test Recall: {test_recall:.2f}%, Test MAP: {test_map:.4f}', flush=True)

        # 早停机制
        early_stop(val_loss, test_rmse, test_recall, test_map,
                   test_inverse_trans_pre, test_inverse_trans_label)
        if early_stop.early_stop:
            print(f"Early Stopping at global step: {global_step}, epoch: {epoch}", flush=True)

            print(f'Best Test RMSE: {early_stop.best_rmse:.4f}, Best Test Recall: {early_stop.best_recall:.2f}%, '
                  f'Best Test MAP: {early_stop.best_map:.4f}', flush=True)

            break
    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map

def main(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_of_gru_layers = config['num_of_gru_layers']
    gru_hidden_size = config['gru_hidden_size']
    gcn_num_filter = config['gcn_num_filter']

    all_data_filename = config['all_data_filename']

    # 获取文件路径的目录部分
    dir_name = os.path.dirname(all_data_filename)  # 结果为 'data/nyc'

    # 从目录路径中获取城市名称
    city_name = os.path.basename(dir_name)  # 结果为 'nyc'

    data_path = config.get('data_path', './data')  # 数据保存和加载的目录

    # 定义保存数据集的文件名
    dataset_filename = os.path.join(data_path,'{}_saved.pt'.format(city_name))

    # 检查数据集是否已经存在，如果不存在创建dataloader#####################################################
    if os.path.exists(dataset_filename):
        print("从本地加载数据集...")
        # 加载已保存的数据
        saved_data = torch.load(dataset_filename, weights_only=False)
        train_dataset = saved_data['train_dataset']
        val_dataset = saved_data['val_dataset']
        test_dataset = saved_data['test_dataset']
        scaler = saved_data['scaler']
        train_data_shape = saved_data['train_data_shape']
        time_shape = saved_data['time_shape']
        graph_feature_shape = saved_data['graph_feature_shape']
    else:
        print("未找到数据集，正在创建新的数据集...")
        # 确保数据目录存在
        os.makedirs(data_path, exist_ok=True)
        # 存储数据的字典
        saved_data = {}

        # 生成训练、验证和测试数据集
        for idx, (x, y, target_times, scaler) in enumerate(normal_and_generate_dataset_time(
            all_data_filename,
            train_rate=train_rate,
            valid_rate=valid_rate,
            recent_prior=recent_prior,
            week_prior=week_prior,
            one_day_period=one_day_period,
            days_of_week=days_of_week,
            pre_len=pre_len)):

            if args.test:
                x = x[:100]
                y = y[:100]
                target_times = target_times[:100]

            # 处理图数据
            if 'nyc' in all_data_filename:
                graph_x = x[:, :, [0, 46, 47], :, :].reshape(
                    (x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
                graph_x = np.dot(graph_x, grid_node_map)
            elif 'chicago' in all_data_filename:
                graph_x = x[:, :, [0, 39, 40], :, :].reshape(
                    (x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
                graph_x = np.dot(graph_x, grid_node_map)
            else:
                # 如果有其他数据集，可以在这里处理
                pass

            print('idx:', str(idx))
            print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape))
            print("graph_x:", str(graph_x.shape))

            # 仅针对训练数据处理
            if idx == 0:
                saved_scaler = scaler  # 保存 scaler
                train_data_shape = x.shape
                time_shape = target_times.shape
                graph_feature_shape = graph_x.shape

            # 将 NumPy 数组转换为 PyTorch 张量
            x_tensor = torch.from_numpy(x)
            target_times_tensor = torch.from_numpy(target_times)
            graph_x_tensor = torch.from_numpy(graph_x)
            y_tensor = torch.from_numpy(y)

            # 创建 TensorDataset
            dataset = torch.utils.data.TensorDataset(
                x_tensor,
                target_times_tensor,
                graph_x_tensor,
                y_tensor
            )

            # 将数据集添加到字典中
            if idx == 0:
                saved_data['train_dataset'] = dataset
            elif idx == 1:
                saved_data['val_dataset'] = dataset
            elif idx == 2:
                saved_data['test_dataset'] = dataset

        # 保存 scaler 和其他变量
        saved_data['scaler'] = saved_scaler
        saved_data['train_data_shape'] = train_data_shape
        saved_data['time_shape'] = time_shape
        saved_data['graph_feature_shape'] = graph_feature_shape

        # 将所有数据保存到一个文件中
        torch.save(saved_data, dataset_filename)

        # 为了后续使用，赋值给对应的变量
        train_dataset = saved_data['train_dataset']
        val_dataset = saved_data['val_dataset']
        test_dataset = saved_data['test_dataset']
        scaler = saved_data['scaler']

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # 仅对训练数据进行打乱
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    ###创建数据集完毕##########################################################

    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)

    TDNet_Model = TDNet(seq_len,pre_len,graph_feature_shape[3],
                   train_data_shape[2],spatial_temporal_embedding_dim,
                    north_south_map,west_east_map)
    #TODO：多线程用不上
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!",flush=True)
        TDNet_Model = nn.DataParallel(TDNet_Model)
    TDNet_Model.to(device)
    print(TDNet_Model)

    # 统计参数量
    num_of_parameters = 0
    for name,parameters in TDNet_Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)


    trainer = optim.Adam(TDNet_Model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience,delta=delta)
    
    risk_mask = get_mask(mask_filename)
    road_adj = torch.from_numpy(get_adjacent(road_adj_filename)).to(device)
    risk_adj = torch.from_numpy(get_adjacent(risk_adj_filename)).to(device)


    best_mae,best_mse,best_rmse = training(
            TDNet_Model,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            road_adj,
            risk_adj,
            risk_mask,
            trainer,
            early_stop,
            device,
            scaler,
            data_type = config['data_type']
            )
    return best_mae,best_mse,best_rmse

if __name__ == "__main__":
    

    main(config)
