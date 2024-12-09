import numpy as np
import pickle as pkl
import configparser
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
from lib.utils import Scaler_NYC,Scaler_Chi

num_workers = 4

def split_and_norm_data(all_data,
                        train_rate = 0.6,
                        valid_rate = 0.2,
                        recent_prior=3,
                        week_prior=4,
                        one_day_period=24,
                        days_of_week=7,
                        pre_len=1):
    num_of_time,channel,_,_ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate+valid_rate))
    for index,(start,end) in enumerate(((0,train_line),(train_line,valid_line),(valid_line,num_of_time))):
        if index == 0:
            if channel == 48:#NYC
                scaler = Scaler_NYC(all_data[start:end,:,:,:])
            if channel == 41:#Chicago
                scaler = Scaler_Chi(all_data[start:end,:,:,:])
        norm_data = scaler.transform(all_data[start:end,:,:,:])
        X,Y = [],[]
        for i in range(len(norm_data)-week_prior*days_of_week*one_day_period-pre_len+1):
            t = i+week_prior*days_of_week*one_day_period
            label = norm_data[t:t+pre_len,0,:,:]
            period_list = []
            for week in range(week_prior):
                period_list.append(i+week*days_of_week*one_day_period)
            for recent in list(range(1,recent_prior+1))[::-1]:
                period_list.append(t-recent)
            feature = norm_data[period_list,:,:,:]
            X.append(feature)
            Y.append(label)
        yield np.array(X),np.array(Y),scaler


def normal_and_generate_dataset(
        all_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    """
    
    Arguments:
        all_data_filename {str} -- all data filename
    
    Keyword Arguments:
        train_rate {float} -- train rate (default: {0.6})
        valid_rate {float} -- valid rate (default: {0.2})
        recent_prior {int} -- the length of recent time (default: {3})
        week_prior {int} -- the length of week  (default: {4})
        one_day_period {int} -- the number of time interval in one day (default: {24})
        days_of_week {int} -- a week has 7 days (default: {7})
        pre_len {int} -- the length of prediction time interval(default: {1})

    Yields:
        {np.array} -- 
                      X shape：(num_of_sample,seq_len,D,W,H)
                      Y shape：(num_of_sample,pre_len,W,H)
        {Scaler} -- train data max/min
    """
    risk_taxi_time_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32)

    for i in split_and_norm_data(risk_taxi_time_data,
                        train_rate = train_rate,
                        valid_rate = valid_rate,
                        recent_prior = recent_prior,
                        week_prior = week_prior,
                        one_day_period = one_day_period,
                        days_of_week = days_of_week,
                        pre_len = pre_len):
        yield i 

def split_and_norm_data_time(all_data,
                        train_rate = 0.6,
                        valid_rate = 0.2,
                        recent_prior=3,
                        week_prior=4,
                        one_day_period=24,
                        days_of_week=7,
                        pre_len=1):
    num_of_time,channel,_,_ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate+valid_rate))
    for index,(start,end) in enumerate(((0,train_line),(train_line,valid_line),(valid_line,num_of_time))):
        if index == 0:
            if channel == 48:
                scaler = Scaler_NYC(all_data[start:end,:,:,:])
            if channel == 41:
                scaler = Scaler_Chi(all_data[start:end,:,:,:])
        norm_data = scaler.transform(all_data[start:end,:,:,:])
        X,Y,target_time = [],[],[]
        for i in range(len(norm_data)-week_prior*days_of_week*one_day_period-pre_len+1):
            t = i+week_prior*days_of_week*one_day_period
            label = norm_data[t:t+pre_len,0,:,:]
            period_list = []
            for week in range(week_prior):
                period_list.append(i+week*days_of_week*one_day_period)
            for recent in list(range(1,recent_prior+1))[::-1]:
                period_list.append(t-recent)
            feature = norm_data[period_list,:,:,:]
            X.append(feature)
            Y.append(label)
            target_time.append(norm_data[t,1:33,0,0])

        yield np.array(X),np.array(Y),np.array(target_time),scaler


def normal_and_generate_dataset_time(
        all_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    all_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32)

    for i in split_and_norm_data_time(all_data,
                        train_rate = train_rate,
                        valid_rate = valid_rate,
                        recent_prior = recent_prior,
                        week_prior = week_prior,
                        one_day_period = one_day_period,
                        days_of_week = days_of_week,
                        pre_len = pre_len):
        yield i 

def get_mask(mask_path):
    """
    Arguments:
        mask_path {str} -- mask filename
    
    Returns:
        {np.array} -- mask matrix，维度(W,H)
    """
    mask = pkl.load(open(mask_path,'rb')).astype(np.float32)
    return mask

def get_adjacent(adjacent_path):
    """
    Arguments:
        adjacent_path {str} -- adjacent matrix path
    
    Returns:
        {np.array} -- shape:(N,N)
    """
    adjacent = pkl.load(open(adjacent_path,'rb')).astype(np.float32)
    return adjacent

def get_grid_node_map_maxtrix(grid_node_path):
    """
    Arguments:
        grid_node_path {str} -- filename
    
    Returns:
        {np.array} -- shape:(W*H,N)
    """
    grid_node_map = pkl.load(open(grid_node_path,'rb')).astype(np.float32)
    return grid_node_map 

def load_or_create_dataset(config, args):
    """
    加载或创建数据集的函数。

    参数：
        config: 配置字典。
        args: 命令行参数。

    返回：
        saved_data: 包含数据集和相关信息的字典。
    """
    batch_size = config['batch_size']
    train_rate = config['train_rate']
    valid_rate = config['valid_rate']
    recent_prior = config['recent_prior']
    week_prior = config['week_prior']
    one_day_period = config['one_day_period']
    days_of_week = config['days_of_week']
    pre_len = config['pre_len']
    all_data_filename = config['all_data_filename']
    grid_node_filename = config['grid_node_filename']
    grid_node_map = get_grid_node_map_maxtrix(grid_node_filename)  # TODO：400x243

    # 获取文件路径的目录部分
    dir_name = os.path.dirname(all_data_filename)
    # 从目录路径中获取城市名称
    city_name = os.path.basename(dir_name)

    data_path = config.get('data_path', './data')  # 数据保存和加载的目录
    # 定义保存数据集的文件名
    dataset_filename = os.path.join(data_path, f'{city_name}_saved.pt')

    if os.path.exists(dataset_filename):
        print("从本地加载数据集...")
        saved_data = torch.load(dataset_filename, weights_only=False)
    else:
        print("未找到数据集，正在创建新的数据集...")
        # 确保数据目录存在
        os.makedirs(data_path, exist_ok=True)
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
            graph_x = process_graph_x(x, grid_node_map, city_name)

            print('idx:', str(idx))
            print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape))
            print("graph_x:", str(graph_x.shape))

            # 仅针对训练数据处理
            if idx == 0:
                saved_scaler = scaler  # 保存 scaler
                train_data_shape = x.shape
                time_shape = target_times.shape
                graph_feature_shape = graph_x.shape

            # 将 NumPy 数组转换为 PyTorch 张量，并指定数据类型
            x_tensor = torch.from_numpy(x).float()
            target_times_tensor = torch.from_numpy(target_times).float()
            graph_x_tensor = torch.from_numpy(graph_x).float()
            y_tensor = torch.from_numpy(y).float()

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

    return saved_data

def process_graph_x(x, grid_node_map, dataset_name):
    """
    处理图数据的函数，根据数据集名称选择处理方式。

    参数：
        x: 输入数据。
        grid_node_map: 网格节点映射矩阵。
        dataset_name: 数据集名称。

    返回：
        graph_x: 处理后的图数据。
    """
    north_south_map = 20
    west_east_map = 20

    if dataset_name == 'nyc':
        indices = [0, 46, 47]
    elif dataset_name == 'chicago':
        indices = [0, 39, 40]
    else:
        # 如果有其他数据集，可以在这里处理
        indices = [0]  # 默认值，需要根据实际情况修改

    graph_x = x[:, :, indices, :, :].reshape(
        (x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
    graph_x = np.dot(graph_x, grid_node_map)
    return graph_x

def create_dataloaders(saved_data, batch_size):
    """
    创建 DataLoader 的函数。

    参数：
        saved_data: 包含数据集的字典。
        batch_size: 批次大小。

    返回：
        train_loader, val_loader, test_loader: 数据加载器。
    """
    train_dataset = saved_data['train_dataset']
    val_dataset = saved_data['val_dataset']
    test_dataset = saved_data['test_dataset']

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers  # 使用配置中的 num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader