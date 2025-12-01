import os
import shutil
import mpl_scatter_density
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torch
import wandb
import math


def make_dir(path):
    try: 
        os.mkdir(path)
    except: 
        pass


def build_folder_and_archive(path):
    working_dir = os.getcwd()
    ori_full_path = os.path.join(working_dir, path)
    tgt_full_path = os.path.join(working_dir, "Archives", path[:-1] + f"_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    check = os.path.exists(ori_full_path)
    if check:
        shutil.move(ori_full_path, tgt_full_path)
        os.makedirs(ori_full_path)
    else:
        os.makedirs(ori_full_path)

    
def build_folder_and_clean(path):
    check = os.path.exists(path)
    if check:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


# check if sweep finished
def get_sweep_status(entity, project, sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    return sweep.state
        
        
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    
    if not torch.cuda.is_available(): 
        torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_settings(required_settings, settings):
    for item in required_settings:
        if item in settings.keys():
            required_settings[item] = settings[item]
    return required_settings


# Returns the closest number that is a power of 2 to the given real number x
def closest_power_of_2(x):
    return 2 ** round(math.log2(x))


# Returns a list of n numbers that are evenly spaced between a and b.
def evenly_spaced_numbers(a, b, n):
    if n == 1:
        return [(a+b)/2]
    step = (b-a)/(n-1)
    return [a + i*step for i in range(n)]
    
    
# generate a V-shape MLP as torch.nn.Sequential given input_size, output_size, and layer_count(only linear layer counted)
def generate_sequential(a, b, n, dropout):
    layer_sizes = evenly_spaced_numbers(a, b, n)
    layer_sizes = [int(layer_sizes[0])] + [int(closest_power_of_2(x)) for x in layer_sizes[1:-1]] + [int(layer_sizes[-1])]
    
    layers = []
    for i in range(n-1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i == 0:
            layers.append(torch.nn.LeakyReLU(0.1))
        elif 0 < i < n-2:
            layers.append(torch.nn.LeakyReLU(0.1))
    
    model = torch.nn.Sequential(*layers)
    return model


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def length_to_mask(lengths, total_len, device):
    max_len = total_len
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1)
    return mask.unsqueeze(-2)
    
    
def save_square_img(contents, xlabel, ylabel, savename, title):
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    
    plt.clf()
    plt.rcParams['font.size'] = 15
    
    max_value = max(contents[0].max(), contents[1].max())
    min_value = min(contents[0].min(), contents[1].min())
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    fig.set_size_inches(7, 6)
    ax.set_position([0, 0, 0.8, 1])
    
    density = ax.scatter_density(contents[0], contents[1], cmap=white_viridis)
    fig.colorbar(density, label='Number of points')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    
    ax.set_xlim([min_value, max_value])
    ax.set_ylim([min_value, max_value])
    ax.plot([min_value, max_value], [min_value, max_value], color='k')
    fig.savefig("%s.png" %(savename))
    plt.close(fig)


def save_csv(contents, savename):
    targets_ex = torch.from_numpy(contents[0]).unsqueeze(1)
    output_ex = torch.from_numpy(contents[1]).unsqueeze(1)
    diff_ex = targets_ex - output_ex
    pd_out = pd.DataFrame(
        torch.cat(
            (targets_ex, output_ex, diff_ex), 1
        ).numpy()
    )
    pd_out.columns = ['Target', 'Output', 'Diff']
    pd_out.to_csv(f'{savename}.csv', index=False)
    
    
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().detach().cpu().numpy())
            max_grads.append(p.grad.abs().max().detach().cpu().numpy())
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.tick_params(axis='x', labelsize=8)    # 设置x轴标签大小
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
    