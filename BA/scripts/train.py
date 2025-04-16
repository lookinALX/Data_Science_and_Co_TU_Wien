import os
import argparse
import torch
import numpy as np
import pandas as pd
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from time import time as get_time

from torch.utils.data import Dataset, DataLoader
from scripts.experimentManagerForWin import ExperimentManagerSaveFunction

from efficientSsmLukin.ssm.model import SC_Model_classifier
from efficientSsmLukin.utils import train_one_epoch, evaluate
from efficientSsmLukin.utils import Echo_STDIO_to_File

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

print(torch.__version__)
print(torch.version.cuda)

results_path = './results/'

train_history_csv = 'train_history.csv'
train_sub_epoch_history_csv = 'train_sub_epoch_history.csv'


if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)


parser = argparse.ArgumentParser(description='Keyword spotting')
# Optimizer
parser.add_argument('--epochs', default=70, type=float, help='Training epochs'),
# Device
parser.add_argument('--device', default='cuda:0', type=str,help='Device', choices=['cuda:0', 'cuda:1', 'cpu'])
# Seed
parser.add_argument('--seed', default=1234, type=int, help='Seed')
# Dataloader
parser.add_argument('--num_workers', default=0, type=int,help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')

parser.add_argument('--zeroOrderHoldRegularization', default=[], type=list, help='Number of output classes, per layer')
parser.add_argument('--trainable_skip_connections', default=True, type=bool, help='Trainable skip connections', choices=[True, False])
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

parser.add_argument('--input_bias', default=True, type=bool,help='Input bias', choices=[True, False])
parser.add_argument('--bias_init', default='uniform', type=str,help='Bias initialization', choices=['zero', 'uniform'])
# parser.add_argument('--bias_init', default='zero', type=str,help='Bias initialization', choices=['zero', 'uniform'])      # Only better if B init modified
# parser.add_argument('--output_bias', default=False, type=bool,help='Output bias', choices=[True, False])
parser.add_argument('--output_bias', default=True, type=bool,help='Output bias', choices=[True, False])
parser.add_argument('--complex_output', default=False,type=bool, help='Complex output', choices=[True, False])
# parser.add_argument('--norm', default=False, type=bool,help='Normalization', choices=[True, False])
parser.add_argument('--norm', default=True, type=bool,help='Normalization', choices=[True, False])
parser.add_argument('--norm_type', default='bn', type=str,help='Normalization type', choices=['ln', 'bn'])
parser.add_argument('--stability', default='abs', type=str,help='ensure stability', choices=['relu', 'abs'])
parser.add_argument('--Reduction', default='mean', type=str,help='Reduction', choices=['mean', 'EWMA'])
# parser.add_argument('--augments', default='strong', type=str, help='Augments', choices=['none', 'weak', 'strong'])
parser.add_argument('--augments', default='weak', type=str, help='Augments', choices=['none', 'weak', 'strong'])
# parser.add_argument('--augments', default='none', type=str, help='Augments', choices=['none', 'weak', 'strong'])
parser.add_argument('--act', default='LeakyRELu', type=str, help='Augments', choices=['RELu', 'LeakyRELu', 'Identity'])
parser.add_argument('--weight_decay', default=0.04, type=float, help='Weight decay')
# parser.add_argument('--weight_decay', default=0.00, type=float, help='Weight decay')
#parser.add_argument('--B_C_init', default='S5', type=str, help='Initialization type',choices=['S5','orthogonal', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'])
parser.add_argument('--B_C_init', default='orthogonal', type=str, help='Initialization type',choices=['S5','orthogonal', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'])


# Model definition
parser.add_argument('--input_size', default=1, type=int, help='Input size')
# S-Edge-Tiny 
#parser.add_argument('--hidden_sizes', default=[32, 16, 8], type=list, help='Hidden sizes of the model')
#parser.add_argument('--output_sizes', default=[8, 32, 64], type=list, help='Number of output classes, per layer')
# S-Edge-L
parser.add_argument('--hidden_sizes', default=[96, 80, 64, 48, 32, 16], type=list, help='Hidden sizes of the model')
parser.add_argument('--output_sizes', default=[24, 32, 40, 48, 56, 64], type=list, help='Number of output classes, per layer')

parser.add_argument('--dataset', default='ESC-50', type=str, help='Dataset', choices=['UrbanSound8K', 'CatMeow', 'ESC-50','SpeechCommands', 'AudioMNIST'])
#parser.add_argument('--classes', default=35, type=int, help='number of classes')
# Parse arguments
args = parser.parse_args()
lr = args.lr
epochs = args.epochs
device = args.device
if args.seed == -1:
    seed = np.random.randint(0, 100000)
else:
    seed = args.seed
num_workers = args.num_workers
pin_memory = False
#pin_memory = True if (device == 'cuda:0') or (device == 'cuda:1') else False
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
batch_size = args.batch_size

input_size = args.input_size
#classes = args.classes
dataset = args.dataset
hidden_sizes = args.hidden_sizes
output_sizes = args.output_sizes
zeroOrderHoldRegularization = args.zeroOrderHoldRegularization
trainable_skip_connections = args.trainable_skip_connections
input_bias = args.input_bias
bias_init = args.bias_init
output_bias = args.output_bias
complex_output = args.complex_output
norm = args.norm
norm_type = args.norm_type
B_C_init = args.B_C_init
stability = args.stability
reduction = args.Reduction
augments = args.augments
act = args.act
weight_decay= args.weight_decay
dropout=0.0

if dataset == 'UrbanSound8K':
    classes = 10
elif dataset == 'CatMeow':
    classes = 3
elif dataset == 'ESC-50':
    classes = 50
else:
    raise ValueError("Dataset not implemented")

# save config dict to file
config_dict = {
    'lr': lr,
    'epochs': epochs,
    'device': device,
    'seed': seed,
    'num_workers': num_workers,
    'batch_size': batch_size,
    'input_size': input_size,
    'hidden_sizes': [hidden_sizes],
    'output_sizes': [output_sizes],
    'zeroOrderHoldRegularization': [zeroOrderHoldRegularization],
    'trainable_skip_connections': trainable_skip_connections,
    'input_bias': input_bias,
    'bias_init': bias_init,
    'output_bias': output_bias,
    'complex_output': complex_output,
    'norm': norm,
    'norm_type': norm_type,
    'B_C_init': B_C_init,
    'stability': stability,
    'reduction': reduction,
    'augments': augments,
    'act': act,
    'weight_decay' : weight_decay,
    'dropout': dropout,
    'classes': classes,
    'dataset': dataset
}

model_save_path = ExperimentManagerSaveFunction(path=results_path, config_dict=config_dict, saveFiles=[])

echo_stdio = Echo_STDIO_to_File(os.path.join(model_save_path, 'output.txt'))
sys.stdout = echo_stdio

echo_sterr = Echo_STDIO_to_File(os.path.join(model_save_path, 'error.txt'))
sys.stderr = echo_sterr

print("Echoing to file start")

# Save config dict to file
print("Save config dict to file")
pd.DataFrame.from_dict(config_dict, orient='index').to_csv(os.path.join(model_save_path, 'config.csv'))


device = torch.device(device if torch.cuda.is_available() else "cpu")

# set seed for pytorch and numpy
print(f"Set seed to {seed}")
torch.manual_seed(seed)
np.random.seed(seed)

# Load data
print("Start loading data")

if dataset == 'UrbanSound8K':
    from datasets_utils_lad import UrbanSoundDataset as SUBSET
    dataset_path_meta = 'data\\UrbanSound8K\\metadata\\UrbanSound8K.csv'
    dataset_path_audio = 'data\\UrbanSound8K\\audio'
    path_to_store_preprocessed_data = "data" # path to store preprocessed data preveribly to speed up reading
if dataset == 'CatMeow':
    from datasets_utils_lad import CatMeowDataset as SUBSET
    dataset_path_meta = 'data\\CatMeow\\metadata\\CatMeow_metadata.csv'
    dataset_path_audio = 'data\\CatMeow\\dataset'
    path_to_store_preprocessed_data = "data\\CatMeow_preprocessed"
if dataset == 'ESC-50':
    from datasets_utils_lad import CatMeowDataset as SUBSET
    dataset_path_meta = 'data\\ESC-50-master\\meta\\esc50.csv'
    dataset_path_audio = 'data\\ESC-50-master\\audio'
    path_to_store_preprocessed_data = "data\\ESC-50_preprocessed"

train_set = SUBSET(dataset_path_meta, dataset_path_audio, "training", path = path_to_store_preprocessed_data) 
test_set = SUBSET(dataset_path_meta, dataset_path_audio, "testing", path = path_to_store_preprocessed_data)
valid_set = SUBSET(dataset_path_meta, dataset_path_audio, "validation", path = path_to_store_preprocessed_data)

#train_set = SubsetSC(dataset_path, "training",  path = path_to_store_preprocessed_data) 
#test_set = SubsetSC(dataset_path, "testing",  path = path_to_store_preprocessed_data)
#valid_set = SubsetSC(dataset_path, "validation",  path = path_to_store_preprocessed_data)
print("End loading data")

print("Generate Datalaoder")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

print("Generate Datalaoder end")

# Create model
model = SC_Model_classifier(input_size=input_size,classes=classes, hidden_sizes=hidden_sizes,
                            output_sizes=output_sizes, ZeroOrderHoldRegularization=zeroOrderHoldRegularization,
                            input_bias=input_bias, bias_init=bias_init, output_bias=output_bias,
                            norm=norm, complex_output=complex_output,
                            norm_type=norm_type, B_C_init=B_C_init, stability=stability,
                            trainable_SkipLayer=trainable_skip_connections,
                            act=act,dropout=dropout)
model.to(device)

params_ssm_lr = [param for name, param in model.named_parameters() if 'B' in name or 'C' in name or 'Lambda' in name or 'log_step' in name] 
params_other_lr = [param for name, param in model.named_parameters() if 'B' not in name and 'C' not in name and 'Lambda' not in name and 'log_step' not in name]

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()

params_ssm_lr = [param for name, param in model.named_parameters() if 'B' in name or 'C' in name or 'Lambda' in name or 'log_step' in name] 
params_other_lr = [param for name, param in model.named_parameters() if 'B' not in name and 'C' not in name and 'Lambda' not in name and 'log_step' not in name]

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = torch.optim.AdamW([
    {'params': params_ssm_lr, 'lr': lr, 'weight_decay': 0},
    {'params': params_other_lr, 'lr': 4*lr, 'weight_decay': weight_decay},
    ])

if dataset != "CatMeow":
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                            first_cycle_steps=epochs*len(train_loader),
                                            cycle_mult=1.0,
                                            max_lr=4*lr,
                                            min_lr=0,
                                            warmup_steps=100,
                                            gamma=1,
    )
else:
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                            first_cycle_steps=epochs*len(train_loader),
                                            cycle_mult=1.0,
                                            max_lr=4*lr,
                                            min_lr=0,
                                            warmup_steps=int(epochs*len(train_loader) * 0.1),
                                            gamma=1,
    )


# Train the model
best_val_loss = 1e3  # Init
best_val_loss_epoch = 0
best_val_acc = 0
best_val_acc_epoch = 0

# subsets of the data
df_metric = pd.DataFrame(columns=['train_loss', 'train_acc', 'valid_loss','valid_acc', 'epoch', 'learning_rate', 'training_time'])
df_sub_epoch = pd.DataFrame()


torch.save(model.state_dict(), os.path.join(model_save_path, 'init_model.pt'))
print("Start training")
start_time = get_time()

for epoch in range(epochs):
    train_loss, train_acc, sub_epoch_info = train_one_epoch(model, criterion, optimizer, train_loader, regularize=True, scheduler=scheduler, sub_epoch_documentation=10, augments_use=augments)
    # train_loss, train_acc, sub_epoch_info = train_one_epoch(model, criterion, optimizer, train_loader, regularize=False, scheduler=scheduler, sub_epoch_documentation=10, augments_use=augments)
    
    valid_loss, val_acc = evaluate(model, criterion, valid_loader)

    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        best_val_loss_epoch = epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_valid_loss_model.pt'))
    if val_acc >= best_val_acc:
        best_val_acc_epoch = epoch
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_valid_acc_model.pt'))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}_model.pt'))

    # lr_tmp = optimizer.param_groups[0]['lr']
    df_new_row = {'train_loss': train_loss,
                  'train_acc': train_acc,
                  'valid_loss': valid_loss,
                  'valid_acc': val_acc,
                  'epoch': epoch + 1,
                #   'learning_rate': lr_tmp,
                  'training_time': get_time() - start_time,
                  'learning_rate': scheduler.max_lr,
                  }


    df_metric.loc[epoch] = df_new_row
    df_metric.to_csv(os.path.join(model_save_path, train_history_csv))
    print(f"Epoch {epoch+1}, train_loss={train_loss:6.4f}, train_acc={train_acc:6.4f} val_loss={valid_loss:6.4f},  val_acc={val_acc:6.4f}")

    # save sub epoch info
    new_row = pd.DataFrame(sub_epoch_info)
    new_row['epoch'] =new_row['epoch']+ epoch
    df_sub_epoch = pd.concat([df_sub_epoch, new_row], ignore_index=True).reset_index(drop=True)
    df_sub_epoch.to_csv(os.path.join(model_save_path, train_sub_epoch_history_csv))


print('Evaluate on test set')
model.load_state_dict(torch.load(os.path.join(model_save_path, f'best_valid_loss_model.pt'),weights_only=True))
test_loss, test_acc = evaluate(model, criterion, test_loader)

# save test results to csv
df_test = pd.DataFrame(columns=['test_loss', 'test_acc', 'best_val_loss','best_val_loss_epoch', 'best_val_acc', 'best_val_acc_epoch'])
df_test.loc[0] = [test_loss, test_acc, best_val_loss,best_val_loss_epoch, best_val_acc, best_val_acc_epoch]
df_test.to_csv(os.path.join(model_save_path, 'test_results.csv'))
