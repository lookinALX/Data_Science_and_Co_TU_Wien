import os
import argparse
import torch
import numpy as np
import pandas as pd
import sys
import copy
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from time import time as get_time

from torch.utils.data import Dataset, DataLoader
from scripts.experimentManagerForWin import ExperimentManagerSaveFunction
#if linux use --> from experimentManager import ExperimentManagerSaveFunction
from efficientSsmLukin.ssm.model import SC_Model_classifier
from efficientSsmLukin.utils import train_one_epoch, evaluate
from efficientSsmLukin.utils import Echo_STDIO_to_File

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

print(torch.__version__)
print(torch.version.cuda)

results_path = './results/'

train_history_csv = 'train_history.csv'
train_sub_epoch_history_csv = 'train_sub_epoch_history.csv'
cv_results_csv = 'cv_results.csv'

if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

parser = argparse.ArgumentParser(description='Keyword spotting with 10-fold cross-validation')
# optimizer
parser.add_argument('--epochs', default=50, type=float, help='Training epochs'),
# device
parser.add_argument('--device', default='cuda:0', type=str,help='Device', choices=['cuda:0', 'cuda:1', 'cpu'])
# seed
parser.add_argument('--seed', default=1234, type=int, help='Seed')
# dataloader
parser.add_argument('--num_workers', default=0, type=int,help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')

parser.add_argument('--zeroOrderHoldRegularization', default=[], type=list, help='Number of output classes, per layer')
parser.add_argument('--trainable_skip_connections', default=True, type=bool, help='Trainable skip connections', choices=[True, False])
parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')

parser.add_argument('--input_bias', default=True, type=bool,help='Input bias', choices=[True, False])
parser.add_argument('--bias_init', default='uniform', type=str,help='Bias initialization', choices=['zero', 'uniform'])
parser.add_argument('--output_bias', default=True, type=bool,help='Output bias', choices=[True, False])
parser.add_argument('--complex_output', default=False,type=bool, help='Complex output', choices=[True, False])
parser.add_argument('--norm', default=True, type=bool,help='Normalization', choices=[True, False])
parser.add_argument('--norm_type', default='bn', type=str,help='Normalization type', choices=['ln', 'bn'])
parser.add_argument('--stability', default='abs', type=str,help='ensure stability', choices=['relu', 'abs'])
parser.add_argument('--Reduction', default='mean', type=str,help='Reduction', choices=['mean', 'EWMA'])
parser.add_argument('--augments', default='none', type=str, help='Augments', choices=['none', 'weak', 'strong'])
parser.add_argument('--act', default='LeakyRELu', type=str, help='Augments', choices=['RELu', 'LeakyRELu', 'Identity'])
parser.add_argument('--weight_decay', default=0.04, type=float, help='Weight decay')
parser.add_argument('--B_C_init', default='orthogonal', type=str, help='Initialization type',choices=['S5','orthogonal', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'])
parser.add_argument('--use_validation', default=False, type=bool, help='Use validation set (1 fold)', choices=[True, False])

# model definition
parser.add_argument('--input_size', default=1, type=int, help='Input size')
# S-Edge-L
parser.add_argument('--hidden_sizes', default=[96, 80, 64, 48, 32, 16], type=list, help='Hidden sizes of the model')
parser.add_argument('--output_sizes', default=[24, 32, 40, 48, 56, 64], type=list, help='Number of output classes, per layer')

parser.add_argument('--dataset', default='UrbanSound8K', type=str, help='Dataset', choices=['UrbanSound8K'])
# parse arguments
args = parser.parse_args()

# extract arguments
lr = args.lr
epochs = args.epochs
device = args.device
if args.seed == -1:
    seed = np.random.randint(0, 100000)
else:
    seed = args.seed
num_workers = args.num_workers
pin_memory = True if (device == 'cuda:0') or (device == 'cuda:1') else False
batch_size = args.batch_size

input_size = args.input_size
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
weight_decay = args.weight_decay
dropout = 0.0
use_validation = args.use_validation

# UrbanSound8K specific parameters
classes = 10

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
    'weight_decay': weight_decay,
    'dropout': dropout,
    'classes': classes,
    'dataset': dataset,
    'use_validation': use_validation
}

model_save_path = ExperimentManagerSaveFunction(path=results_path, config_dict=config_dict, saveFiles=[])

echo_stdio = Echo_STDIO_to_File(os.path.join(model_save_path, 'output.txt'))
sys.stdout = echo_stdio

echo_sterr = Echo_STDIO_to_File(os.path.join(model_save_path, 'error.txt'))
sys.stderr = echo_sterr

print("Echoing to file start")

# save config dict to file
print("Save config dict to file")
pd.DataFrame.from_dict(config_dict, orient='index').to_csv(os.path.join(model_save_path, 'config.csv'))

device = torch.device(device if torch.cuda.is_available() else "cpu")

# set seed for reproducibility
print(f"Set seed to {seed}")
torch.manual_seed(seed)
np.random.seed(seed)

# define paths to the dataset
dataset_path_meta = 'data\\UrbanSound8K\\metadata\\UrbanSound8K.csv'
dataset_path_audio = 'data\\UrbanSound8K\\audio'

from datasets_utils_lad import UrbanSoundDataset

class UrbanSoundCV(Dataset):
    def __init__(self, metadata_file, audio_dir, train_folds, test_fold, 
                 device='cpu', sample_rate=16000, transform=None, 
                 path='data\\UrbanSound8K_preprocessed'):
        
        self.metadata_file = metadata_file
        self.audio_dir = audio_dir
        self.train_folds = train_folds
        self.test_fold = test_fold
        self.device = device
        self.sample_rate = sample_rate
        self.transform = transform
        self.path = path
        
        os.makedirs(self.path, exist_ok=True)
        self.metadata = pd.read_csv(self.metadata_file)

        self.named_labels = sorted(self.metadata['class'].unique())
        self.class_mapping = {label: idx for idx, label in enumerate(self.named_labels)}
        
    def get_train_loader(self, batch_size, num_workers=0, pin_memory=False):
        train_metadata = self.metadata[self.metadata['fold'].isin(self.train_folds)]
        train_set = UrbanSoundDataset(
            self.metadata_file, 
            self.audio_dir, 
            subset="custom", 
            device='cpu',
            sample_rate=self.sample_rate,
            transform=self.transform,
            path=self.path,
            custom_metadata=train_metadata
        )
        return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=pin_memory)
    
    def get_test_loader(self, batch_size, num_workers=0, pin_memory=False):
        test_metadata = self.metadata[self.metadata['fold'] == self.test_fold]
        test_set = UrbanSoundDataset(
            self.metadata_file, 
            self.audio_dir, 
            subset="custom", 
            device='cpu',
            sample_rate=self.sample_rate,
            transform=self.transform,
            path=self.path,
            custom_metadata=test_metadata
        )
        return torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=pin_memory)
    
    def get_validation_loader(self, val_fold, batch_size, num_workers=0, pin_memory=False):
        val_metadata = self.metadata[self.metadata['fold'] == val_fold]
        val_set = UrbanSoundDataset(
            self.metadata_file, 
            self.audio_dir, 
            subset="custom", 
            device='cpu',
            sample_rate=self.sample_rate,
            transform=self.transform,
            path=self.path,
            custom_metadata=val_metadata
        )
        return torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=pin_memory)


# function to create and train a model for a specific fold
def train_fold(fold, metadata_file, audio_dir, device, batch_size, num_workers, pin_memory, 
               epochs, lr, weight_decay, augments, use_validation=False):
    
    fold_save_path = os.path.join(model_save_path, f'fold_{fold}')
    os.makedirs(fold_save_path, exist_ok=True)
    
    test_fold = fold
    all_folds = list(range(1, 11))
    train_folds = [f for f in all_folds if f != test_fold]
    
    if use_validation:
        val_fold = train_folds.pop()  # remove last fold for validation
    
    # create dataset and loaders
    cv_dataset = UrbanSoundCV(
        metadata_file, 
        audio_dir, 
        train_folds, 
        test_fold, 
        device=device,
        sample_rate=16000
    )
    
    train_loader = cv_dataset.get_train_loader(batch_size, num_workers, pin_memory)
    test_loader = cv_dataset.get_test_loader(batch_size, num_workers, pin_memory)
    
    if use_validation:
        valid_loader = cv_dataset.get_validation_loader(val_fold, batch_size, num_workers, pin_memory)
    
    # Create model
    model = SC_Model_classifier(
        input_size=input_size,
        classes=classes, 
        hidden_sizes=hidden_sizes,
        output_sizes=output_sizes, 
        ZeroOrderHoldRegularization=zeroOrderHoldRegularization,
        input_bias=input_bias, 
        bias_init=bias_init, 
        output_bias=output_bias,
        norm=norm, 
        complex_output=complex_output,
        norm_type=norm_type, 
        B_C_init=B_C_init, 
        stability=stability,
        trainable_SkipLayer=trainable_skip_connections,
        act=act,
        dropout=dropout
    )
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

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                            first_cycle_steps=epochs*len(train_loader),
                                            cycle_mult=1.0,
                                            max_lr=4*lr,
                                            min_lr=0,
                                            warmup_steps=100,
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
    
    torch.save(model.state_dict(), os.path.join(fold_save_path, 'init_model.pt'))
    
    print(f"Start training fold {fold}")
    start_time = get_time()
    
    for epoch in range(epochs):
        # Train the model
        train_loss, train_acc, sub_epoch_info = train_one_epoch(
            model, criterion, optimizer, train_loader, 
            regularize=True, scheduler=scheduler, 
            sub_epoch_documentation=10, augments_use=augments
        )
        
        # Evaluate on validation set if using validation
        if use_validation:
            valid_loss, val_acc = evaluate(model, criterion, valid_loader)
            
            # save best models based on validation performance
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_loss_epoch = epoch
                torch.save(model.state_dict(), os.path.join(fold_save_path, 'best_valid_loss_model.pt'))
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_acc_epoch = epoch
                torch.save(model.state_dict(), os.path.join(fold_save_path, 'best_valid_acc_model.pt'))
        else:
            # when not using validation, use training metrics
            valid_loss = train_loss
            val_acc = train_acc
            
            # save model every epoch as we don't have validation
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(fold_save_path, f'epoch_{epoch}_model.pt'))
            
            # save final model
            if epoch == epochs - 1:
                torch.save(model.state_dict(), os.path.join(fold_save_path, 'final_model.pt'))
        
        df_new_row = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': val_acc,
            'epoch': epoch + 1,
            'learning_rate': scheduler.max_lr,
            'training_time': get_time() - start_time,
        }
        
        df_metric.loc[epoch] = df_new_row
        df_metric.to_csv(os.path.join(fold_save_path, train_history_csv))
        
        print(f"Fold {fold}, Epoch {epoch+1}, train_loss={train_loss:6.4f}, "
              f"train_acc={train_acc:6.4f} val_loss={valid_loss:6.4f}, val_acc={val_acc:6.4f}")
        
        # save sub-epoch info
        new_row = pd.DataFrame(sub_epoch_info)
        new_row['epoch'] = new_row['epoch'] + epoch
        df_sub_epoch = pd.concat([df_sub_epoch, new_row], ignore_index=True).reset_index(drop=True)
        df_sub_epoch.to_csv(os.path.join(fold_save_path, train_sub_epoch_history_csv))
    
    # evaluate on test set
    if use_validation:
        model_path = os.path.join(fold_save_path, 'best_valid_acc_model.pt')
    else:
        model_path = os.path.join(fold_save_path, 'final_model.pt')
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_loss, test_acc = evaluate(model, criterion, test_loader)
    
    # save test results
    results = {
        'fold': fold,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }
    
    if use_validation:
        results.update({
            'best_val_loss': best_val_loss,
            'best_val_loss_epoch': best_val_loss_epoch,
            'best_val_acc': best_val_acc,
            'best_val_acc_epoch': best_val_acc_epoch,
        })
    
    print(f"Fold {fold} complete. Test accuracy: {test_acc:.4f}")
    
    return results


# run 10-fold cross-validation
def run_cross_validation():
    all_results = []
    
    print("Starting 10-fold cross-validation")
    
    for fold in range(1, 11):
        print(f"\n--- Starting fold {fold} ---\n")
        
        fold_results = train_fold(
            fold=fold,
            metadata_file=dataset_path_meta,
            audio_dir=dataset_path_audio,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            augments=augments,
            use_validation=use_validation
        )
        
        all_results.append(fold_results)
        
        # save intermediate results
        cv_df = pd.DataFrame(all_results)
        cv_df.to_csv(os.path.join(model_save_path, cv_results_csv))
    
    cv_df = pd.DataFrame(all_results)
    
    mean_results = cv_df.mean(numeric_only=True)
    std_results = cv_df.std(numeric_only=True)
    
    cv_df.loc['mean'] = mean_results
    cv_df.loc['std'] = std_results

    cv_df.to_csv(os.path.join(model_save_path, cv_results_csv))
    
    print("\n--- Cross-validation complete ---\n")
    print(f"Mean test accuracy: {mean_results['test_acc']:.4f} ± {std_results['test_acc']:.4f}")
    
    return cv_df


if __name__ == "__main__":
    # run the cross-validation
    cv_results = run_cross_validation()
    
    # display final results
    print("\nCross-Validation Results Summary:")
    print(cv_results.tail(2))
    
    # save a summary text file
    with open(os.path.join(model_save_path, 'cv_summary.txt'), 'w') as f:
        f.write("UrbanSound8K 10-fold Cross-Validation Results\n")
        f.write("-----------------------------------------\n\n")
        f.write(f"Mean Test Accuracy: {cv_results.loc['mean']['test_acc']:.4f} ± {cv_results.loc['std']['test_acc']:.4f}\n")
        f.write(f"Mean Test Loss: {cv_results.loc['mean']['test_loss']:.4f} ± {cv_results.loc['std']['test_loss']:.4f}\n")
        
        if use_validation:
            f.write(f"Mean Best Validation Accuracy: {cv_results.loc['mean']['best_val_acc']:.4f} ± {cv_results.loc['std']['best_val_acc']:.4f}\n")
        
        f.write("\nIndividual Fold Results:\n")
        for fold in range(1, 11):
            fold_row = cv_results[cv_results['fold'] == fold].iloc[0]
            f.write(f"Fold {fold}: Test Accuracy = {fold_row['test_acc']:.4f}, Test Loss = {fold_row['test_loss']:.4f}\n")