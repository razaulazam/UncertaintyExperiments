

# In[ ]:


import torch.optim as optim
import os
import copy
import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
from torchvision.datasets import MNIST
from transform_MNIST import TrainTransform, TestTransform, TransformDataSet, ProduceMixUpAleatoricConfusedHard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from LeNetAleatoricEpistemicMNIST import LeNet5
from helper_functions_MNIST import display_batch_images, calculate_accuracy, heteroscedastic_loss_classification_MN, adjust_learning_rate


# In[ ]:


use_gpu = torch.cuda.is_available()
device = ('cuda:0' if use_gpu else 'cpu:0')
seed = 9999999
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True # For picking up the fastest algorithm of CUDA operations


# In[ ]:


choice_color = {1: 'BLACK', 2: 'WHITE'}
chosen_color = 1
name_color = choice_color[chosen_color]


# In[ ]:

ensemble_num = 8
drop_rate = 0.5
curr_dir = os.getcwd()
data_dir = '/fs/scratch/rng_cr_bcai_dl_students/OpenData/'
save_model_checkpoints = curr_dir + '/training_LeNet5_standard_aleatoric_epistemic_%.1f_%s_hard_horizontal_1_3_en' % (drop_rate, name_color)
save_model_ensemble = '/ensemble_%d' % ensemble_num
save_tensorboard_logs = curr_dir + '/training_LeNet5_standard_aleatoric_epistemic_%.1f_%s_hard_horizontal_1_3_en_logs' % (drop_rate, name_color)
save_tensorboard_logs_ensemble = '/ensemble_%d' % ensemble_num

# In[ ]:


data_training = MNIST(root=data_dir, train=True)


# In[ ]:

print('>>>> MAKING A NEW DIRECTORIES <<<<')

os.makedirs(save_tensorboard_logs, exist_ok=True)
os.makedirs(save_model_checkpoints, exist_ok=True)
os.makedirs(save_model_checkpoints + save_model_ensemble, exist_ok=True)
os.makedirs(save_tensorboard_logs + save_tensorboard_logs_ensemble, exist_ok=True)


# In[ ]:

# In[ ]:


writer = SummaryWriter(log_dir=(save_tensorboard_logs + save_tensorboard_logs_ensemble))


# In[ ]:


mean = data_training.data.float().mean()
std = data_training.data.float().std()

mean = mean.numpy()
std = std.numpy()


# In[ ]:


batch_size = 64
total_epochs = 35
best_accuracy = 0.0
initial_lr = 0.01

data_validation = MNIST(root=data_dir, train=False, transform=TestTransform(mean, std))



# In[ ]:


print('>>>> NUMBER OF TRAINING EXAMPLES = {0}'.format(len(data_training)))
print('>>>> NUMBER OF VALIDATION EXAMPLES = {0}'.format(len(data_validation)))


# In[ ]:

# 4th Argument : 1 - Horizontal MixUp, 2 - Vertical MixUp (in Consufed Hard cases)

data_training = ProduceMixUpAleatoricConfusedHard(data_training, 1, 3, 1, TrainTransform(mean, std))


# In[ ]:


print('>>>> NUMBER OF NEW TRAINING EXAMPLES = {0}'.format(len(data_training)))
print('>>>> NUMBER OF NEW VALIDATION EXAMPLES = {0}'.format(len(data_validation)))


# In[ ]:


training_data_loader = DataLoader(data_training, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
validation_data_loader = DataLoader(data_validation, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


# In[ ]:


net = LeNet5(10, layer_drop=drop_rate)

if torch.cuda.device_count() > 1:
    print('>>>> PARALLEZING THE MODEL ON MULTIPLE GPUS <<<<')
    net = nn.DataParallel(net)

net.to(device)


# In[ ]:


print('>>>> DROPOUT RATE = %.1f <<<<' % drop_rate)


# In[ ]:


optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4)
criterion = heteroscedastic_loss_classification_MN()
criterion_validation = nn.CrossEntropyLoss()


# In[ ]:


def train():
    print('>>>> TRAINING ...')
    net.train()
    epoch_loss = 0.0
    accuracy = 0.0
    
    for data in training_data_loader:
        image = data[0].to(device, non_blocking=True)
        label = data[1].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        output = net(image)
        logits = output[:, 0, :]
        
        prob_output = F.softmax(logits, dim=1)
        
        epoch_accuracy = calculate_accuracy(pred=prob_output, label_true=label)
        loss = criterion(output, label, device) 
        
        epoch_loss += loss.item()
        accuracy += epoch_accuracy.item()
        
        loss.backward()
        optimizer.step()
    
    average_epoch_loss = epoch_loss / len(training_data_loader)
    average_accuracy = accuracy / len(training_data_loader)
    
    return average_epoch_loss, average_accuracy 


# In[ ]:


def validation():
    print('>>>> PERFORMING THE VALIDATION STEP <<<<')
    net.eval()
    epoch_loss = 0.0
    accuracy = 0.0
    
    for data in validation_data_loader:
        with torch.no_grad():
            image = data[0].to(device, non_blocking=True)
            label = data[1].to(device, non_blocking=True).long()
        
            output = net(image)
            logits = output[:, 0, :]
            
            prob_output = F.softmax(logits, dim=1)

            loss = criterion_validation(logits, label).to(device)
            epoch_accuracy = calculate_accuracy(pred=prob_output, label_true=label)
        
            epoch_loss += loss.item()
            accuracy += epoch_accuracy.item()
    
    average_epoch_loss = epoch_loss / len(validation_data_loader)
    average_accuracy = accuracy / len(validation_data_loader)
    
    return average_epoch_loss, average_accuracy 


# In[ ]:


print('>>>> TRAINING BEGINS')

for epoch in range(total_epochs):
    
    adjust_learning_rate(optimizer, initial_lr, epoch)
    
    train_loss, train_accuracy = train()
    validation_loss, validation_accuracy = validation()
    
    print('>>>> EPOCH = {0}'.format(epoch+1))
    print('>>>> TRAINING LOSS = {0:.4f}, TRAINING ACCURACY = {1:.4f}'.format(train_loss, train_accuracy))
    print('>>>> VALIDATION LOSS = {0:.4f}, VALIDATION ACCURACY = {1:.4f}'.format(validation_loss, validation_accuracy))
    print('\n')
    
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', validation_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/Validation', validation_accuracy, epoch)

    
    if validation_accuracy > best_accuracy:
        print('>>>> SAVING THE BEST MODEL <<<<')
        best_accuracy = validation_accuracy
        checkpoint = {
            'epoch': epoch,
            'acc': validation_accuracy,
            'state_dict': net.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.cuda.get_rng_state_all()
        }

        os.chdir(save_model_checkpoints + save_model_ensemble)
        for filename in os.listdir():
            os.unlink(filename)

        os.chdir(curr_dir)

        model_path = os.path.join((save_model_checkpoints + save_model_ensemble), 'BEST_MODEL_EPOCH_'+str(epoch)+'.tar')
        torch.save(checkpoint, model_path)
        
        with open('metrics_model_aleatoric_epistemic_%.1f_hard_horizontal_1_3_en_%d.txt' % (drop_rate, ensemble_num), 'w+') as f:
            f.write('BEST EPOCH METRICS:\n')
            f.write('VALIDATION LOSS: {0:.4f}\n'. format(validation_loss))
            f.write('VALIDATION ACCURACY: {0:.4f}\n'. format(validation_accuracy))
            f.write('TRAINING LOSS: {0:.4f}\n'. format(train_loss))
            f.write('TRAINING ACCURACY: {0:.4f}\n'. format(train_accuracy))            



# %%
