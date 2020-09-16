import torch
import shutil
import torch.nn as nn
from train_data_loader_deepLabv3 import train_loader
from validation_data_loader_deepLabv3 import validation_loader
from torch.utils.tensorboard import SummaryWriter
from helper_functions_deepLabv3 import loss_classification_standard, AverageMeter
from helper_functions_deepLabv3 import poly_learning_rate, intersectionAndUnionGPU
from deepLabv3_standard import DeepLab
from collections import OrderedDict
import torch.optim as optim
import numpy as np
import os
import math

use_gpu = torch.cuda.is_available()
device = ('cuda:0' if use_gpu else 'cpu')

curr_dir = os.getcwd()
devices_avail = [0, 1, 2, 3]

backbone = 'resnet'

data_dir = '/fs/scratch/rng_cr_bcai_dl_students/OpenData/cityscapes/'
save_model_checkpoints = curr_dir + '/deepLabv3_training_standard_pretrained_%s_separable' % backbone
log_dir_tensor_board = curr_dir + '/deepLabv3_training_logs_standard_pretrained_%s_separable' % backbone

if os.path.isdir(save_model_checkpoints):
    print('>>>> REMOVING AND MAKING THE DIRECTORY FOR MODEL CHECKPOINTS AGAIN <<<<\n')
    shutil.rmtree(save_model_checkpoints)
    os.mkdir(save_model_checkpoints)
else:
    print('>>>> MAKING THE DIRECTORY FOR MODEL CHECKPOINTS <<<<\n')
    os.mkdir(save_model_checkpoints)

if os.path.isdir(log_dir_tensor_board):
    print('>>>> REMOVING AND MAKING THE DIRECTORY FOR TENSOR-BOARD LOGS AGAIN <<<<\n')
    shutil.rmtree(log_dir_tensor_board)
    os.mkdir(log_dir_tensor_board)
else:
    print('>>>> MAKING THE DIRECTORY FOR TENSOR-BOARD LOGS <<<<\n')
    os.mkdir(log_dir_tensor_board)


writer = SummaryWriter(log_dir=log_dir_tensor_board)

class_encodings = OrderedDict([
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32))])

batch_size = 8
total_epochs = 230
learning_rate = 0.01
weight_decay = 0.0005
num_classes = 19
ignore_index = 255
power = 0.9
momentum = 0.9
start_epoch = 0
best_miou = 0

train_data_loader = train_loader(data_dir=data_dir, batch_size=batch_size)
validation_data_loader = validation_loader(data_dir=data_dir, batch_size=batch_size)

net = DeepLab()

train_params = [{'params': net.get_1x_lr_params(), 'lr': learning_rate},
                {'params': net.get_10x_lr_params(), 'lr': learning_rate * 10}]


if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)

net = net.to(device)

optimizer_required = optim.SGD(train_params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
criterion_train = loss_classification_standard

print("\nTRAINING...\n")


def train(net, train_data_loader, optimizer_required, criterion_train, device, epoch):
    net.train()
    epoch_loss = 0.0
    max_iter = total_epochs * len(train_data_loader)

    for i, (image_data, label_data) in enumerate(train_data_loader):
        inputs = image_data.to(device, non_blocking=True)
        labels = label_data.to(device, non_blocking=True).long()

        outputs = net(inputs)

        optimizer_required.zero_grad()

        loss = criterion_train(outputs, labels, device)

        loss.backward()

        if math.isnan(loss) or math.isinf(loss) or loss == float('-inf'):
            optimizer_required.zero_grad()

        optimizer_required.step()

        current_iter = epoch * len(train_data_loader) + i + 1

        current_lr = poly_learning_rate(learning_rate, current_iter, max_iter, power=power)

        optimizer_required.param_groups[0]['lr'] = current_lr

        for j in range(1, len(optimizer_required.param_groups)):
            optimizer_required.param_groups[j]['lr'] = current_lr*10

        if math.isnan(loss) or math.isinf(loss) or loss == float('-inf'):
            epoch_loss += 0.0
        else:
            epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(train_data_loader)

    return average_epoch_loss


def validation(net, validation_data_loader, device, ignore_index=255):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    net.eval()

    for image_data, label_data in validation_data_loader:
        inputs = image_data
        labels = label_data.long()

        with torch.no_grad():
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = net(inputs)

        prediction = outputs

        output = prediction.max(1)[1]

        intersection, union, target = intersectionAndUnionGPU(output, labels, num_classes, device, ignore_index)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return allAcc, mAcc, mIoU, iou_class


for epochs in range(start_epoch, total_epochs):

    print('>>>> [Epoch: {0:d}] TRAINING'.format(epochs))

    train_loss = train(net, train_data_loader, optimizer_required, criterion_train, device, epochs)

    writer.add_scalar('LOSS', train_loss, epochs)

    print('>>>> [Epoch: {0:d}] AVERAGE LOSS: {1:.4f}'.format(epochs, train_loss))

    print('>>>> [Epoch: {0:d}] VALIDATION'.format(epochs))

    overall_acc, class_acc, miou, iu = validation(net, validation_data_loader, device)

    writer.add_scalar('MEAN-IOU', miou, epochs)

    print('>>>> [Epoch: {0:d}] MEAN IOU: {1:.4f}, ACCURACY: {2:.4f}, CLASS ACCURACY: {3:.4f}'.format(epochs, miou,
                                                                                                     overall_acc,
                                                                                                     class_acc))

    if miou > best_miou:
        print("\nBEST MODEL OBTAINED SO FAR. SAVING...\n")
        best_miou = miou
        checkpoint = {
            'epoch': epochs,
            'miou': miou,
            'state_dict': net.module.state_dict(),
            'optimizer': optimizer_required.state_dict()
        }

        os.chdir(save_model_checkpoints)
        for filename in os.listdir():
            os.unlink(filename)

        os.chdir(curr_dir)

        model_path = os.path.join(save_model_checkpoints, 'BEST_MODEL_EPOCH_'+str(epochs)+'.tar')
        torch.save(checkpoint, model_path)

        summary_filename = os.path.join(curr_dir, ('summary_deepLabv3_standard_pretrained_%s_separable.txt' % backbone))
        with open(summary_filename, 'w') as summary_file:
            summary_file.write("\nBEST VALIDATION\n")
            summary_file.write("EPOCH: {0}\n".format(epochs))
            summary_file.write("MEAN IOU: {0}\n".format(miou))
            summary_file.write("CLASS ACCURACY: {0}\n".format(class_acc))
            summary_file.write("MEAN ACCURACY: {0}\n".format(overall_acc))

        print('PRINTING PER CLASS IOU FOR BEST MEAN IOU:\n')

        for key, class_iou in zip(class_encodings.keys(), iu):
            print("{0}: {1:.4f}".format(key, class_iou))


checkpoint = {'state_dict': net.module.state_dict(), 'optimizer': optimizer_required.state_dict()}
model_path = os.path.join(save_model_checkpoints, 'FINAL_MODEL.tar')
torch.save(checkpoint, model_path)

