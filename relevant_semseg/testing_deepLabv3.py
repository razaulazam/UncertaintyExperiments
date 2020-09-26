import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from deepLabv3 import DeepLab
from helper_functions_deepLabv3 import mc_dropout, AverageMeter, compute_aleatoric_uncertainty
from helper_functions_deepLabv3 import compute_aleatoric_uncertainty_class_average
from helper_functions_deepLabv3 import compute_epistemic_uncertainty, intersectionAndUnionGPU, colored_labels
from helper_functions_deepLabv3 import colored_annotations
from test_data_loader_deepLabv3 import test_loader

use_gpu = torch.cuda.is_available()
device = ('cuda:0' if use_gpu else 'cpu')

epoch_load = 227 # Needs to be changed according to the epoch
batch_size = 4
mc_samples = 0
drop_rate = 0.2
patch_size = 40 # Needs to be changed
best_mIou_batch = 0.0

forward_pass = 'mixed_patch_%d' % patch_size
possible_evaluation = {1: 'NORMAL', 2: 'MC'}
chosen_evaluation = 2
name_evaluation = possible_evaluation[chosen_evaluation]

curr_dir = os.getcwd()
result_dir = 'deepLabv3_%s_evaluation_%s' % (forward_pass, name_evaluation)
save_result = os.path.join(curr_dir, result_dir)

if chosen_evaluation == 1:
    mc_samples = 1
elif chosen_evaluation == 2:
    mc_samples = 25

mc_samples_float = float(mc_samples)

if os.path.isdir(save_result):
    print('>>>> REMOVING THE CURRENT DIRECTORY AND MAKING A NEW ONE <<<<')
    shutil.rmtree(save_result)
    os.mkdir(save_result)
else:
    print('>>>> MAKING A NEW DIRECTORY FOR EVALUATION RESULTS')
    os.mkdir(save_result)

data_dir = '/fs/scratch/rng_cr_bcai_dl_students/OpenData/cityscapes/'
restore_from = 'deepLabv3_training_alea_epis_%.1f_resnet_mixed_10_12' % drop_rate
model_epoch = 'BEST_MODEL_EPOCH_%d.tar' % epoch_load
overall_dir = os.path.join(restore_from, model_epoch)
load_dir = os.path.join(curr_dir, overall_dir)

num_classes = 19
ignore_index = 255

test_data_loader = test_loader(data_dir=data_dir, batch_size=batch_size)
checkpoint = torch.load(load_dir)

net = DeepLab(layer_drop=drop_rate)
net.load_state_dict(checkpoint['state_dict'])

if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)

net = net.to(device)
net.eval()

if chosen_evaluation == 2:
    print('APPLYING DROPOUT')
    net.apply(mc_dropout)

intersection_meter = AverageMeter()
union_meter = AverageMeter()
target_meter = AverageMeter()

for step, (image_data, label_data) in enumerate(test_data_loader):
    with torch.no_grad():
        print("%d/%d" % (step+1, len(test_data_loader)))

        image = image_data.to(device, non_blocking=True)
        label = label_data.to(device, non_blocking=True).long()

        batch_size = image.size()[0]
        h = image.size()[2]
        w = image.size()[3]

        p = torch.zeros(batch_size, num_classes, h, w).to(device)
        aleatoric_variance = torch.zeros(batch_size, num_classes, h, w).to(device)

        for i in range(mc_samples):
            output = net(image)

            logits = output[:, 0, :, :, :]
            variance = output[:, 1, :, :, :]**2

            p_value = F.softmax(logits, dim=1)

            p = p + (p_value/mc_samples_float)
            aleatoric_variance = aleatoric_variance + (variance/mc_samples_float)

        output_mIou = p.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_mIou, label, num_classes, device, ignore_index)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        aleatoric_uncertainty_average = compute_aleatoric_uncertainty(p, aleatoric_variance)
        aleatoric_uncertainty_class = compute_aleatoric_uncertainty_class_average(aleatoric_variance)
        epistemic_uncertainty = compute_epistemic_uncertainty(p)

        pred_label_colored = colored_annotations(output_mIou, label, ignore_label=ignore_index)
        label_colored = colored_labels(label, ignore_label=ignore_index)    

        if step % 16 == 0:
            for i in range(image.size()[0]):
                if i == 0:
                    images = image.cpu().numpy()
                    images = np.transpose(images, (0, 2, 3, 1))
                    img = images[i]
                    img = img + np.array([103.93, 116.77, 123.68])

                    label_img_color = label_colored[i]
                    label_over_lay = 0.20 * img + 0.80 * label_img_color

                    pred_label_img = pred_label_colored[i]
                    pred_over_lay = 0.20 * img + 0.80 * pred_label_img

                    entropy_epistemic = epistemic_uncertainty[i]
                    entropy_img = (entropy_epistemic / np.max(epistemic_uncertainty)) * 255
                    entropy_img_epistemic = entropy_img.astype(np.uint8)

                    entropy_aleatoric_average = aleatoric_uncertainty_average[i]
                    entropy_img = (entropy_aleatoric_average / np.max(aleatoric_uncertainty_average)) * 255
                    entropy_img_aleatoric_average = entropy_img.astype(np.uint8)
                    
                    entropy_aleatoric_class_average = aleatoric_uncertainty_class[i]
                    entropy_img = (entropy_aleatoric_class_average / np.max(aleatoric_uncertainty_class)) * 255
                    entropy_img_aleatoric_class_average = entropy_img.astype(np.uint8)

                    plt.figure(figsize=(20, 20))
                    plt.subplot(2, 3, 1), plt.imshow(img / np.max(images))
                    plt.title('Image')
                    plt.axis('off')
                    plt.subplot(2, 3, 2), plt.imshow(label_over_lay / np.max(label_colored))
                    plt.title('Ground Truth')
                    plt.axis('off')
                    plt.subplot(2, 3, 3), plt.imshow(pred_over_lay / np.max(pred_label_colored))
                    plt.title('Prediction')
                    plt.axis('off')
                    plt.subplot(2, 3, 4), plt.imshow(entropy_img_epistemic / 255, cmap='hot')
                    plt.title('Epistemic')
                    plt.axis('off')
                    plt.subplot(2, 3, 5), plt.imshow(entropy_img_aleatoric_average / 255, cmap='hot')
                    plt.title('Aleatoric (Maximum confidence)')
                    plt.axis('off')
                    plt.subplot(2, 3, 6), plt.imshow(entropy_img_aleatoric_class_average / 255, cmap='hot')
                    plt.title('Aleatoric (Average confidence)')
                    plt.axis('off')
      
                    os.chdir(save_result)
                    plt.savefig('result_step_%d.png' % (step/16))
                    plt.close()
                    os.chdir(curr_dir)

iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
mIoU = np.mean(iou_class)
mAcc = np.mean(accuracy_class)
allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

with open('mIou_deepLabv3_%s_evaluation_%s.txt' % (forward_pass, name_evaluation), 'w') as summary:
    summary.write('\n OVERALL STATISTICS OF THE TEST DATASET\n')
    summary.write('MIOU: {0}\n'.format(mIoU))
    summary.write('CLASS ACCURACY: {0}\n'.format(mAcc))
    summary.write('MEAN ACCURACY: {0}\n'.format(allAcc))





