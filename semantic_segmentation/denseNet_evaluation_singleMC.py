import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from denseNet import DenseNet
from helper_functions_denseNet import mc_dropout, AverageMeter, compute_aleatoric_uncertainty
from helper_functions_denseNet import compute_aleatoric_uncertainty_class_average
from helper_functions_denseNet import compute_epistemic_uncertainty, intersectionAndUnionGPU, colored_labels
from helper_functions_denseNet import colored_annotations
from test_data_loader_denseNet import test_loader

use_gpu = torch.cuda.is_available()
device = ('cuda:0' if use_gpu else 'cpu')

epoch_load = 1
mc_samples = 1
mc_samples_float = float(mc_samples)
batch_size = 4
forward_pass = 'mc_single'
aleatoric_method = 'averageClassMC'
conf_interval_num = 10
conf_interval_size = 1.0/conf_interval_num
total_val_interval = [0]*conf_interval_num
actual_val_interval = [0]*conf_interval_num
conf_val_interval_all = [np.array([])]*conf_interval_num

curr_dir = os.getcwd()
result_dir = 'denseNet_%s_%d_bs_%d_evaluation_%s' % (forward_pass, mc_samples, batch_size, aleatoric_method)
tensor_dir = 'denseNet_%s_%d_bs_%d_evaluation_store_tensor_%s' % (forward_pass, mc_samples, batch_size, aleatoric_method)
save_tensor = os.path.join(curr_dir, tensor_dir)
save_result = os.path.join(curr_dir, result_dir)

if os.path.isdir(save_result):
    print('>>>> REMOVING THE CURRENT DIRECTORY AND MAKING A NEW ONE <<<<')
    shutil.rmtree(save_result)
    os.mkdir(save_result)
    shutil.rmtree(save_tensor)
    os.mkdir(save_tensor)
else:
    print('>>>> MAKING A NEW DIRECTORY FOR EVALUATION RESULTS')
    os.mkdir(save_result)
    os.mkdir(save_tensor)

data_dir = '/fs/scratch/rng_cr_bcai_dl_students/OpenData/cityscapes/'
restore_from = 'denseNet_fine_tuning'
model_epoch = 'BEST_MODEL_EPOCH_%d.tar' % epoch_load
overall_dir = os.path.join(restore_from, model_epoch)
load_dir = os.path.join(curr_dir, overall_dir)

num_classes = 19
ignore_index = 255
max_entropy = np.log(num_classes)

test_data_loader = test_loader(data_dir=data_dir, batch_size=batch_size)
checkpoint = torch.load(load_dir)

net = DenseNet()
net.load_state_dict(checkpoint['state_dict'])

if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)

net = net.to(device)
net.eval()
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
            variance = output[:, 1, :, :, :]

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

        label_non_ignore_mask = (label != 255).detach()
        label = label[label_non_ignore_mask]

        p = torch.transpose(p, 1, 2)
        p = torch.transpose(p, 2, 3)

        p = p[label_non_ignore_mask, :]

        label = label.cpu().numpy()
        p_numpy = p.cpu().numpy()

        pred = np.argmax(p_numpy, axis=1).astype(np.uint32)
        conf = np.max(p_numpy, axis=1)

        for i in range(conf_interval_num):
            lower_bound = i * conf_interval_size
            upper_bound = (i + 1) * conf_interval_size

            conf_in_interval = conf[np.nonzero(np.logical_and(conf >= lower_bound, conf < upper_bound))]
            val_in_interval = conf_in_interval.shape[0]

            num_correct_pred_in_interval = np.count_nonzero(np.logical_and(np.logical_and(conf >= lower_bound, conf < upper_bound), pred == label))
            actual_val_interval[i] += num_correct_pred_in_interval
            total_val_interval[i] += val_in_interval
            conf_val_interval_all[i] = np.concatenate((conf_val_interval_all[i], conf_in_interval))

        if step % 8 == 0:
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
                    plt.subplot(3, 2, 1), plt.imshow(img / np.max(images))
                    plt.title('Image')
                    plt.axis('off')
                    plt.subplot(3, 2, 2), plt.imshow(label_over_lay / np.max(label_colored))
                    plt.title('Ground Truth')
                    plt.axis('off')
                    plt.subplot(3, 2, 3), plt.imshow(pred_over_lay / np.max(pred_label_colored))
                    plt.title('Prediction')
                    plt.axis('off')
                    plt.subplot(3, 2, 4), plt.imshow(entropy_img_epistemic / 255, cmap='hot')
                    plt.title('Epistemic')
                    plt.axis('off')
                    plt.subplot(3, 2, 5), plt.imshow(entropy_img_aleatoric_average / 255, cmap='hot')
                    plt.title('Aleatoric (average)')
                    plt.axis('off')
                    plt.subplot(3, 2, 6), plt.imshow(entropy_img_aleatoric_class_average / 255, cmap='hot')
                    plt.title('Aleatoric (class average)')
                    plt.axis('off')

                    os.chdir(save_result)
                    plt.savefig('result_step_%d.png' % (step/8))
                    plt.close()
                    os.chdir(curr_dir)

iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
mIoU = np.mean(iou_class)
mAcc = np.mean(accuracy_class)
allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

with open('summary_denseNet_evaluation_%s_%d_bs_%d_%s' % (forward_pass, mc_samples, batch_size,
                                                          aleatoric_method), 'w') as summary:
    summary.write('\n OVERALL STATISTICS OF THE TEST DATASET\n')
    summary.write('MIOU: {0}\n'.format(mIoU))
    summary.write('CLASS ACCURACY: {0}\n'.format(mAcc))
    summary.write('MEAN ACCURACY: {0}\n'.format(allAcc))

total_val_interval = np.add(total_val_interval, 1e-10)
calibration_freq = np.divide(actual_val_interval, total_val_interval)
interval_mean_conf = np.zeros(conf_interval_num)

for i in range(conf_interval_num):
    mean_conf = np.mean(conf_val_interval_all[i])
    interval_mean_conf[i] = mean_conf

np.save(os.path.join(save_tensor, 'calibrated_freq'), calibration_freq)
np.save(os.path.join(save_tensor, 'bins_prob'), interval_mean_conf)


