import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from deepLabv3_xception_standard_drop import DeepLab
from helper_functions_deepLabv3 import AverageMeter, mc_dropout
from helper_functions_deepLabv3 import intersectionAndUnionGPU, colored_labels, compute_epistemic_uncertainty
from helper_functions_deepLabv3 import colored_annotations
from test_data_loader_deepLabv3 import test_loader
from helper_functions_deepLabv3 import get_equal_interval_bins

use_gpu = torch.cuda.is_available()
device = ('cuda:0' if use_gpu else 'cpu')

epoch_load = 223
batch_size = 4
mc_samples = 0
drop_rate = 0.5
forward_pass = 'standard_drop_xception_%.1f' % drop_rate
possible_evaluation = {1: 'NORMAL', 2: 'MC'}
chosen_evaluation = 2
name_evaluation = possible_evaluation[chosen_evaluation]

num_bins = 10
bins = get_equal_interval_bins(num_bins=num_bins)
bin_sums_pred = np.zeros((num_bins, 1))
bin_sums_prob = np.zeros((num_bins, 1))
counts = np.zeros((num_bins, 1))

if chosen_evaluation == 1:
    mc_samples = 1
elif chosen_evaluation == 2:
    mc_samples = 25

mc_samples_float = float(mc_samples)

curr_dir = os.getcwd()
result_dir = 'deepLabv3_%s_evaluation_%s' % (forward_pass, name_evaluation)
tensor_dir = 'deepLabv3_%s_evaluation_store_tensor_%s' % (forward_pass, name_evaluation)

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
restore_from = 'deepLabv3_training_standard_drop_%.1f_xception' % drop_rate
model_epoch = 'BEST_MODEL_EPOCH_%d.tar' % epoch_load
overall_dir = os.path.join(restore_from, model_epoch)
load_dir = os.path.join(curr_dir, overall_dir)

num_classes = 19
ignore_index = 255
max_entropy = np.log(num_classes)

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

        for i in range(mc_samples):
            output = net(image)

            logits = output

            p_value = F.softmax(logits, dim=1)
            p = p + (p_value / mc_samples_float)

        output_mIou = p.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_mIou, label, num_classes, device, ignore_index)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        epistemic_uncertainty = compute_epistemic_uncertainty(p)

        pred_label_colored = colored_annotations(output_mIou, label, ignore_label=ignore_index)
        label_colored = colored_labels(label, ignore_label=ignore_index)
        
        gts = label.detach().cpu().numpy()
        gts = gts.ravel()

        preds = p.cpu().numpy()
        confs = preds.max(axis=1).ravel()
        pred_classes = preds.argmax(axis=1).ravel()
        del preds
        
        correct = pred_classes == gts # without ignoring the ignore pixels
        assigned = np.digitize(confs, bins)
        
        bin_sums_prob_temp = np.bincount(assigned, weights=confs, minlength=len(bins))
        bin_sums_prob_temp = np.expand_dims(bin_sums_prob_temp, axis=1)
        bin_sums_prob += bin_sums_prob_temp

        bin_sums_pred_temp = np.bincount(assigned, weights=correct, minlength=len(bins))
        bin_sums_pred_temp = np.expand_dims(bin_sums_pred_temp, axis=1)
        bin_sums_pred += bin_sums_pred_temp

        counts_temp = np.bincount(assigned, minlength=len(bins))
        counts_temp = np.expand_dims(counts_temp, axis=1)
        counts += counts_temp
        
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

filt = counts > 0
prob_pred = (bin_sums_pred[filt] / counts[filt])
average_prob = (bin_sums_prob[filt] / counts[filt])
count_store = counts[filt]

np.save(os.path.join(save_tensor, 'prob_pred'), prob_pred)
np.save(os.path.join(save_tensor, 'average_prob'), average_prob)
np.save(os.path.join(save_tensor, 'counts'), count_store)
