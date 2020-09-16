import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
from collections import OrderedDict
import torch.nn.init as initer
from torch import nn


def crop_image(output_tensor, required_height, required_width):
    """For cropping the up-sampled image in order to perform the concatenation operation"""
    _, _, actual_height, actual_width = output_tensor.size()
    cropped_height = (actual_height - required_height) // 2
    cropped_width = (actual_width - required_width) // 2

    required_tensor = output_tensor[:, :, cropped_height:(cropped_height+required_height), cropped_width:(cropped_width+required_width)]
    return required_tensor


def compute_mask(label_true, ignore_val=255):
    """For computing the mask which can be used to ignore the pixels while computing loss"""
    label_ignore = torch.ones_like(label_true)
    label_ignore = label_ignore * ignore_val
    ignore_label_mask = torch.ne(label_true, label_ignore).long()

    return ignore_label_mask


def heteroscedastic_loss_classification_not_vectorized(pred_tensor, true_tensor, device, logit_samples=50):
    """Loss function for classification problems"""
    total_loss = torch.zeros_like(true_tensor, dtype=torch.float32)

    ignore_label = 255
    predicted_values = pred_tensor[:, 0, :, :, :]
    predicted_variances = pred_tensor[:, 1, :, :, :]

    aleatoric_distribution = Normal(predicted_values, predicted_variances)
    ignore_label_mask = compute_mask(true_tensor)

    for _ in range(logit_samples):
        reparam_sample = aleatoric_distribution.rsample()
        loss = F.cross_entropy(reparam_sample, true_tensor, ignore_index=ignore_label, reduction='none').to(device)
        total_loss = total_loss + (loss * ignore_label_mask.float())

    average_loss_logits = total_loss / logit_samples
    overall_loss = average_loss_logits.sum() / ignore_label_mask.sum()

    return overall_loss


def loss_classification_standard(pred_tensor, true_tensor, device):
    """Standard classification cross entropy loss"""
    ignore_label = 255
    loss = F.cross_entropy(pred_tensor, true_tensor, ignore_index=ignore_label).to(device)

    return loss


def heteroscedastic_loss_classification_not_vectorized_alternative(pred_tensor, true_tensor, device, logit_samples=50):
    """Loss function for classification problems"""
    total_loss = torch.FloatTensor([0.0]).to(device)

    ignore_label = 255
    predicted_values = pred_tensor[:, 0, :, :, :]
    predicted_variances = pred_tensor[:, 1, :, :, :]

    aleatoric_distribution = Normal(predicted_values, predicted_variances)

    for _ in range(logit_samples):
        reparam_sample = aleatoric_distribution.rsample()
        loss = F.cross_entropy(reparam_sample, true_tensor, ignore_index=ignore_label).to(device)
        total_loss = total_loss + loss

    overall_loss = total_loss / logit_samples

    return overall_loss


def mc_dropout(net):
    """For setting the dropout layers to training mode during inference for MC dropout"""
    for module in net.modules():
        if type(module) == torch.nn.Dropout:
            module.train()


def colored_annotations(prediction, labels, ignore_label=255):
    """For annotating the predicted tensor according to color encodings with input being a tensor"""
    color_encoding = OrderedDict([
        (0, (128, 64, 128)),
        (1, (244, 35, 232)),
        (2, (70, 70, 70)),
        (3, (102, 102, 156)),
        (4, (190, 153, 153)),
        (5, (153, 153, 153)),
        (6, (250, 170, 30)),
        (7, (220, 220, 0)),
        (8, (107, 142, 35)),
        (9, (152, 251, 152)),
        (10, (70, 130, 180)),
        (11, (220, 20, 60)),
        (12, (255, 0, 0)),
        (13, (0, 0, 142)),
        (14, (0, 0, 70)),
        (15, (0, 60, 100)),
        (16, (0, 80, 100)),
        (17, (0, 0, 230)),
        (18, (119, 11, 32)),
        (19, (0, 0, 0))
    ])

    prediction_array = prediction.cpu().numpy()
    labels = labels.cpu()
    shape_tensor = list(prediction.size())
    assert len(shape_tensor) == 3 or len(shape_tensor) == 4

    prediction_masked = np.ma.masked_array(prediction_array, mask=torch.eq(labels, ignore_label))
    np.ma.set_fill_value(prediction_masked, 19)

    if len(shape_tensor) == 3:
        batch_size = shape_tensor[0]
        height = shape_tensor[1]
        width = shape_tensor[2]
    else:
        batch_size = shape_tensor[0]
        height = shape_tensor[2]
        width = shape_tensor[3]

    annotation_color = np.zeros((batch_size, height, width, 3))

    for key, value in color_encoding.items():
        annotation_color[prediction_masked == key] = value

    return annotation_color


def colored_labels(labels, ignore_label=255):
    """For annotating the predicted tensor according to color encodings with input being a tensor"""
    color_encoding = OrderedDict([
        (0, (128, 64, 128)),
        (1, (244, 35, 232)),
        (2, (70, 70, 70)),
        (3, (102, 102, 156)),
        (4, (190, 153, 153)),
        (5, (153, 153, 153)),
        (6, (250, 170, 30)),
        (7, (220, 220, 0)),
        (8, (107, 142, 35)),
        (9, (152, 251, 152)),
        (10, (70, 130, 180)),
        (11, (220, 20, 60)),
        (12, (255, 0, 0)),
        (13, (0, 0, 142)),
        (14, (0, 0, 70)),
        (15, (0, 60, 100)),
        (16, (0, 80, 100)),
        (17, (0, 0, 230)),
        (18, (119, 11, 32)),
        (19, (0, 0, 0))
    ])

    labels = labels.cpu()

    labels_masked = np.ma.masked_array(labels, mask=torch.eq(labels, ignore_label))
    np.ma.set_fill_value(labels_masked, 19)

    annotation_color = np.zeros((labels.size()[0], labels.size()[1], labels.size()[2], 3))

    for key, value in color_encoding.items():
        annotation_color[labels_masked == key] = value

    return annotation_color


def compute_epistemic_uncertainty(logits_probs):
    """Function for computing the epistemic uncertainty with input being the tensor"""
    log_prediction = logits_probs.log()
    individual_entropy = logits_probs*log_prediction
    epistemic_uncertain = -torch.sum(individual_entropy, dim=1)

    return epistemic_uncertain.cpu().numpy()


def compute_aleatoric_uncertainty(logits_probs, aleatoric_tensor):
    """Function for computing the aleatoric uncertainty with inputs being the tensors"""
    pred_class = torch.argmax(logits_probs, dim=1)
    sampled_aleatoric = aleatoric_tensor.gather(1, pred_class.unsqueeze(1))
    sampled_aleatoric = sampled_aleatoric.squeeze(1)
    return sampled_aleatoric.cpu().numpy()


def compute_aleatoric_uncertainty_class_average(aleatoric_tensor):
    """Function for computing the aleatoric uncertainty with inputs being the tensors"""
    sampled_aleatoric = aleatoric_tensor.cpu().numpy()
    aleatoric_average = np.mean(sampled_aleatoric, axis=1)
    return aleatoric_average


def display_batch_images(batch_images):
    """Function for displaying a batch of images"""
    num_images = batch_images.size()[0]
    fig = plt.figure(figsize=(10, 5))
    rows = 1
    columns = num_images

    for i in range(num_images):
        img = np.asarray(batch_images[i], dtype=np.float32)
        plot_img = img.transpose((1, 2, 0))
        fig.add_subplot(rows, columns, i+1)
        plt.title('Random crop number %d' % i)
        plt.imshow(plot_img)
    plt.show()


def display_uncertainties(tensor_required):
    """Function for displaying uncertainties"""
    num_images = tensor_required.shape[0]
    fig = plt.figure(figsize=(10, 5))
    rows = 1
    columns = num_images

    for i in range(num_images):
        fig.add_subplot(rows, columns, i + 1)
        plt.title('Random crop number %s' % str(i+1))
        plt.imshow((tensor_required[i]/np.max(tensor_required[i])), cmap="jet")
    plt.show()


def display_predictions(predictions, labels):
    """Function for displaying a batch of predictions"""
    batch_images = colored_annotations(predictions, labels)
    num_images = batch_images.shape[0]
    fig = plt.figure(figsize=(10, 5))
    rows = 1
    columns = num_images

    for i in range(num_images):
        fig.add_subplot(rows, columns, i+1)
        plt.title('Prediction number %s' % str(i+1))
        plt.imshow(batch_images[i]/np.max(batch_images[i]))
    plt.show()


def display_labels(labels):
    """Function for displaying a batch of labels"""
    batch_images = colored_labels(labels)
    num_images = batch_images.shape[0]
    fig = plt.figure(figsize=(10, 5))
    rows = 1
    columns = num_images

    for i in range(num_images):
        fig.add_subplot(rows, columns, i+1)
        plt.title('Random crop number %s' % str(i+1))
        plt.imshow(batch_images[i]/np.max(batch_images[i]))
    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    """For computing mIou on the CPU"""
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, device, ignore_index=255):
    """For computing mIou on the GPU"""
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.to(device), area_union.to(device), area_target.to(device)


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """For initializing the weights of the network"""
    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)















