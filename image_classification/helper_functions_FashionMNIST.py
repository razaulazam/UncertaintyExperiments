import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.distributions import Normal
from torch.distributions.independent import Independent
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet


def display_batch_images(batch_images, num_display):
    assert num_display <= batch_images.size()[0], 'Number of images to display should be equal to or less than the batch size'
    figure_images = plt.figure(figsize=(10, 5))
    columns = 5
    rows = math.ceil(num_display / columns)



    for i in range(num_display):

        img = batch_images[i]

        img = np.asarray(img, dtype=np.float32)

        img = img.transpose((1, 2, 0))

        img = np.squeeze(img, axis=2)



        figure_images.add_subplot(rows, columns, i + 1)

        plt.imshow(img, cmap='gray')

        plt.axis('off')

        plt.title('Image:{0}'.format(i + 1))



    plt.show()







def calculate_accuracy(pred, label_true):

    label_pred = pred.argmax(dim=1, keepdim=True)

    correct = label_pred.eq(label_true.view_as(label_pred)).sum()



    acc = correct.float()/label_true.size()[0]



    return acc







def calculate_accuracy_back_labels(pred, label_true):

    non_ignore_labels = label_true != 10

    label_true = label_true[non_ignore_labels]

    pred = pred[non_ignore_labels, :]

    

    if len(pred) > 0:

        label_pred = pred.argmax(dim=1, keepdim=True)

        correct = label_pred.eq(label_true.view_as(label_pred)).sum()

        acc = correct.float()/label_true.size()[0]

    else:

        acc = torch.FloatTensor([0.0])



    return acc







def calculate_top_k_error(output, target, topk=(1,)):

    maxk = max(topk)

    batch_size = output.size()[0]

    _, pred_indices = torch.topk(output, maxk, 1, True, True)

    pred_indices = pred_indices.t()

    correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))

    

    res = []

    

    for k in topk:

        val_correct = correct[:k].view(-1).float().sum()

        res.append(val_correct.mul_(100/batch_size))

    

    return res









def calculate_top_k_error_test_back_class(output, target, topk=(1,)):

    non_ignore_labels = target != 10

    target = target[non_ignore_labels]

    output = output[non_ignore_labels, :]

    maxk = max(topk)

    batch_size = output.size()[0]

    _, pred_indices = torch.topk(output, maxk, 1, True, True)

    pred_indices = pred_indices.t()

    correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))

    

    res = []

    

    for k in topk:

        if batch_size > 0:

            val_correct = correct[:k].view(-1).float().sum()

            res.append(val_correct.mul_(100/batch_size))

        else:

            continue

    

    return res





def CrossEntropySoftLabels(logits, label):

    prob = F.softmax(logits, dim=1)

    loss = -((prob+1e-10).log() * label).sum(dim=1)

    

    return loss





def heteroscedastic_loss_classification_one_hot_labels_MN(logits_samples=500):

    def loss_function(logits_variances_pred, label_true, device):

        loss = torch.zeros(label_true.size()[0]).float().to(device)

        logits = logits_variances_pred[:, 0, :]

        variances = logits_variances_pred[:, 1, :]



        aleatoric_distribution = Independent(Normal(logits, variances), 1)



        for i in range(logits_samples):

            reparam_sample = aleatoric_distribution.rsample()

            loss += CrossEntropySoftLabels(reparam_sample, label_true).to(device)



        average_loss = loss / logits_samples

        overall_loss = average_loss.sum()/logits.size()[0]



        return overall_loss

    return loss_function









def heteroscedastic_loss_classification_dirichlet(logits_samples=500):

    def loss_function(logits_variances_pred, c1, c2, label_true, device):

        loss = torch.zeros_like(label_true).float()

        logits = logits_variances_pred[:, 0, :]

        variances = logits_variances_pred[:, 1, :]
        
        mu = F.softmax(logits, dim=1)
        stddev = torch.sqrt(torch.sum(mu * variances, dim=1, keepdim=True))
        
        s = 1.0 / (c1 + (c2 * stddev) + 1e-5)
        alpha = mu * s

        aleatoric_distribution = Dirichlet(alpha)



        for i in range(logits_samples):

            reparam_sample = aleatoric_distribution.rsample()

            loss += F.nll_loss(reparam_sample, label_true, reduction='none').to(device)


        average_loss = loss / logits_samples

        overall_loss = average_loss.sum()/logits.size()[0]


        return overall_loss

    return loss_function




def heteroscedastic_loss_classification_MN(logits_samples=500):

    def loss_function(logits_variances_pred, label_true, device):

        loss = torch.zeros_like(label_true).float()

        logits = logits_variances_pred[:, 0, :]

        variances = logits_variances_pred[:, 1, :]

        logits_vec = torch.zeros_like(logits).float()

        aleatoric_distribution = Independent(Normal(logits, variances), 1)



        for i in range(logits_samples):

            reparam_sample = aleatoric_distribution.rsample()
        
            loss += F.cross_entropy(reparam_sample, label_true, reduction='none').to(device)
            
            logits_vec += reparam_sample.detach()




        average_loss = loss / logits_samples
        
        logits_vec /= logits_samples
        
        prob_vec_ent = compute_prob_vec_entropy(logits_vec)
        
        overall_loss = (average_loss.sum()/logits.size()[0]) + (prob_vec_ent.sum()/logits.size()[0])

        return overall_loss
    

    return loss_function



def heteroscedastic_loss_classification_MN_KL(logits_samples=500):

    def loss_function(logits_variances_pred, label_true, device):

        loss = torch.zeros_like(label_true).float()

        logits = logits_variances_pred[:, 0, :]

        variances = logits_variances_pred[:, 1, :]

        prob_vec = torch.zeros_like(logits).float()
        
        uniform_vec = torch.ones_like(logits)*float(0.1)

        aleatoric_distribution = Independent(Normal(logits, variances), 1)



        for i in range(logits_samples):

            reparam_sample = aleatoric_distribution.rsample()
            
            prob_vec += F.softmax(reparam_sample, dim=1)

            loss += F.cross_entropy(reparam_sample, label_true, reduction='none').to(device)



        average_loss = loss / logits_samples
        
        prob_vec /= logits_samples
        
        kl_dist = torch.log(prob_vec+1e-10) * uniform_vec
        
        kl_dist_total = -torch.sum(kl_dist, dim=1)
        
        overall_loss = (average_loss.sum()/logits.size()[0]) + (kl_dist_total.sum()/logits.size()[0])

        return overall_loss
    

    return loss_function




def compute_prob_vec_entropy(logits_vec):
    
    prob_vec = F.softmax(logits_vec, dim=1)
    
    prob_vec += 1e-10
    
    log_prediction = prob_vec.log()

    individual_entropy = prob_vec*log_prediction

    epistemic_uncertain = -torch.sum(individual_entropy, dim=1)

    return epistemic_uncertain




def compute_epistemic_uncertainty(logits_probs):

    """Function for computing the epistemic uncertainty with input being the tensor"""

    log_prediction = logits_probs.log()

    individual_entropy = logits_probs*log_prediction

    epistemic_uncertain = -torch.sum(individual_entropy, dim=1)



    return epistemic_uncertain.cpu().numpy()









def compute_entropy_mutual(logits_probs, num_samples):

    """Function for computing the epistemic uncertainty with input being the tensor epoch wise"""

    log_prediction = logits_probs.log()

    individual_entropy = logits_probs*log_prediction

    epistemic_uncertain = (torch.sum(individual_entropy, dim=1))/num_samples

    

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









def mc_dropout(net):

    """For setting the dropout layers to training mode during inference for MC dropout"""

    for module in net.modules():

        if type(module) == nn.Dropout:

            module.train()

  

            

                       

def adjust_learning_rate(optimizer, initial_lr, epoch):

    print(' >>> LR SETTING <<<')

    new_lr = initial_lr * (0.1 ** (epoch//20))

    for group in optimizer.param_groups:

        group['lr'] = new_lr 

    

            

def calculate_accuracy_one_hot(pred, label_true):

    label_true = label_true.argmax(dim=1)

    label_pred = pred.argmax(dim=1, keepdim=True)

    correct = label_pred.eq(label_true.view_as(label_pred)).sum()



    acc = correct.float()/label_true.size()[0]



    return acc





def calculate_top_k_error_one_hot(output, target, topk=(1,)):

    target = target.argmax(dim=1)

    maxk = max(topk)

    batch_size = output.size()[0]

    _, pred_indices = torch.topk(output, maxk, 1, True, True)

    pred_indices = pred_indices.t()

    correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))

    

    res = []

    

    for k in topk:

        val_correct = correct[:k].view(-1).float().sum()

        res.append(val_correct.mul_(100/batch_size))

    

    return res








