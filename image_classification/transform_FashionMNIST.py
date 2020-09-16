import cv2

import numpy as np

import numbers

import collections

import os

import random

from torch.utils.data import Dataset

from torchvision.datasets import MNIST





# Transforms on the Training data

class TrainTransform:

    def __init__(self, mean, std):

        self.mean = mean

        self.std = std



    def __call__(self, image):

        image = np.asarray(image, dtype=np.float32)

        image = rand_rotate(image)

        image -= self.mean

        image /= self.std

        image = np.expand_dims(image, axis=0)



        return image







# Transform on the Test data

class TestTransform:

    def __init__(self, mean, std):

        self.mean = mean

        self.std = std



    def __call__(self, image):

        image = np.asarray(image, dtype=np.float32)

        image -= self.mean

        image /= self.std

        image = np.expand_dims(image, axis=0)



        return image







# Transform which randomly rotates the image

class RandomRotate:

    def __init__(self, rotate, p=0.5):

        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)

        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:

            self.rotate = rotate

        else:

            raise(RuntimeError('Enter a valid combination for the rotation factor'))



        self.padding = [0, 0, 0]

        self.p = p



    def __call__(self, image):

        if random.random() < self.p:

            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()

            h, w = image.shape



            matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)

            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,

                                   borderValue=self.padding)

        return image





rand_rotate = RandomRotate(rotate=[-10, 10])







# For generating black images test set

class ProduceBlackImagesDataSet:

    def __init__(self, num_black_images, transform=None):

        self.data = []

        self.transform = transform  

        black_images = [np.zeros((28, 28), dtype=np.float32)]*num_black_images

        self.data = black_images

                

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        img = self.data[idx]

        if self.transform is not None:

            img = self.transform(img)

        return img
    
    def returnDataList(self):
        return self.data







# For generating train data-set with black (or white) images (two different classes assigned to half half images)

class ProduceTrainDataBlack:

    def __init__(self, subset, num_images, class_one, class_two, transform=None):

        self.subset = subset

        self.transform = transform

        self.data = []

        self.label = []

        

        index_all = len(subset)

        

        for index in range(index_all):

            self.data.append(self.subset.__getitem__(int(index))[0])

            self.label.append(self.subset.__getitem__(int(index))[1])

    

        

        sample = np.zeros((28, 28), dtype=np.float32)



        sample_images = [sample]*num_images

        class_one_num = num_images // 2

        label_sample_images = [class_one]*class_one_num + [class_two]*(num_images - class_one_num)

        

        self.data = self.data + sample_images

        self.label = self.label + label_sample_images

                

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        img, target = self.data[idx], int(self.label[idx])

        if self.transform is not None:

            img = self.transform(img)

        return img, target

    

    

    

# For loading MNIST data

def loadMNIST(self, root=None, transform=None, train=True):

    

    assert root is not None, 'Please specify the root directory!'

    self.root = root

    self.transform = transform

    self.train = train



    data_set = MNIST(root=data_dir, train=self.train, transform=transform)



    return data_set

    





# For extracting only one class from the training data-set of MNIST

class extractOOD:

    def __init__(self, dataset):

        self.dataset = dataset

        total_samples = self.dataset.__len__()

        self.data = []

        self.labels = []

        

        for i in range(total_samples):

            if self.dataset.__getitem__(i)[1] == int(2):

                self.data.append(self.dataset.__getitem__(i)[0])

                self.labels.append(self.dataset.__getitem__(i)[1])

            else:

                continue

        

    def __len__(self):

        return len(self.data)

        

    def __getitem__(self, idx):

        data, label = self.data[idx], self.labels[idx]

        return data, label

    

    





# For loading Fashion MNIST data-set

class FashionMNIST:

    def __init__(self, root=None, train=True, transform=None):

        assert root is not None, 'Please specify the root directory!'

        self.root = root

        self.transform = transform

        self.train = train

        self.data = []

        self.labels = []

        

        train_test_data = np.load(os.path.join(self.root, 'fashion_mnist.npz'))

        

        if self.train:

            data = train_test_data['x_train']

            labels = train_test_data['y_train']

        else:

            data = train_test_data['x_test']

            labels = train_test_data['y_test']

        

        for i in range(len(labels)):

            self.data.append(data[i])

            self.labels.append(labels[i])

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        data, label = self.data[idx], self.labels[idx]

        if self.transform is not None:

            data = self.transform(data)

            

        return data, label
    
    
    def returnDataList(self):
        
        return self.data, self.labels

    

    



# For computing the statistics of the data

def compute_mean_std():

    root = '/fs/scratch/rng_cr_bcai_dl_students/OpenData/fashion_mnist/'

    train_test_data = np.load(os.path.join(root, 'fashion_mnist.npz'))

    train_data = train_test_data['x_train']

    mean = train_data.mean()

    std = train_data.std()

    return mean, std

    



    

# Mixes up the Images horizontally

def transformMixUpImagesHorizontal(image_one_class, image_two_class):

    mixed_images = []

    for image_one, image_two in zip(image_one_class, image_two_class):               



        image_one = np.array(image_one, dtype=np.float32)

        image_two = np.array(image_two, dtype=np.float32)



        crop_one_image = image_one[:, :14]

        crop_two_image = image_two[:, :14]



        mixed_images.append(np.concatenate([crop_one_image, crop_two_image], axis=1))

    

    return mixed_images 







# Mixed up the Images vertically

def transformMixUpImagesVertical(image_one_class, image_two_class):

    mixed_images = []

    for image_one, image_two in zip(image_one_class, image_two_class):               



        image_one = np.array(image_one, dtype=np.float32)

        image_two = np.array(image_two, dtype=np.float32)



        crop_one_image = image_one[:14, :]

        crop_two_image = image_two[:14, :]



        mixed_images.append(np.concatenate([crop_one_image, crop_two_image], axis=0))

    

    return mixed_images 

 

       



# For assigning two hard labels to the mixed up class

class ProduceMixUpAleatoricConfusedHard: 

    def __init__(self, subset, class_one, class_two, chosen_mix, transform=None):

        print('Applying MixUp - Combining parts of two Images')

        self.subset = subset

        self.transform = transform

        self.class_one = class_one

        self.class_two = class_two

        self.data_class_one = []

        self.data_class_two = []

        self.data = []

        self.label = []

        self.chosen_mix = chosen_mix

        

        index_all = len(self.subset)

        

        for index in range(index_all):

            if self.subset.__getitem__(index)[1] == self.class_one:

                self.data_class_one.append(self.subset.__getitem__(index)[0])

            elif self.subset.__getitem__(index)[1] == self.class_two:

                self.data_class_two.append(self.subset.__getitem__(index)[0])

            else:

                self.data.append(self.subset.__getitem__(index)[0])

                self.label.append(self.subset.__getitem__(index)[1])

                

        

        use_len = min(len(self.data_class_one), len(self.data_class_two))

        

        self.data_class_one_transform = self.data_class_one[:(use_len//2)]

        self.data_class_one_training = self.data_class_one[(use_len//2):]

        self.label_class_one_training = [self.class_one]*(len(self.data_class_one_training))

        self.data_class_two_transform = self.data_class_two[:(use_len//2)]

        self.data_class_two_training = self.data_class_two[(use_len//2):]

        self.label_class_two_training = [self.class_two]*(len(self.data_class_two_training))



        self.data = self.data + self.data_class_one_training + self.data_class_two_training

        self.label = self.label + self.label_class_one_training + self.label_class_two_training

        

        if self.chosen_mix == 1:

            self.mixed_images = transformMixUpImagesHorizontal(self.data_class_one_transform, self.data_class_two_transform)

        elif self.chosen_mix == 2:

            self.mixed_images = transformMixUpImagesVertical(self.data_class_one_transform, self.data_class_two_transform)

        else:

            raise RuntimeError('Please enter a valid number')

        

        mixed_one_labels = len(self.mixed_images) // 2

        mixed_two_labels = len(self.mixed_images) - mixed_one_labels

           

        self.label_mixed_images = [self.class_one]*mixed_one_labels + [self.class_two]*mixed_two_labels

        

        self.data = self.data + self.mixed_images

        self.label = self.label + self.label_mixed_images

        

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        data, label = self.data[idx], self.label[idx]

        if self.transform is not None:

            data = self.transform(data)

        return data, label

    


# Function for combining the required Images   
def transformCombineImages(image_one_class, image_two_class):
    mixed_images = []
    
    for image_one, image_two in zip(image_one_class, image_two_class):               
        image_one = np.array(image_one, dtype=np.float32)
        image_two = np.array(image_two, dtype=np.float32)

        mixed_images.append(0.5*image_one + 0.5*image_two)
           
    return mixed_images 


# For assigning two hard labels to the combined images
class ProduceCombineAleatoricConfusedHard: 
    def __init__(self, subset, class_one, class_two, transform=None):
        self.subset = subset
        self.transform = transform
        self.class_one = class_one
        self.class_two = class_two
        self.data_class_one = []
        self.data_class_two = []
        self.data = []
        self.label = []
        
        index_all = len(self.subset)
        
        for index in range(index_all):
            if self.subset.__getitem__(index)[1] == self.class_one:
                self.data_class_one.append(self.subset.__getitem__(index)[0])
            elif self.subset.__getitem__(index)[1] == self.class_two:
                self.data_class_two.append(self.subset.__getitem__(index)[0])
            else:
                self.data.append(self.subset.__getitem__(index)[0])
                self.label.append(self.subset.__getitem__(index)[1])
                
        
        use_len = min(len(self.data_class_one), len(self.data_class_two))
        
        self.data_class_one_transform = self.data_class_one[:(use_len//2)]
        self.data_class_one_training = self.data_class_one[(use_len//2):]
        self.label_class_one_training = [self.class_one]*(len(self.data_class_one_training))
        self.data_class_two_transform = self.data_class_two[:(use_len//2)]
        self.data_class_two_training = self.data_class_two[(use_len//2):]
        self.label_class_two_training = [self.class_two]*(len(self.data_class_two_training))

        self.data = self.data + self.data_class_one_training + self.data_class_two_training
        self.label = self.label + self.label_class_one_training + self.label_class_two_training
        
        self.mixed_images = transformCombineImages(self.data_class_one_transform, self.data_class_two_transform)
        
        mixed_one_labels = len(self.mixed_images) // 2
        mixed_two_labels = len(self.mixed_images) - mixed_one_labels
           
        self.label_mixed_images = [self.class_one]*mixed_one_labels + [self.class_two]*mixed_two_labels
        
        self.data = self.data + self.mixed_images
        self.label = self.label + self.label_mixed_images
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, label = self.data[idx], self.label[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
        


    

# For assigning soft labels to the mixed up class

class ProduceMixUpAleatoricSoftLabels:

    def __init__(self, subset, class_one, class_two, soft_classes, chosen_mix, transform=None):

        print('Applying MixUp - Combining parts of two Images')

        self.subset = subset

        self.transform = transform

        self.class_one = class_one

        self.class_two = class_two

        self.chosen_mix = chosen_mix

        self.soft_classes = soft_classes

        self.data_class_one = []

        self.data_class_two = []

        self.data = []

        self.label = None

        self.label_mixed_images = None

        

        label = []

        num_classes = 10

        

        index_all = len(self.subset)

        

        for index in range(index_all):

            if self.subset.__getitem__(index)[1] == self.class_one:

                self.data_class_one.append(self.subset.__getitem__(index)[0])

            elif self.subset.__getitem__(index)[1] == self.class_two:

                self.data_class_two.append(self.subset.__getitem__(index)[0])

            else:

                self.data.append(self.subset.__getitem__(index)[0])

                label.append(self.subset.__getitem__(index)[1])

                

        

        use_len = min(len(self.data_class_one), len(self.data_class_two))

        

        self.data_class_one_transform = self.data_class_one[:(use_len//2)]

        self.data_class_one_training = self.data_class_one[(use_len//2):]

        self.label_class_one_training = [self.class_one]*(len(self.data_class_one_training))

        self.data_class_two_transform = self.data_class_two[:(use_len//2)]

        self.data_class_two_training = self.data_class_two[(use_len//2):]

        self.label_class_two_training = [self.class_two]*(len(self.data_class_two_training))



        self.data = self.data + self.data_class_one_training + self.data_class_two_training

        label = label + self.label_class_one_training + self.label_class_two_training

        

        label = np.array(label)

        self.label = np.eye(num_classes)[label]

        

        if self.chosen_mix == 1:

            self.mixed_images = transformMixUpImagesHorizontal(self.data_class_one_transform, self.data_class_two_transform)

        elif self.chosen_mix == 2:

            self.mixed_images = transformMixUpImagesVertical(self.data_class_one_transform, self.data_class_two_transform)

        else:

            raise RuntimeError('Please Enter a Valid Number')

          

        label_change = np.array([self.soft_classes[0]]*len(self.mixed_images))

        self.label_mixed_images = np.eye(num_classes)[label_change]

        self.label_mixed_images[:, self.soft_classes] = 0.5

        

        self.data = self.data + self.mixed_images

        self.label = np.concatenate([self.label, self.label_mixed_images], axis=0)

        

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        data, label = self.data[idx], self.label[idx]

        if self.transform is not None:

            data = self.transform(data)

        return data, label



# Combining images and assigning soft labels for the target vector    
class ProduceCombineAleatoricSoftLabels:
    def __init__(self, subset, class_one, class_two, soft_classes, transform=None):
        print('Applying MixUp - Combining parts of two Images')
        self.subset = subset
        self.transform = transform
        self.class_one = class_one
        self.class_two = class_two
        self.soft_classes = soft_classes
        self.data_class_one = []
        self.data_class_two = []
        self.data = []
        self.label = None
        self.label_mixed_images = None
        
        label = []
        num_classes = 10
        
        index_all = len(self.subset)
        
        for index in range(index_all):
            if self.subset.__getitem__(index)[1] == self.class_one:
                self.data_class_one.append(self.subset.__getitem__(index)[0])
            elif self.subset.__getitem__(index)[1] == self.class_two:
                self.data_class_two.append(self.subset.__getitem__(index)[0])
            else:
                self.data.append(self.subset.__getitem__(index)[0])
                label.append(self.subset.__getitem__(index)[1])
                
        
        use_len = min(len(self.data_class_one), len(self.data_class_two))
        
        self.data_class_one_transform = self.data_class_one[:(use_len//2)]
        self.data_class_one_training = self.data_class_one[(use_len//2):]
        self.label_class_one_training = [self.class_one]*(len(self.data_class_one_training))
        self.data_class_two_transform = self.data_class_two[:(use_len//2)]
        self.data_class_two_training = self.data_class_two[(use_len//2):]
        self.label_class_two_training = [self.class_two]*(len(self.data_class_two_training))

        self.data = self.data + self.data_class_one_training + self.data_class_two_training
        label = label + self.label_class_one_training + self.label_class_two_training
        
        label = np.array(label)
        self.label = np.eye(num_classes)[label]
        
        self.mixed_images = transformCombineImages(self.data_class_one_transform, self.data_class_two_transform)
        label_change = np.array([self.soft_classes[0]]*len(self.mixed_images))
        self.label_mixed_images = np.eye(num_classes)[label_change]
        self.label_mixed_images[:, self.soft_classes] = 0.5
        
        self.data = self.data + self.mixed_images
        self.label = np.concatenate([self.label, self.label_mixed_images], axis=0)
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, label = self.data[idx], self.label[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    

            



# Incorporates only MixUp - Both horizontal and vertical

class TestDataAleatoricMixUp:

    def __init__(self, subset, class_one, class_two, back_label, chosen_mix, transform=None):     

        self.subset = subset

        self.class_one = class_one

        self.class_two = class_two

        self.back_label = back_label

        self.transform = transform

        self.chosen_mix = chosen_mix

        

        index_all = len(self.subset)

        

        self.data = []

        self.label = []

        

        mix_up_one = []

        mix_up_two = []

        

        for i in range(index_all):          

            if self.subset.__getitem__(i)[1] == self.class_one:

                mix_up_one.append(self.subset.__getitem__(i)[0])

            elif self.subset.__getitem__(i)[1] == self.class_two:

                mix_up_two.append(self.subset.__getitem__(i)[0])

            else:

                self.data.append(self.subset.__getitem__(i)[0])

                self.label.append(self.subset.__getitem__(i)[1])

        

        use_len = min(len(mix_up_one), len(mix_up_two))

        

        self.data = self.data + mix_up_one[(use_len//2):] + mix_up_two[(use_len//2):]

        self.label = self.label + [self.class_one]*len(mix_up_one[(use_len//2):]) + [self.class_two]*len(mix_up_two[(use_len//2):])

        

        if self.chosen_mix == 1:

            mixed_images = transformMixUpImagesHorizontal(mix_up_one[:(use_len//2)], mix_up_two[:(use_len//2)])

        elif self.chosen_mix == 2:

            mixed_images = transformMixUpImagesVertical(mix_up_one[:(use_len//2)], mix_up_two[:(use_len//2)])

        else:

            raise RuntimeError('Please enter a valid number')

        

        self.data += mixed_images

        self.label +=  [self.back_label]*len(mixed_images)

        

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        data, label = self.data[idx], self.label[idx]

        if self.transform is not None:

            data = self.transform(data)

        return data, label         


        
# Incorporates only Combining images - equal weight assigned to both images
class TestDataAleatoricCombine:
    def __init__(self, subset, class_one, class_two, back_label, transform=None):     
        self.subset = subset
        self.class_one = class_one
        self.class_two = class_two
        self.back_label = back_label
        self.transform = transform
        
        index_all = len(self.subset)
        
        self.data = []
        self.label = []
        
        mix_up_one = []
        mix_up_two = []
        
        for i in range(index_all):          
            if self.subset.__getitem__(i)[1] == self.class_one:
                mix_up_one.append(self.subset.__getitem__(i)[0])
            elif self.subset.__getitem__(i)[1] == self.class_two:
                mix_up_two.append(self.subset.__getitem__(i)[0])
            else:
                self.data.append(self.subset.__getitem__(i)[0])
                self.label.append(self.subset.__getitem__(i)[1])
        
        use_len = min(len(mix_up_one), len(mix_up_two))
        
        self.data = self.data + mix_up_one[(use_len//2):] + mix_up_two[(use_len//2):]
        self.label = self.label + [self.class_one]*len(mix_up_one[(use_len//2):]) + [self.class_two]*len(mix_up_two[(use_len//2):])
        
        mixed_images = transformCombineImages(mix_up_one[:(use_len//2)], mix_up_two[:(use_len//2)])
        
        self.data += mixed_images
        self.label +=  [self.back_label]*len(mixed_images)
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, label = self.data[idx], self.label[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label  
        

# Incorporates only Combining images - equal weight assigned to both images - For performance evaluation
class TestDataAleatoricCombinePerformance:
    def __init__(self, subset, class_one, class_two):     
        self.subset = subset
        self.class_one = class_one
        self.class_two = class_two
        
        index_all = len(self.subset)
        
        mix_up_one = []
        mix_up_two = []
        
        for i in range(index_all):          
            if self.subset.__getitem__(i)[1] == self.class_one:
                mix_up_one.append(self.subset.__getitem__(i)[0])
            elif self.subset.__getitem__(i)[1] == self.class_two:
                mix_up_two.append(self.subset.__getitem__(i)[0])
        
        use_len = min(len(mix_up_one), len(mix_up_two))

        self.mixed_images = transformCombineImages(mix_up_one[:use_len], mix_up_two[:use_len])
             
    def __len__(self):
        return len(self.mixed_images)
    
    def returnDataList(self):
        return self.mixed_images   
    
    
# Incorporates only MixUp - Both horizontal and vertical - Performance Evaluation
class TestDataAleatoricMixUpPerformance:
    def __init__(self, subset, class_one, class_two, chosen_mix):     
        self.subset = subset
        self.class_one = class_one
        self.class_two = class_two
        self.chosen_mix = chosen_mix
        
        index_all = len(self.subset)
        
        mix_up_one = []
        mix_up_two = []
        
        for i in range(index_all):          
            if self.subset.__getitem__(i)[1] == self.class_one:
                mix_up_one.append(self.subset.__getitem__(i)[0])
            elif self.subset.__getitem__(i)[1] == self.class_two:
                mix_up_two.append(self.subset.__getitem__(i)[0])
            else:
                pass
        
        use_len = min(len(mix_up_one), len(mix_up_two))
                
        if self.chosen_mix == 1:
            self.mixed_images = transformMixUpImagesHorizontal(mix_up_one[:use_len], mix_up_two[:use_len])
        elif self.chosen_mix == 2:
            self.mixed_images = transformMixUpImagesVertical(mix_up_one[:use_len], mix_up_two[:use_len])
        else:
            raise RuntimeError('Please enter a valid number')
                
        
    def __len__(self):
        return len(self.mixed_images)
    
    def returnDataList(self):
        return self.mixed_images       


# For getting combine performance data-set
class ProducePerformanceEvaluationDataset:
    def __init__(self, main_data, black=None, mix=None, combine=None, chosen_mix=None, ood_data=None, transform=None):
        self.main_data = main_data
        
        if chosen_mix == 0:
            self.highALEA = black
        elif chosen_mix == 1:
            self.highALEA = mix
        else:
            self.highALEA = combine
        
        self.ood_data = ood_data
        self.transform = transform
        
        data_ood = []
        
        index_all = len(self.ood_data)
        
        for index in range(index_all):
            data_ood.append(self.ood_data.__getitem__(int(index))[0])
            
        self.data_main, self.label_main = self.main_data.returnDataList()
        data_alea = self.highALEA.returnDataList()
        
        label_ood = [11]*len(data_ood)
        label_alea = [10]*len(data_alea)
        
        self.data_main = self.data_main + data_ood + data_alea
        self.label_main = self.label_main + label_ood + label_alea
        
        
    def __len__(self):
        return len(self.data_main)
    
    
    def __getitem__(self, idx):
        data, label = self.data_main[idx], self.label_main[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    
    
# For getting combine performance data-set
class ProducePerformanceEvaluationDatasetSV:
    def __init__(self, main_data, black=None, mix=None, combine=None, chosen_mix=None, ood_data=None, transform=None):
        self.main_data = main_data
        
        if chosen_mix == 0:
            self.highALEA = black
        elif chosen_mix == 1:
            self.highALEA = mix
        else:
            self.highALEA = combine
        
        self.ood_data = ood_data
        self.transform = transform
            
        self.data_main, self.label_main = self.main_data.returnDataList()
        data_alea = self.highALEA.returnDataList()
        data_ood = self.ood_data.returnDataList()
        
        label_ood = [11]*len(data_ood)
        label_alea = [10]*len(data_alea)
        
        self.data_main = self.data_main + data_ood + data_alea
        self.label_main = self.label_main + label_ood + label_alea
        
        
    def __len__(self):
        return len(self.data_main)
    
    
    def __getitem__(self, idx):
        data, label = self.data_main[idx], self.label_main[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    
    
# For training MLP classifier
class ProduceMLPTrainData:
    def __init__(self, main_data, black=None, mix=None, combine=None, chosen_mix=None, ood_data=None, transform=None):
        self.main_data = main_data
        if chosen_mix == 0:
            self.highALEA = black
        elif chosen_mix == 1:
            self.highALEA = mix
        else:
            self.highALEA = combine
        self.ood_data = ood_data
        self.transform = transform
        
        self.data_main = []
        self.label_main = []
        
        data_ood = []
        
        for data_subset in self.main_data:
            index_all = data_subset.indices
            for index in index_all:
                self.data_main.append(data_subset.dataset.__getitem__(int(index))[0])
                self.label_main.append(data_subset.dataset.__getitem__(int(index))[1])
        
        
        data_ood = self.ood_data.returnDataList()           
        data_alea = self.highALEA.returnDataList()
        
        label_ood = [11]*len(data_ood)
        label_alea = [10]*len(data_alea)
        
        self.data_main = self.data_main + data_ood + data_alea
        self.label_main = self.label_main + label_ood + label_alea
        
        
    def __len__(self):
        return len(self.data_main)
    
    
    def __getitem__(self, idx):
        data, label = self.data_main[idx], self.label_main[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    


# For training MLP classifier
class ProduceMLPTrainDataMNIST:
    def __init__(self, main_data, black=None, mix=None, combine=None, chosen_mix=None, ood_data=None, transform=None):
        self.main_data = main_data
        if chosen_mix == 0:
            self.highALEA = black
        elif chosen_mix == 1:
            self.highALEA = mix
        else:
            self.highALEA = combine
        self.ood_data = ood_data
        self.transform = transform
        
        self.data_main = []
        self.label_main = []
        
        data_ood = []
        
        for data_subset in self.main_data:
            index_all = data_subset.indices
            for index in index_all:
                self.data_main.append(data_subset.dataset.__getitem__(int(index))[0])
                self.label_main.append(data_subset.dataset.__getitem__(int(index))[1])
        
        for data_subset_ood in self.ood_data:
            index_all = data_subset_ood.indices
            for index in index_all:
                data_ood.append(data_subset_ood.dataset.__getitem__(int(index))[0])
        
        
        data_alea = self.highALEA.returnDataList()
        
        label_ood = [11]*len(data_ood)
        label_alea = [10]*len(data_alea)
        
        self.data_main = self.data_main + data_ood + data_alea
        self.label_main = self.label_main + label_ood + label_alea
        
        
    def __len__(self):
        return len(self.data_main)
    
    
    def __getitem__(self, idx):
        data, label = self.data_main[idx], self.label_main[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    
    

class transformSVHN:
    
    def __init__(self, subset):
        
        rgb_weights = [0.2989, 0.5870, 0.1140]
        
        self.subset = subset
        self.data = []
        
        index_all = self.subset.indices
        
        for index in index_all:
            image = self.subset.dataset.__getitem__(index)[0]
            image = np.array(image).astype(np.float32)
                        
            image = np.dot(image[:, :, :3], rgb_weights)
            image = cv2.resize(image, (28, 28))
            
            self.data.append(image)
            
    
    
    def __len__(self):
        return len(self.data)
    
    
    def returnDataList(self):
        return self.data


class transformSVHNnoSub:
    
    def __init__(self, subset):
        
        rgb_weights = [0.2989, 0.5870, 0.1140]
        
        self.subset = subset
        self.data = []
        
        index_all = len(self.subset)
        
        for index in range(index_all):
            image = self.subset.__getitem__(index)[0]
            image = np.array(image).astype(np.float32)
                        
            image = np.dot(image[:, :, :3], rgb_weights)
            image = cv2.resize(image, (28, 28))
            
            self.data.append(image)
            
            
    def __len__(self):
        return len(self.data)
    
    
    def returnDataList(self):
        return self.data 