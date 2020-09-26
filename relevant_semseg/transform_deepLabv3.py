import random
import numpy as np
import cv2
import collections
import numbers

value_scale = 255
mean_crop = [0.485, 0.458, 0.408]
mean_crop = [item * value_scale for item in mean_crop]
rotate_min = -10
rotate_max = 10


def generate_scale_label(image, label):
    image = np.array(image)
    label = np.array(label)
    w = 0.5 + random.randint(0, 15)/10.0
    h = 0.5 + random.randint(0, 15)/10.0
    image = cv2.resize(image, None, fx=w, fy=h, interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, None, fx=w, fy=h, interpolation=cv2.INTER_NEAREST)
    return image, label


class RandomCropsTrain:
    def __init__(self, crop_size=(512, 512), ignore_label=255, padding=None, patch_size=None):
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label
        self.patch_size = patch_size
        self.num_train_images = 2975
        self.count = -1
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        if padding is None:
            self.padding = [0.0, 0.0, 0.0]
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))

    def __call__(self, image, label):
        image, label = generate_scale_label(image, label)
        image, label = rand_rotate(image, label)
        image, label = rand_h_flip(image, label)

        image = np.asarray(image, np.float32)
        label = np.array(label)

        for key, value in self.id_to_trainid.items():
            label[label == key] = value

        mean = (103.93, 116.77, 123.68)
        image -= mean

        img_h, img_w = label.shape

        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)

        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.padding)
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        
        self._fill_patch(image, label)  
        image = image.transpose((2, 0, 1))  
        
        return image.copy(), label.copy()
    
    def _fill_patch(self, image: np.ndarray, label: np.ndarray) -> None:
        """Selects a random square patch, fill it with a certain color and assign a label at the same patch position
            in the label tensor"""
        # Number of training images = 2975            
        shape_image = image.shape 
        self.count += 1
        
        if self.count == self.num_train_images:
            self.count = 0
        
        patch_x = random.randint(0, shape_image[0] - self.patch_size)
        patch_y = random.randint(0, shape_image[1] - self.patch_size)
              
        image[patch_x:patch_x+self.patch_size, patch_y:patch_y+self.patch_size, :] = 0.
        if self.count % 2 == 0:
            label[patch_x:patch_x+self.patch_size, patch_y:patch_y+self.patch_size] = 10 # sky
        else:
            label[patch_x:patch_x+self.patch_size, patch_y:patch_y+self.patch_size] = 12 # rider
        
        
class ValidationTransform:
    def __init__(self, ignore_label=255):
        self.ignore_label = ignore_label

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __call__(self, image, label):
        image = np.asarray(image, np.float32)
        label = np.asarray(label, dtype=np.int32)

        for key, value in self.id_to_trainid.items():
            label[label == key] = value

        image, label = center_crop(image, label)

        mean = (103.93, 116.77, 123.68)
        image -= mean

        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy()


class TestTransform:
    def __init__(self, ignore_label=255, patch_size=None):
        self.ignore_label = ignore_label
        self.patch_size = patch_size
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __call__(self, image, label):
        image = np.asarray(image, np.float32)
        label = np.asarray(label, dtype=np.int32)

        for key, value in self.id_to_trainid.items():
            label[label == key] = value

        # image, label = center_crop(image, label)

        mean = (103.93, 116.77, 123.68)
        image -= mean

        self._draw_patch(image)
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy()
    
    def _draw_patch(self, image: np.ndarray) -> None:
        """Function which draws a patch at random locations within an image"""
        shape_image = image.shape
             
        patch_x = random.randint(0, shape_image[0] - self.patch_size)
        patch_y = random.randint(0, shape_image[1] - self.patch_size)
              
        image[patch_x:patch_x+self.patch_size, patch_y:patch_y+self.patch_size, :] = 0
        


class Crop(object):
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = [0.0, 0.0, 0.0]
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return image, label



class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.ignore_label)
        return image, label



class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label



class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label



center_crop = Crop([712, 712], padding=None)
rand_rotate = RandRotate([rotate_min, rotate_max], padding=[0.0, 0.0, 0.0])
rand_h_flip = RandomHorizontalFlip()
