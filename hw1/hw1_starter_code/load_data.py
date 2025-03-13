import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio
import  cv2

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [
        (i, str(image)  )
        for i, path in zip(labels, paths)
        for image in sampler(list( Path(path).iterdir() ) )
    ]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        config={},
        device=torch.device("cpu"),
        cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_size: size of meta batch size (e.g. number of functions)
            batch_type: train/val/test
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", "./omniglot_resized")
        assert Path(data_folder).is_dir()
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes
        
        character_folders = [
            str(character)
            for family in Path(data_folder).iterdir()
            if family.is_dir()
            for character in  family.iterdir()   
            if character.is_dir() 
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train : num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val :]
        self.device = device
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]

        image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)  # misc.imread(filename)
        
        if image.shape !=self.img_size:
            image = cv2.resize( image,self.img_size)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / 255.0
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image

    def _sample(self):
        """
        Samples a batch for training, validation, or testing
        Args:
            does not take any arguments
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and
                2. label batch has shape [K+1, N, N]
            where K is the number of "shots", N is number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.
            
            3. The value for `self.num_samples_per_class` will be set to K+1 
            since for K-shot classification you need to sample K supports and 
            1 query.
        """

        #############################
        #### YOUR CODE GOES HERE ####
        # 
        
        paths = random.sample(self.folders,self.num_classes)  # each sampler ,random select N classes
        
        mapping_label_idx= {   path:idx  for idx,path in  enumerate(np.unique(paths))}
        # convert to 0- self.num_classes ,facilite mapping between label and label_vector 
        labels=[  mapping_label_idx[path]  for path in paths]   
        
        labels_imagesPath=  get_images(paths, labels, nb_samples=self.num_samples_per_class, shuffle=False) 
        #init
        image_batch= np.empty((self.num_samples_per_class,self.num_classes,self.dim_input),dtype=float)
        label_batch= np.empty((self.num_samples_per_class,self.num_classes,self.num_classes),dtype=int)
        
        
        count_label=dict.fromkeys( mapping_label_idx.values(),0 )
        labels_vecs=np.eye( self.dim_output )
        
        for label,image_path in  labels_imagesPath:
            image_vec  =self.image_file_to_array(image_path, self.dim_input)  # np.array(self.dim_input)
            label_vec=  labels_vecs[label]  # np.array(self.num_classes)
            num_count= count_label[label]
            image_batch[ num_count,label ]=image_vec
            label_batch[num_count,label] =label_vec

            count_label[label]=num_count+1
            

        # shuffle (K+1) set，shuffle last batch    
        last_image_label=np.concatenate([ image_batch[-1], label_batch[-1] ],axis=1) #(N,784+N)
        np.random.shuffle( last_image_label ) 
        image_batch[-1] = last_image_label[:,:self.dim_input ]  
        label_batch[-1] = last_image_label[:,self.dim_input: ]  

        return image_batch,label_batch

        pass
        #############################

    def __iter__(self):
        while True:
            yield self._sample()
