import os
import glob

import psutil
import yaml, h5py
import numpy as np
from tqdm import tqdm
import openslide
from openslide import OpenSlide
import torch
from torchvision import transforms
import torch.multiprocessing as mp
import logging


import submodule_utils as utils
from submodule_utils.image.extract import SlidePatchExtractor
from submodule_cv import (ChunkLookupException, setup_log_file,
        gpu_selector, PatchHanger)
import submodule_utils.image.preprocess as image_preprocess
import submodule_cv.tensor.preprocess as tensor_preprocess


logger = logging.getLogger('auto_annotate')

def get_tile_dimensions(os_slide, patch_size):
    width, height = os_slide.dimensions
    return int(width / patch_size), int(height / patch_size)


class AutoAnnotator(PatchHanger):
    """Extracts and classify patches and creates a heatmap in the form of an HDF5 file that is saved at the HDF5 ID 'patch_size/magnification/heatmap_name/n_category

    Attributes
    ----------
    log_file_location : str
        Path to the log file produced during training.

    log_dir_location : str
        Path to log directory to save component logs

    patch_location : str
        Path to directory to save the heatmap H5 files (i.e. /path/to/heatmaps/).

    heatmap_location : str
        Path to directory to save the heatmap H5 files (i.e. /path/to/heatmaps/).
        
        
    slide_location : str
        Path to root directory containing all of the slides.

    slide_pattern

    patch_size : int
        Patch size in pixels to extract from slide to use in evaluation.
    
    instance_name : str

    model_file_location : str

    model_config_location : str

    resize_sizes : list of int
        List of patch sizes in pixels to resize the extracted patchs and save. Size patch_size is guaranteed to be the first size in the list.

    evaluation_size : int or None
        The size in pixel to resize patch before passing to model for evaluation. Size evaluation_size should  be one of resize_sizes.

    num_patch_workers : int

    gpu_id

    slide_location : str
        The root location of all the slides

    FULL_MAGNIFICATION : int
        The magnification at slide full resolution.

    MAX_N_PROCESS : int
        Max number of subprocesses to spawn.
    """
    FULL_MAGNIFICATION = 40
    MAX_N_PROCESS = 10

    def get_magnification(self, resize_size):
        return int(float(resize_size) * float(self.FULL_MAGNIFICATION) \
            / float(self.patch_size))

    def __init__(self, config, log_params):
        self.log_file_location = config.log_file_location
        self.log_dir_location = config.log_dir_location
        self.store_extracted_patches = config.store_extracted_patches or config.patch_location!="./"
        self.generate_heatmap = config.generate_heatmap or config.heatmap_location!="./"
        self.patch_location = config.patch_location
        self.heatmap_location = config.heatmap_location
        self.classification_threshold = config.classification_threshold
        self.slide_location = config.slide_location
        self.slide_pattern = utils.create_patch_pattern(config.slide_pattern)
        self.patch_size = config.patch_size
        self.is_tumor = config.is_tumor
        if config.resize_sizes:
            self.resize_sizes = config.resize_sizes
        else:
            self.resize_sizes = [self.patch_size]
        # if self.patch_size not in self.resize_sizes:
            # self.resize_sizes.insert(0, self.patch_size)
        self.evaluation_size = config.evaluation_size
        if self.evaluation_size and self.evaluation_size not in self.resize_sizes:
            raise ValueError(f"evaluation_size {self.evaluation_size} is not any of {tuple(self.resize_sizes)}")

        if self.evaluation_size:
            self.magnification = int(float(self.evaluation_size) * float(self.FULL_MAGNIFICATION) \
                    / float(self.patch_size))
        else:
            self.magnification = self.FULL_MAGNIFICATION
        # self.extract_foreground = extract_foreground
        self.gpu_id = config.gpu_id
        self.num_gpus = config.num_gpus

        # self.num_tumor_patches = config.num_tumor_patches
        # self.num_normal_patches = config.num_normal_patches
        self.maximum_number_patches = config.maximum_number_patches

        if config.num_patch_workers:
            self.n_process = config.num_patch_workers
        else:
            self.n_process = psutil.cpu_count()
        self.slide_paths = utils.get_paths(self.slide_location, self.slide_pattern,
                extensions=['tiff', 'svs', 'scn'])

        # log parameters
        self.is_binary = log_params['is_binary']
        self.model_file_location = log_params['model_file_location']
        self.model_config_location = log_params['model_config_location']
        self.model_config = self.load_model_config()
        self.instance_name = log_params['instance_name']
        self.raw_subtypes = log_params['subtypes']


        transforms_array = [transforms.ToTensor()]
        if ('normalize' in model_config and model_config['normalize']['normalize']):
            transforms_array.append(transforms.Normalize(mean=model_config['normalize']['mean'], std=model_config['normalize']['std']))
        else :
            transforms_array.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        self.transform = transforms.Compose(transforms_array)

        self.print_parameters(config, log_params)


    
    @classmethod
    def from_log_file(cls, config):
        log_params = utils.extract_yaml_from_file(
                config.log_file_location)
        return cls(config, log_params)

    def print_parameters(self, config, log_params):
        """Print argument parameters"""
        parameters = config.__dict__.copy()
        parameters['log_params'] = log_params
        payload = yaml.dump(parameters)
        print('---') # begin YAML
        print(payload)
        print('...') # end YAML

    def create_hdf_datasets(self, hdf, os_slide, CategoryEnum):
        tile_width, tile_height = get_tile_dimensions(os_slide, self.patch_size)
        group_name = "{}/{}".format(self.patch_size, self.magnification)
        group = hdf.require_group(group_name)
        datasets = { }
        for c in CategoryEnum:
            if c.name in group:
                del group[c.name]
            datasets[c.name] = group.create_dataset(c.name,
                    (tile_height, tile_width, ), dtype='f')
        return datasets

    def extract_patches(self, model, slide_path, class_size_to_patch_path, device=None):
        """Extracts and auto annotates patches using the steps:
         1. Moves a sliding, non-overlaping window to extract each patch to Pillow patch.
         2. Converts patch to ndarray ndpatch and skips to next patch if background
         3. Converts ndpatch to tensor and evaluates with model
         4. If model detects tumor, then save all extracted resizes to patch 

        Parameters
        ----------
        model : torch.nn.Module
            Model used to evaluate extracted patches.

        slide_path : str
            Path of slide to extract patch

        class_size_to_patch_path :dict
            To get the patch path a store evaluated patch using evaluated label name and patch size as keys

        device : torch.device
            To tell evaluation which GPU / CPU to send tensor
        """
        CategoryEnum = utils.create_category_enum(self.is_binary,
                subtypes=self.raw_subtypes)

        logger.info([c.name for c in CategoryEnum])
        logger.info(f'Opening and reading {slide_path} ...')
        os_slide = OpenSlide(slide_path)
    
        shuffle_coordinate = len(self.maximum_number_patches)>0
        slide_patches = SlidePatchExtractor(os_slide, self.patch_size,
                resize_sizes=self.resize_sizes, shuffle=shuffle_coordinate)
        if (self.generate_heatmap) :
            slide_name = utils.path_to_filename(slide_path)
            heatmap_filepath = os.path.join(self.heatmap_location,
                    f'heatmap.{slide_name}.h5')
            hdf = h5py.File(heatmap_filepath, 'w')
            datasets = self.create_hdf_datasets(hdf, os_slide, CategoryEnum)
        temp = ", ".join(f"{ke.upper()}={va}" for ke,va in self.maximum_number_patches.items())
        logger.info(f'Starting Extracting {temp if shuffle_coordinate else len(slide_patches)} Patches From {os.path.basename(slide_path)} on {mp.current_process()}')
        extracted_patches = self.maximum_number_patches.copy()
        with torch.no_grad():
            for data in slide_patches:
                if (shuffle_coordinate and all([x == 0 for x in extracted_patches.values()])):
                    break
                patch, tile_loc, resized_patches = data
                tile_x, tile_y, _, _ = tile_loc
                if self.evaluation_size:
                    ndpatch = image_preprocess.pillow_image_to_ndarray(
                            resized_patches[self.evaluation_size])
                else:
                    ndpatch = image_preprocess.pillow_image_to_ndarray(patch)
                if image_preprocess.check_luminance(ndpatch):
                    if self.evaluation_size:
                        cur_data = self.transform(resized_patches[self.evaluation_size])
                    else :
                        cur_data = self.transform(patch)
                    # cur_data = tensor_preprocess.ndarray_image_to_tensor(ndpatch)
                    # convert tensor image to batch of size 1
                    cur_data = cur_data.cuda().unsqueeze(0)
                    _, pred_prob, _ = model.forward(cur_data)
                    pred_prob = torch.squeeze(pred_prob)

                    pred_label = torch.argmax(pred_prob).type(torch.int).cpu().item()
                    pred_value = torch.max(pred_prob).type(torch.int).cpu().item()
                    if (pred_value >= self.classification_threshold):
                    
                        if (CategoryEnum(pred_label).name.upper() in extracted_patches):
                            # logger.info(extracted_patches)
                            if ( extracted_patches[CategoryEnum(pred_label).name.upper()]==0):
                                continue
                            extracted_patches[CategoryEnum(pred_label).name.upper()]-=1
                        if (self.generate_heatmap) :
                            pred_prob = pred_prob.cpu().numpy().tolist()
                            for c in CategoryEnum:
                                datasets[c.name][tile_y, tile_x] = pred_prob[c.value]
                        
                        if self.store_extracted_patches:
                            if self.is_tumor:
                                if pred_label == 1:
                                    for resize_size in self.resize_sizes:
                                        patch_path = class_size_to_patch_path[CategoryEnum(1).name][resize_size]
                                        resized_patches[resize_size].save(os.path.join(patch_path,
                                                "{}_{}.png".format(tile_x * self.patch_size, tile_y * self.patch_size)))
                            else:
                                for resize_size in self.resize_sizes:
                                    patch_path = class_size_to_patch_path[CategoryEnum(pred_label).name][resize_size]
                                    resized_patches[resize_size].save(os.path.join(patch_path,
                                            "{}_{}.png".format(tile_x * self.patch_size, tile_y * self.patch_size)))
                else:
                    if (self.generate_heatmap) :
                        for c in CategoryEnum:
                            datasets[c.name][tile_y, tile_x] = 0.
        temp = ", ".join(f"{key}={val-extracted_patches[key]}" for key,val in self.maximum_number_patches.items())
        logger.info(f'Finished Extracting {temp if shuffle_coordinate else len(slide_patches)} Patches From {os.path.basename(slide_path)} on {mp.current_process()}')

    def produce_args(self, model, cur_slide_paths):
        """Produce arguments to send to patch extraction subprocess. Creates subdirectories for patches if necessary.

        Parameters
        ----------
        model : torch.nn.Module
            Model used to evaluate extracted patches

        cur_slide_paths : list of str
            List of slide paths. Each path is sent to a subprocess to get slide to evaluate.
        
        device : torch.device
            to tell evaluation subprocess which GPU / CPU to send tensor

        Returns
        -------
        list of tuple
            List of argument tuples to pass through each process. Each argument tuple contains:
             - model (torch.nn.Module) used to evaluate extracted patches
             - slide_path (str) path of slide to extract patch
             - class_size_to_patch_path (dict) to get the patch path a store evaluated patch using evaluated label name and patch size as keys
             - device (torch.device) to tell evaluation which GPU / CPU to send tensor
        """
        CategoryEnum = utils.create_category_enum(self.is_binary,
            subtypes=self.raw_subtypes)
        args = []
        for slide_path in cur_slide_paths:
            slide_id = utils.create_patch_id(slide_path, self.slide_pattern)

            def make_patch_path(class_name):
                size_patch_path = { }
                for resize_size in self.resize_sizes:
                    size_patch_path[resize_size] = os.path.join(
                            self.patch_location, class_name, slide_id,
                            str(resize_size), str(self.get_magnification(resize_size)))
                return size_patch_path
            
            if self.is_tumor:
                tumor_label = CategoryEnum(1).name
                class_size_to_patch_path = { tumor_label: make_patch_path(tumor_label) }
            else:
                class_size_to_patch_path = { c.name: make_patch_path(c.name) \
                        for c in CategoryEnum }
            for size_patch_path in class_size_to_patch_path.values():
                for patch_path in size_patch_path.values():
                    if not os.path.exists(patch_path):
                        os.makedirs(patch_path)
            arg = (model, slide_path, class_size_to_patch_path)
            args.append(arg)
        return args

    def run(self):
        """Run auto annotation
        """
        setup_log_file(self.log_dir_location, self.instance_name)
        print(f"Train instance name: {self.instance_name}")
        if self.n_process > self.MAX_N_PROCESS:
            print(f"Number of CPU processes of {self.n_process} is too high. Setting to {self.MAX_N_PROCESS}")
            self.n_process = self.MAX_N_PROCESS
        print(f"Number of CPU processes: {self.n_process}")
        gpu_devices = gpu_selector(self.gpu_id, self.num_gpus)
        # create torch.device for selected GPU device 
        # device = torch.device(f'cuda:{torch.cuda.current_device()}')
        mp.set_start_method('spawn')
        model = self.build_model(gpu_devices)
        model.load_state(self.model_file_location,)
        model.model.eval()
        model.model.share_memory()
        with torch.no_grad():
            n_slides = len(self.slide_paths)
            prefix = f'Generating from {n_slides} slides: '
            for idx in tqdm(range(0, n_slides, self.n_process),
                    desc=prefix, dynamic_ncols=True):
                # print(f"starting subprocesses for slides numbered {idx} to {idx + self.n_process}")
                cur_slide_paths = self.slide_paths[idx:idx + self.n_process]
                processes = []
                for args in self.produce_args(model, cur_slide_paths):
                    p = mp.Process(target=self.extract_patches, args=args)
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
        print("Done.")
