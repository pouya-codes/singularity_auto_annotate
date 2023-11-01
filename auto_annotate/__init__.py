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
import staintools
from PIL import Image
import datetime
import random

import submodule_utils as utils
from submodule_utils.thumbnail import PlotThumbnail
from submodule_utils.fake_annotation import FakeAnnotation
from submodule_utils.image.extract import SlidePatchExtractor
from submodule_cv import (ChunkLookupException, setup_log_file,
                          gpu_selector, PatchHanger)
import submodule_utils.image.preprocess as image_preprocess
import submodule_cv.tensor.preprocess as tensor_preprocess
from submodule_utils.metadata.tissue_mask import TissueMask

logger = logging.getLogger('auto_annotate')


def get_tile_dimensions(os_slide, patch_size, patch_overlap):
    width, height = os_slide.dimensions
    stride = int((1 - patch_overlap) * patch_size)
    tile_width = int((width - patch_size) / stride + 1)
    tile_height = int((height - patch_size) / stride + 1)
    return tile_width, tile_height


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
    MAX_N_PROCESS = 48

    def get_magnification(self, resize_size):
        return int(float(resize_size) * float(self.FULL_MAGNIFICATION) \
                   / float(self.patch_size))

    @property
    def should_use_manifest(self):
        return self.load_method == 'use-manifest'

    @property
    def should_use_directory(self):
        return self.load_method == 'use-directory'

    def get_slide_paths(self):
        """Get paths of slides that should be extracted.
        """
        if self.should_use_manifest:
            return self.manifest['slide_path']
        elif self.should_use_directory:
            return utils.get_paths(self.slide_location, self.slide_pattern,
                                   extensions=['tiff', 'tif', 'svs', 'scn'])
        else:
            raise NotImplementedError()

    def load_slide_tissue_mask(self):
        """Load tissue masks from slide names.
        """
        if self.should_use_manifest:
            generator = self.manifest['mask_path']
        elif self.should_use_directory:
            generator = os.listdir(self.mask_location)
        else:
            raise NotImplementedError()
        # if it is .png, we need to know the true size of slide
        # since the mask is scale down version of it
        list_slides = self.get_slide_paths()
        slide_tissue_mask = {}
        for file in generator:
            if file.endswith(".png") or file.endswith(".txt"):
                slide_name = utils.path_to_filename(file)
                slide_path = utils.find_slide_path(list_slides, slide_name)
                if slide_path is None:  # the path to that slide was not found
                    continue
                else:
                    print(slide_path)
                    try:
                        os_slide = OpenSlide(slide_path)
                    except:
                        continue
                    slide_size = os_slide.dimensions
                if self.should_use_manifest:
                    filepath = file
                else:
                    filepath = os.path.join(self.mask_location, file)
                slide_tissue_mask[slide_name] = TissueMask(filepath, 0.4, self.patch_size,
                                                           slide_size)
        return slide_tissue_mask

    def __init__(self, config, log_params):

        self.load_method = config.load_method
        if self.should_use_manifest:
            self.manifest = utils.read_manifest(config.manifest_location)
        elif self.should_use_directory:
            self.slide_location = config.slide_location
            self.slide_pattern = utils.create_patch_pattern(config.slide_pattern)
            self.mask_location = config.mask_location
        self.log_file_location = config.log_file_location
        self.log_dir_location = config.log_dir_location
        self.store_extracted_patches = config.store_extracted_patches
        self.store_patches_statistics = config.store_patches_statistics
        self.generate_annotation = config.generate_annotation
        self.skip_area = config.skip_area
        self.generate_heatmap = config.generate_heatmap or config.heatmap_location != "./"
        self.patch_location = config.patch_location
        self.patch_overlap = config.patch_overlap
        self.hd5_location = config.hd5_location
        self.heatmap_location = config.heatmap_location
        self.classification_threshold = config.classification_threshold
        self.classification_max_threshold = config.classification_max_threshold
        self.label = config.label
        self.patch_size = config.patch_size
        self.is_tumor = config.is_tumor
        self.store_thumbnail = config.store_thumbnail
        self.use_radius = config.use_radius
        self.radius = config.radius
        self.use_color_norm = config.use_color_norm
        self.method = config.method
        self.reference_images = config.reference_image
        self.use_standarizer = config.use_standarizer

        if self.should_use_directory and self.mask_location is not None:
            self.use_mask = True
            self.mask = self.load_slide_tissue_mask()
        elif self.should_use_manifest and 'mask_path' in self.manifest:
            self.use_mask = True
            self.mask = self.load_slide_tissue_mask()
        else:
            self.use_mask = False

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
        self.old_version = config.old_version
        # self.num_tumor_patches = config.num_tumor_patches
        # self.num_normal_patches = config.num_normal_patches
        self.maximum_number_patches = config.maximum_number_patches

        if config.num_patch_workers:
            self.n_process = config.num_patch_workers
        else:
            self.n_process = psutil.cpu_count()
        self.slide_paths = self.get_slide_paths()
        self.slide_idx = config.slide_idx

        # log parameters
        self.is_binary = log_params['is_binary']
        self.model_file_location = log_params['model_file_location']
        self.model_config_location = log_params['model_config_location']
        self.model_config = self.load_model_config()
        self.instance_name = log_params['instance_name']
        self.raw_subtypes = log_params['subtypes']

        if self.use_color_norm:
            normalizer = staintools.StainNormalizer(method=self.method)
            ref_image = np.array(Image.open(random.choice(self.reference_images)))
            if self.use_standarizer:
                ref_image = staintools.LuminosityStandardizer.standardize(ref_image)
            normalizer.fit(ref_image)
            self.normalizer = normalizer.transform

        transforms_array = [transforms.ToTensor()]
        if ('normalize' in self.model_config and self.model_config['normalize']['use_normalize']):
            transforms_array.append(transforms.Normalize(mean=self.model_config['normalize']['mean'],
                                                         std=self.model_config['normalize']['std']))
        else:
            transforms_array.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
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
        print('---')  # begin YAML
        print(payload)
        print('...')  # end YAML

    def create_hdf_datasets(self, hdf, os_slide, CategoryEnum):
        tile_width, tile_height = get_tile_dimensions(os_slide, self.patch_size, self.patch_overlap)
        group_name = "{}/{}".format(self.patch_size, self.magnification)
        group = hdf.require_group(group_name)
        datasets = {}
        for c in CategoryEnum:
            if c.name in group:
                del group[c.name]
            datasets[c.name] = group.create_dataset(c.name,
                                                    (tile_height, tile_width,), dtype='f')
        return datasets

    def check_background(self, resized_patches, patch):
        if self.evaluation_size:
            ndpatch = image_preprocess.pillow_image_to_ndarray(
                resized_patches[self.evaluation_size])
        else:
            ndpatch = image_preprocess.pillow_image_to_ndarray(patch)
        return image_preprocess.check_luminance(ndpatch)

    def check_tissue(self, slide_name, x, y):
        label = self.mask[slide_name].points_to_label(
            np.array([[x, y],
                      [x, y + self.patch_size],
                      [x + self.patch_size, y + self.patch_size],
                      [x + self.patch_size, y]]))
        if not label:
            return False
        return True

    def handle_radius_coordiante(self, os_slide, Coords):
        patches = []
        resizeds = []
        for coord in Coords:
            x_, y_ = coord
            patch = image_preprocess.extract(os_slide, x_, y_, self.patch_size)
            if self.use_color_norm:
                patch = self.normalize_patch(patch)
                if patch is None:  # blank ones
                    continue
            if self.resize_sizes:
                resized_patches = {}
                for resize_size in self.resize_sizes:
                    if resize_size == self.patch_size:
                        resized_patches[resize_size] = patch
                    else:
                        resized_patches[resize_size] = image_preprocess.resize(patch, resize_size)
            if self.check_background(resized_patches, patch):
                Coords.remove(coord)
                continue
            patches.append(patch)
            resizeds.append(resized_patches)
        return Coords, patches, resizeds

    def normalize_patch(self, patch):
        try:
            patch = np.array(patch)
            if self.use_standarizer:
                patch = staintools.LuminosityStandardizer.standardize(patch)
            patch = self.normalizer(patch)
            patch = Image.fromarray(patch)
            return patch
        except:
            # it could not be normalized (probably blank)
            return None

    def extract_patches(self, model, slide_path, class_size_to_patch_path, device=None, send_end=None):
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
        paths = []
        extracted_coordinates = []
        # If we are interested in one category such as Tumor
        # Only we need Tumor's probability
        if self.label is not None:
            index = utils.find_value_from_name_enum(self.label, CategoryEnum)

        logger.info(f'Opening and reading {slide_path} ...')
        os_slide = OpenSlide(slide_path)

        shuffle_coordinate = len(self.maximum_number_patches) > 0
        slide_patches = SlidePatchExtractor(os_slide, self.patch_size, patch_overlap=self.patch_overlap,
                                            resize_sizes=self.resize_sizes, shuffle=shuffle_coordinate)
        slide_name = utils.path_to_filename(slide_path)
        hd5_file_path = os.path.join(self.hd5_location, f"{slide_name}.h5")
        # annotation
        if self.generate_annotation:
            fake_annot = FakeAnnotation(slide_name, os_slide, hd5_file_path,
                                        self.magnification, self.patch_size,
                                        self.skip_area)

        if (self.generate_heatmap):
            heatmap_filepath = os.path.join(self.heatmap_location,
                                            f'heatmap.{slide_name}.h5')
            hdf = h5py.File(heatmap_filepath, 'w')
            datasets = self.create_hdf_datasets(hdf, os_slide, CategoryEnum)
        temp = ", ".join(f"{ke.upper()}={va}" for ke, va in self.maximum_number_patches.items())
        logger.info(
            f'Starting Extracting {temp if shuffle_coordinate else len(slide_patches)} Patches From {os.path.basename(slide_path)} on {mp.current_process()}')

        # if (len(self.maximum_number_patches) > 0):
        #     extracted_patches = self.maximum_number_patches.copy()
        # else:
        #
        extracted_patches = {}
        for enum in CategoryEnum:
            extracted_patches[enum.name.upper()] = 0

        with torch.no_grad():
            for data in slide_patches:
                if (shuffle_coordinate and all([extracted_patches[x] >= self.maximum_number_patches[x]
                 for x in extracted_patches.keys()])):
                    break
                patch, tile_loc, resized_patches = data
                tile_x, tile_y, x, y = tile_loc
                if self.use_mask and slide_name in self.mask:  # check if it is tissue
                    check_tissue = self.check_tissue(slide_name, x, y)
                    if not check_tissue:
                        continue
                if self.check_background(resized_patches, patch):
                    stride = int((1 - self.patch_overlap) * self.patch_size)
                    if self.use_radius:
                        Coords = utils.get_circular_coordinates(self.radius, x, y, stride,
                                                                os_slide.dimensions, self.patch_size)
                        Coords, patches, resizeds = self.handle_radius_coordiante(os_slide, Coords)
                    else:
                        if self.use_color_norm:
                            patch = self.normalize_patch(patch)
                            if patch is None:  # blank ones
                                continue
                            for size, patch_ in resized_patches.items():
                                resized_patches[size] = self.normalize_patch(patch_)
                        Coords = [(x, y)]
                        patches = [patch]
                        resizeds = [resized_patches]

                    for coord, patch, resized_patches in zip(Coords, patches, resizeds):
                        x, y = coord
                        if self.use_mask and slide_name in self.mask:  # check if it is tissue
                            check_tissue = self.check_tissue(slide_name, x, y)
                            if not check_tissue:
                                continue
                        if (x, y) in extracted_coordinates:  # it has been previously extracted (usefull for radius)
                            continue
                        tile_x = int(x / stride)
                        tile_y = int(y / stride)
                        if self.evaluation_size:
                            cur_data = self.transform(resized_patches[self.evaluation_size])
                        else:
                            cur_data = self.transform(patch)
                        # convert tensor image to batch of size 1
                        if torch.cuda.is_available():
                            cur_data = cur_data.cuda().unsqueeze(0)
                        else:
                            cur_data = cur_data.unsqueeze(0)
                        _, pred_prob, _ = model.forward(cur_data)
                        pred_prob = torch.squeeze(pred_prob)

                        if self.label is None:
                            pred_label = torch.argmax(pred_prob).type(torch.int).cpu().item()
                            pred_value = torch.max(pred_prob).type(torch.float).cpu().item()
                        else:
                            pred_label = int(index)
                            pred_value = pred_prob[pred_label].type(torch.float).cpu().item()
                        pred_label_name = CategoryEnum(pred_label).name.upper()

                        if (self.classification_threshold <= pred_value <= self.classification_max_threshold):
                            # if (CategoryEnum(pred_label).name.upper() in extracted_patches):

                            if (pred_label_name in self.maximum_number_patches and
                                    extracted_patches[pred_label_name] == self.maximum_number_patches[pred_label_name]):
                                continue
                            extracted_patches[CategoryEnum(pred_label).name.upper()] += 1
                            extracted_coordinates.append((x, y))

                            if (self.generate_heatmap):
                                pred_prob = pred_prob.cpu().numpy().tolist()
                                for c in CategoryEnum:
                                    datasets[c.name][tile_y, tile_x] = pred_prob[c.value]

                            if self.is_tumor:
                                if pred_label == 1:
                                    if self.generate_annotation:
                                        fake_annot.add_poly(x, y)
                                    for resize_size in self.resize_sizes:
                                        patch_path = class_size_to_patch_path[CategoryEnum(1).name][resize_size]
                                        patch_path_ = os.path.join(patch_path,
                                                                   "{}_{}.png".format(x, y))
                                        paths.append(patch_path_)
                                        if self.store_extracted_patches:
                                            resized_patches[resize_size].save(patch_path_)
                            else:
                                for resize_size in self.resize_sizes:
                                    patch_path = class_size_to_patch_path[CategoryEnum(pred_label).name][resize_size]
                                    patch_path_ = os.path.join(patch_path,
                                                               "{}_{}.png".format(x, y))
                                    paths.append(patch_path_)
                                    if self.store_extracted_patches:
                                        resized_patches[resize_size].save(patch_path_)
                else:
                    if (self.generate_heatmap):
                        for c in CategoryEnum:
                            datasets[c.name][tile_y, tile_x] = 0.
        logger.info(
            f'Finished Extracting {", ".join(f"{key}={val}" for key, val in extracted_patches.items())} Patches From {os.path.basename(slide_path)} on {mp.current_process()}')
        utils.save_hdf5(hd5_file_path, paths, self.patch_size)
        if self.store_thumbnail:
            mask = self.mask[slide_name] if self.use_mask and slide_name in self.mask else None
            PlotThumbnail(slide_name, os_slide, hd5_file_path, None, mask=mask)
        if self.generate_annotation:
            fake_annot.run()
        send_end.send([slide_path, extracted_patches])

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
            if self.should_use_manifest:
                slide_name = utils.path_to_filename(slide_path)
                if 'subtype' in self.manifest:
                    idx = self.manifest['slide_path'].index(slide_path)
                    subtype_ = self.manifest['subtype'][idx]
                    slide_id = f"{subtype_}/{slide_name}"
                else:
                    slide_id = slide_name
            else:
                slide_id = utils.create_patch_id(slide_path, self.slide_pattern)

            def make_patch_path(class_name):
                size_patch_path = {}
                for resize_size in self.resize_sizes:
                    size_patch_path[resize_size] = os.path.join(
                        self.patch_location, class_name, slide_id,
                        str(resize_size), str(self.get_magnification(resize_size)))
                return size_patch_path

            if self.is_tumor:
                tumor_label = CategoryEnum(1).name
                class_size_to_patch_path = {tumor_label: make_patch_path(tumor_label)}
            else:
                class_size_to_patch_path = {c.name: make_patch_path(c.name) \
                                            for c in CategoryEnum}

            if self.store_extracted_patches:
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
        if self.slide_idx is not None:
            self.slide_paths = utils.select_slides(self.slide_paths, self.slide_idx, self.n_process)
        mp.set_start_method('spawn')
        if torch.cuda.is_available():
            print("Start using GPU ... ")
            # gpu_devices = gpu_selector(self.gpu_id, self.num_gpus)
            model = self.build_model()
        else:
            print("Start using CPU ... ")
            model = self.build_model(None)
        if self.old_version:
            model.load_state_old_version(self.model_file_location)
        else:
            model.load_state(self.model_file_location, )
        model.model.eval()
        model.model.share_memory()
        with torch.no_grad():
            n_slides = len(self.slide_paths)
            prefix = f'Generating from {n_slides} slides: '
            results_to_write = []
            for idx in tqdm(range(0, n_slides, self.n_process),
                            desc=prefix, dynamic_ncols=True):
                # print(f"starting subprocesses for slides numbered {idx} to {idx + self.n_process}")
                cur_slide_paths = self.slide_paths[idx:idx + self.n_process]
                processes = []
                recv_end_list = []
                for args in self.produce_args(model, cur_slide_paths):
                    recv_end, send_end = mp.Pipe(False)
                    recv_end_list.append(recv_end)
                    p = mp.Process(target=self.extract_patches,
                                   args=(args[0], args[1], args[2], args[3] if len(args) > 3 else None, send_end))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                if self.store_patches_statistics:
                    results_to_write.extend(map(lambda x: x.recv(), recv_end_list))

        if self.store_patches_statistics:
            output_file_path = os.path.join(self.log_dir_location,
                                            f'Patches_Statistics_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}.csv')
            output_file = open(output_file_path, "w")
            Categories = utils.create_category_enum(self.is_binary,
                                                      subtypes=self.raw_subtypes)
            output_file.write(f"slide_name,{','.join([enum.name.upper() for enum in Categories])}\n")
            output_file.write("\n".join([f"{path},{','.join([str(value) for value in extracted_patches.values()])}" for path, extracted_patches in results_to_write]))
            output_file.close()

        print("Done.")
