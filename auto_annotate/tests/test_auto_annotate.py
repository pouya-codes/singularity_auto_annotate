import os.path
import pytest
import unittest
import random
from PIL import Image
import torchvision
import torch

import submodule_utils as utils
from submodule_utils.metadata.annotation import GroovyAnnotation
from submodule_cv import (ChunkLookupException, setup_log_file,
    gpu_selector, PatchHanger)
import submodule_cv.models as models
import submodule_utils.image.preprocess as image_preprocess
import submodule_cv.tensor.preprocess as tensor_preprocess

from auto_annotate.tests import (OUTPUT_DIR, OUTPUT_LOG_DIR, OUTPUT_PATCH_DIR,
        SLIDE_DIR)
from auto_annotate.parser import create_parser
from auto_annotate import *
random.seed(1)

def get_slide_subtype_dict(rootdir, slide_pattern):
    slide_paths = utils.get_paths(rootdir, slide_pattern, extensions=['tiff'])
    parts = map(lambda p: utils.create_patch_id(p, slide_pattern), slide_paths)
    slide_subtype = {}
    for part in parts:
        subtype, slide = part.split('/')
        slide_subtype[slide] = subtype
    return slide_subtype

def get_slide_annotation_dict(rootdir, anndir, slide_pattern):
    slide_paths = utils.get_paths(rootdir, slide_pattern, extensions=['tiff'])
    slide_names = map(utils.path_to_filename, slide_paths)
    slide_annotation = {}
    for slide_name in slide_names:
        slide_annotation[slide_name] = GroovyAnnotation(os.path.join(
                anndir, f"{slide_name}.txt"))
    return slide_annotation

def get_slide_counts_dict(rootdir, slide_pattern):
    slide_paths = utils.get_paths(rootdir, slide_pattern, extensions=['tiff'])
    slide_names = map(utils.path_to_filename, slide_paths)
    slide_counts = {}
    for slide_name in slide_names:
        slide_counts[slide_name] = {
                'true_tumor': 0,
                'false_tumor': 0,
                'true_normal': 0,
                'false_normal': 0,
                'unannotated': 0,
                'total': 0}
    return slide_counts

def get_slide_names(rootdir, slide_pattern):
    slide_paths = utils.get_paths(rootdir, slide_pattern, extensions=['tiff'])
    return map(utils.path_to_filename, slide_paths)

def test_auto_annotate():
    """Generate patches using a model, and then test dataset labels. Currently using

    || Slide Name     || Extracted Patch Count ||
    | MMRd/VOA-1099A   | 12316 |
    | p53abn/VOA-3088B | 11374 |
    | p53wt/VOA-3266C  | 6662  |
    | POLE/VOA-1932A   | 8604  |
    | Total            | 38956 |

    To run tests:
        (1) add the slides (symlinks) and their annotations to
    auto_annotate/tests/mock/slides
    auto_annotate/tests/mock/annotations
        (2) set the path of training log for the binary T/N model to log_file_location
    """
    log_file_location = '/projects/ovcare/classification/cchen/ml/data/test_ec/logs/train/log_test_ec_20200824-155806.txt'
    slide_pattern = 'subtype'
    slide_pattern = utils.create_patch_pattern(slide_pattern)
    patch_size = 1024
    resize_sizes = [256]
    evaluation_size = 256
    
    # after extraction variables
    annotation_location = 'auto_annotate/tests/mock/annotations'
    patch_pattern = 'annotation/subtype/slide/patch_size/magnification'
    patch_pattern = utils.create_patch_pattern(patch_pattern)
    filter_labels = { 'patch_size': '256',
            'magnification': '10' }
    args_str = f"""
    from-arguments
    --log_file_location {log_file_location}
    --log_dir_location {OUTPUT_LOG_DIR}
    --patch_location {OUTPUT_PATCH_DIR}
    --slide_location {SLIDE_DIR}
    --slide_pattern {slide_pattern}
    --patch_size {patch_size}
    --resize_sizes {utils.list_to_space_sep_str(resize_sizes)}
    --evaluation_size {evaluation_size}
    """
    print()
    print('test_auto_annotate/setup AutoAnnotator')
    parser = create_parser()
    config = parser.get_args(args_str.split())
    aa = AutoAnnotator.from_log_file(config)
    print('test_auto_annotate/AutoAnnotator.run()')
    aa.run()
    print('test_auto_annotate/setup DeepModel')
    model = models.DeepModel(
            utils.load_json(aa.model_config_location))
    model.load_state(aa.model_file_location)
    patch_paths = utils.get_patch_paths(aa.patch_location,
            patch_pattern, filter_labels)
    slide_subtype = get_slide_subtype_dict(aa.slide_location, aa.slide_pattern)
    slide_annotation = get_slide_annotation_dict(aa.slide_location,
            annotation_location, aa.slide_pattern)
    slide_counts = get_slide_counts_dict(aa.slide_location, aa.slide_pattern)
    assert len(patch_paths) > 0
    idx = 0
    model.model.eval()
    with torch.no_grad():
        print('test_auto_annotate/checking patches')
        for patch_path in patch_paths:
            patch_id = utils.create_patch_id(patch_path, patch_pattern)
            subtype = patch_id.split('/')[patch_pattern['subtype']]
            annotation = patch_id.split('/')[patch_pattern['annotation']]
            is_tumor_01 = 1 if annotation == 'Tumor' else 0
            slide_name = patch_id.split('/')[patch_pattern['slide']]

            ## Test slide subtype matches patch subtype
            assert slide_subtype[slide_name] == subtype

            ## Test that dataset label equals classifier predicted label
            patch = Image.open(patch_path).convert('RGB')
            ndpatch = image_preprocess.pillow_image_to_ndarray(patch)
            cur_data = tensor_preprocess.ndarray_image_to_tensor(ndpatch)
            cur_data = cur_data.cuda().unsqueeze(0)
            _, pred_prob, _ = model.forward(cur_data)
            pred_prob = torch.squeeze(pred_prob)
            pred_label = torch.argmax(pred_prob).type(torch.int).cpu().item()
            assert pred_label == is_tumor_01
            
            ## Test that dataset label equals annotated label
            ## may fail; if so then count the failures
            x, y = patch_id.split('/')[-1].split('_')
            x = int(x)
            y = int(y)
            manual_annotation = slide_annotation[slide_name].points_to_label(
                    np.array([[x, y], [x+1024, y], [x, y+1024], [x+1024, y+1024]]))
            if manual_annotation == 'Tumor':
                # tumor label
                if annotation == 'Tumor':
                    slide_counts[slide_name]['true_tumor'] += 1
                if annotation == 'Normal':
                    slide_counts[slide_name]['false_tumor'] += 1

            elif manual_annotation is None:
                # unannotated
                slide_counts[slide_name]['unannotated'] += 1

            else:
                # normal label
                if annotation == 'Tumor':
                    slide_counts[slide_name]['false_normal'] += 1
                if annotation == 'Normal':
                    slide_counts[slide_name]['true_normal'] += 1
            
            slide_counts[slide_name]['total'] += 1

            if idx % 300 == 0:
                print(f"iteration {idx} sample ID: ", patch_id)
            idx += 1

    print('test_auto_annotate/done checks')
    print(
            "Checking that patches annotated as tumor/normal by pathologist"
            "used to train tumor/normal classifier when extracted again by"
            "AutoAnnotate is has the same label as the pathologist annotations.")
    print("Watch out for false_tumor and false_normal counts.")
    for slide_name in get_slide_names(aa.slide_location, aa.slide_pattern):
        print(slide_name)
        print(slide_counts[slide_name])


