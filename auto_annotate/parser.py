import argparse

from submodule_utils import (BALANCE_PATCHES_OPTIONS, DATASET_ORIGINS,
        PATCH_PATTERN_WORDS)
from submodule_utils.manifest.arguments import manifest_arguments
from submodule_utils.arguments import (
        AIMArgumentParser,
        dir_path, file_path, dataset_origin, balance_patches_options,
        str_kv, int_kv, subtype_kv, make_dict,
        ParseKVToDictAction, CustomHelpFormatter)

description="""
"""

epilog="""
TODO: --slide_location should be dir_path
TODO: refactor parser out of app.py
TODO: fix --patch_location help
TODO: tests with updated submodule code
"""

@manifest_arguments(default_component_id="auto_annotate",
        description=description, epilog=epilog)
def create_parser(parser):
    parser.add_argument("--log_file_location", type=file_path, required=True,
            help="Path to the log file produced during training.")

    parser.add_argument("--log_dir_location", type=dir_path, required=True,
            help="Path to log directory to save testing logs (i.e. "
            "/path/to/logs/testing/).")

    parser.add_argument("--patch_location", type=dir_path, required=False,
            help="Path to root directory to extract patches into.")

    parser.add_argument("--slide_location", type=str, required=True,
            help="Path to root directory containing all of the slides.")
    
    parser.add_argument("--slide_pattern", type=str,
            default='subtype',
            help="'/' separated words describing the directory structure of the "
            "slide paths. Normally slides paths look like "
            "/path/to/slide/rootdir/subtype/slide.svs and if slide paths are "
            "/path/to/slide/rootdir/slide.svs then simply pass ''.")

    parser.add_argument("--patch_size", type=int, required=True,
            default=1024,
            help="Patch size in pixels to extract from slide to use in evaluation.")
    
    parser.add_argument("--resize_sizes", nargs='+', type=int, required=False,
            help="List of patch sizes in pixels to resize the extracted patchs and save. "
            "Each size should be at most patch_size. Default does not resize.")
    
    parser.add_argument("--evaluation_size", type=int, required=False,
            help="The size in pixel to resize patch before passing to model for evaluation. "
            "evaluation_size should be one of resize_sizes or set to patch_size. "
            "Default uses patch of patch_size for evaluation.")
    
    parser.add_argument("--is_tumor", action='store_true',
            help="Only extract tumor patches. Default extracts tumor and normal patches.")

    parser.add_argument("--num_patch_workers", type=int, default=0,
            help="Number of loader worker processes to multi-process data loading. "
            "Default uses single-process data loading.")
    
    parser.add_argument("--gpu_id", type=int, required=False,
            help="The ID of GPU to select. Default uses GPU with the most free memory.")
    
    parser.add_argument("--num_gpus", type=int, required=False, default=1,
                help="The number of GPUs to use. "
                "Default uses a GPU with the most free memory.")
