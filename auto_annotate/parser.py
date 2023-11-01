import argparse

from submodule_utils import (BALANCE_PATCHES_OPTIONS, DATASET_ORIGINS,
        PATCH_PATTERN_WORDS)
from submodule_utils.manifest.arguments import manifest_arguments
from submodule_utils.arguments import (
        AIMArgumentParser,
        dir_path, file_path, dataset_origin, balance_patches_options,
        str_kv, int_kv, subtype_kv, make_dict,
        ParseKVToDictAction, CustomHelpFormatter)

description="""Use trained model to extract patches.

Auto annotate extracts foreground patches from WSI, predicts whether the patch is a Tumor or Normal (aka. non-tumor) patch, and then extracts and downsamples the patches.

If slide dataset has a slide_pattern of 'subtype'. Each patch dataset has the pattern 'annotation/subtype/slide/patch_size/magnification'. For example, if the slide paths look like:

/path/to/slides/MMRd/VOA-1099A.tiff
/path/to/slides/POLE/VOA-1932A.tiff
/path/to/slides/p53abn/VOA-3088B.tiff
/path/to/slides/p53wt/VOA-3266C.tiff

And we use '--resize_sizes 512' then the extracted patch paths will look something like:

/path/to/patches/Tumor/MMRd/VOA-1099A/512/20/30140_12402.png
/path/to/patches/Normal/MMRd/VOA-1099A/512/20/42038_12402.png
...
/path/to/patches/Tumor/POLE/VOA-1932A/512/20/42038_12402.png
/path/to/patches/Normal/POLE/VOA-1932A/512/20/30140_12402.png
...
/path/to/patches/Tumor/p53abn/VOA-3088B/512/20/42038_12402.png
/path/to/patches/Normal/p53abn/VOA-3088B/512/20/30140_12402.png
...
/path/to/patches/Tumor/p53wt/VOA-3266C/512/20/30140_12402.png
/path/to/patches/Normal/p53wt/VOA-3266C/512/20/42038_12402.png
...

This component looks in the YAML section of the training log to obtain the trained model and it's hyperparameters."""

epilog=None

@manifest_arguments(default_component_id="auto_annotate",
        description=description, epilog=epilog)
def create_parser(parser):
    parser.add_argument("--log_file_location", type=file_path, required=True,
            help="Path to the log file produced during training.")

    parser.add_argument("--log_dir_location", type=dir_path, required=True,
            help="Path to log directory to save testing logs (i.e. "
            "/path/to/logs/testing/).")

    parser.add_argument("--slide_location", type=str, required=dir_path,
            help="Path to root directory containing all of the slides.")

    parser.add_argument("--store_extracted_patches", action='store_true',
            help="Store extracted patches. Default does not store extracted patches.")

    parser.add_argument("--patch_location", type=dir_path, required=False,
            help="Path to root directory to extract patches into.")

    parser.add_argument("--generate_heatmap", action='store_true',
            help="Generate heatmaps. Default does not generate heatmap.")

    parser.add_argument("--heatmap_location", type=dir_path, required=True,
            help="Path to directory to save the heatmap H5 files (i.e. "
            "/path/to/heatmaps/).")

    parser.add_argument("--classification_threshold", type=float, default=0, required=False,
                help="Minimum obtained probability for the most probable class."
                "Default: 0")


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

#     parser.add_argument("--num_tumor_patches", type=int, required=False, default=-1,
#             help="The maximum number of extracted tumor patches for each slide. "
#             "Default extracts all the patches.")

#     parser.add_argument("--num_normal_patches", type=int, required=False, default=-1,
#             help="The maximum number of extracted normal patches for each slide. "
#             "Default extracts all the patches.")

    parser.add_argument("--maximum_number_patches", nargs='+', type=subtype_kv,
                action=ParseKVToDictAction, required=False, default={},
                help="Caution: when you use this flag the code while shuffles the extracted patches from each slide.space separated words describing subtype=maximum_number_of_extracted_patches pairs for each slide. "
                "Example: if want to extract 500 Tumor, 0 Normal patches and unlimited POLE patches "
                "then the input should be 'Tumor=500 Normal=0 POLE=-1'")
