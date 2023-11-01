import argparse

from submodule_utils import (BALANCE_PATCHES_OPTIONS, DATASET_ORIGINS,
        PATCH_PATTERN_WORDS)
from submodule_utils.manifest.arguments import manifest_arguments
from submodule_utils.arguments import (
        AIMArgumentParser,
        dir_path, file_path, dataset_origin, balance_patches_options,
        str_kv, int_kv, subtype_kv, make_dict,
        ParseKVToDictAction, CustomHelpFormatter)

from submodule_utils import DEAFULT_SEED

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

    parser.add_argument("--store_extracted_patches", action='store_true',
            help="Store extracted patches. Default does not store extracted patches.")

    parser.add_argument("--store_patches_statistics", action='store_true',
            help="Store statistics of the extracting patches process as a csv file at log_dir_location. this file "
                 "contains [slide_name, number of extracted patches for each class]. Default does not store the csv file.")

    parser.add_argument("--store_thumbnail", action='store_true',
            help="Whether or not save thumbnail with showing the position "
            "of extracted patches. If yes, it will be stored at a folder called "
            "Thumbnails in HD5 folder.")

    parser.add_argument("--generate_annotation", action='store_true',
            help="Whether or not save annotation for slide "
            "If yes, it will be stored at a folder called Annotation in HD5 folder. "
            "Also a folder called Thubmnails will be created in Annotation that shows the "
            "annotation on thumbnails. Only works for Tumor.")

    parser.add_argument("--skip_area", type=int,
            default=None, required=False,
            help="If this flag is set, when the final annotation is created, "
            "polygons (areas) with less than area of determined will be skipped. "
            "note that smallest area is patch_size*patch_size.")

    parser.add_argument("--patch_location", type=dir_path, default="./",
            help="Path to root directory to extract patches into.")

    parser.add_argument("--patch_overlap", type=float, default=0,
            help="Overlap between extracted patches.")

    parser.add_argument("--hd5_location", type=dir_path, required=True,
            help="Path to root directory to save hd5 into.")

    parser.add_argument("--generate_heatmap", action='store_true',
            help="Generate heatmaps. Default does not generate heatmap.")

    parser.add_argument("--heatmap_location", type=dir_path, default="./",
            help="Path to directory to save the heatmap H5 files (i.e. "
            "/path/to/heatmaps/).")

    parser.add_argument("--classification_threshold", type=float, default=0,
            help="Minimum obtained probability for the most probable class")

    parser.add_argument("--classification_max_threshold", type=float, default=1.0,
            help="Maximum obtained probability for the most probable class")

    parser.add_argument("--label", type=str,
            help="Only search for this label in output probability of the model"
            "useful when you set the --classification_threshold threshold and you want"
            "consider only one of the labels such as tumor")

    parser.add_argument("--patch_size", type=int,
            default=1024, required=True,
            help="Patch size in pixels to extract from slide to use in evaluation.")

    parser.add_argument("--resize_sizes", nargs='+', type=int,
            help="List of patch sizes in pixels to resize the extracted patchs and save. "
            "Each size should be at most patch_size. Default does not resize.")

    parser.add_argument("--evaluation_size", type=int,
            help="The size in pixel to resize patch before passing to model for evaluation. "
            "evaluation_size should be one of resize_sizes or set to patch_size. "
            "Default uses patch of patch_size for evaluation.")

    parser.add_argument("--is_tumor", action='store_true',
            help="Only extract tumor patches. Default extracts tumor and normal patches.")

    parser.add_argument("--num_patch_workers", type=int, default=0,
            help="Number of loader worker processes to multi-process data loading. "
            "Default uses single-process data loading.")

    parser.add_argument("--gpu_id", type=int,
            help="The ID of GPU to select. Default uses GPU with the most free memory.")

    parser.add_argument("--num_gpus", type=int, default=1,
            help="The number of GPUs to use. "
            "Default uses a GPU with the most free memory.")

    parser.add_argument("--old_version", action='store_true',
            help="Convert trained model on previous version to the current one")

    parser.add_argument("--slide_idx", type=int,
            help="Select a specif slide from all the slides in that directory (usefull for running multiple jobs).")

    parser.add_argument("--maximum_number_patches", nargs='+', type=subtype_kv,
            action=ParseKVToDictAction, default={},
            help="Caution: when you use this flag the code while shuffles the extracted patches from each slide. "
                 "Space separated words describing subtype=maximum_number_of_extracted_patches pairs for each slide. "
            "Example: if want to extract 500 Tumor, 0 Normal patches and unlimited POLE patches "
            "then the input should be 'TUMOR=500 NORMAL=0 POLE=-1'"
                 "You need to pass the class names in uppercase")

    parser.add_argument("--use_radius", action='store_true',
            help="Activating this subparser will enable extracting "
            "all patches within radius of the coordinate.")

    parser.add_argument("--radius", type=int, default=1,
            help="From each selected coordinate, all its neighbours will be extracted. "
            "This number will be multiplied by the patch size."
            "Note: In use-annotation, the number will be multiplied*stride.")

    parser.add_argument('--use_color_norm', action='store_true',
        help="""Whether use normlization of patches before feeding to the model or not.""")

    parser.add_argument('--method', type=str, required=False, default='vahadane',
        help="""The Normalization method.""")

    parser.add_argument('--reference_image', nargs="+", type=file_path,
        help="""The path to reference image(s) for normalization.""")

    parser.add_argument('--use_standarizer', action='store_true',
        help="""Whether to apply brighness standarizer on the images.""")

    parser.add_argument("--seed", type=int,
                                default=DEAFULT_SEED,
                                help="Seed for random library.")

    help_subparsers_load = """Specify how to load slides to annotate.
    There are 2 ways: by manifest and by directory."""
    subparsers_load = parser.add_subparsers(dest='load_method',
            required=True,
            parser_class=AIMArgumentParser,
            help=help_subparsers_load)

    help_manifest = """Use manifest file to locate slides.
        a CSV file with minimum of 4 column and maximum of 6 columns. The name of columns
        should be among ['origin', 'patient_id', 'slide_id', 'slide_path', 'annotation_path', 'subtype'].
        origin, slide_id, patient_id must be one of the columns."""
    parser_manifest = subparsers_load.add_parser("use-manifest",
            help=help_manifest)
    parser_manifest.add_argument("--manifest_location", type=file_path, required=True,
            help="Path to manifest CSV file.")

    parser_directory = subparsers_load.add_parser("use-directory",
            help="Use a rootdir to locate slidesIt is expected that slide paths "
            "have the structure '/path/to/rootdir/slide_pattern/slide_name.extension' where slide_pattern is usually 'subtype'. Patient IDs are extrapolated from slide_name using known, hardcoded regex.")

    parser_directory.add_argument("--slide_location", type=dir_path, required=True,
            help="Path to root directory containing all of the slides.")
    parser_directory.add_argument("--slide_pattern", type=str,
                default='', required=False,
                help="'/' separated words describing the directory structure of the "
                "slide paths. Normally slides paths look like "
                "/path/to/slide/rootdir/subtype/slide.svs and if slide paths are "
                "/path/to/slide/rootdir/slide.svs then simply pass ''.")
    parser_directory.add_argument("--mask_location", type=dir_path, required=False,
            help="Path to root directory which contains mask for tissue selection. "
            "It should contain png files or annotation file with label clear_area.")


