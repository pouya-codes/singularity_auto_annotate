# Auto Annotate

**Caution: you need to check the error log constantly to monitor _CUDA out of memory_ error. You need to use a smaller number for _--num_patch_workers_ if you get this error.**

To build the singularity image do:

```
singularity build --remote auto_annotate.sif Singularityfile.def
```

In the SH file, you should bind the path to the slides if the slides in your slides directory specified by `--slide_location` is symlinked.

```
singularity run     -B /projects/ovcare/classification/cchen     -B /projects/ovcare/WSI     auto_annotate.sif     from-experiment-manifest /path/to/experiment.yaml ```

### Development Information ###

```
Date Created: 22 July 2020
Last Update: Fri Oct 23 14:54:27 PDT 2020 by pouya
Developer: Colin Chen
Version: 1.0
```

### Usage ###


usage: app.py [-h] {from-experiment-manifest,from-arguments} ...

Use trained model to extract patches.

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

This component looks in the YAML section of the training log to obtain the trained model and it's hyperparameters.

positional arguments:
  {from-experiment-manifest,from-arguments}
                        Choose whether to use arguments from experiment manifest or from commandline
    from-experiment-manifest
                        Use experiment manifest

    from-arguments      Use arguments

optional arguments:
  -h, --help            show this help message and exit

usage: app.py from-experiment-manifest [-h] [--component_id COMPONENT_ID]
                                       experiment_manifest_location

positional arguments:
  experiment_manifest_location

optional arguments:
  -h, --help            show this help message and exit

  --component_id COMPONENT_ID

usage: app.py from-arguments [-h] --log_file_location LOG_FILE_LOCATION
                             --log_dir_location LOG_DIR_LOCATION
                             --slide_location SLIDE_LOCATION
                             [--store_extracted_patches]
                             [--patch_location PATCH_LOCATION]
                             [--generate_heatmap] --heatmap_location
                             HEATMAP_LOCATION
                             [--classification_threshold CLASSIFICATION_THRESHOLD]
                             [--slide_pattern SLIDE_PATTERN] --patch_size
                             PATCH_SIZE
                             [--resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]]
                             [--evaluation_size EVALUATION_SIZE] [--is_tumor]
                             [--num_patch_workers NUM_PATCH_WORKERS]
                             [--gpu_id GPU_ID] [--num_gpus NUM_GPUS]
                             [--maximum_number_patches MAXIMUM_NUMBER_PATCHES [MAXIMUM_NUMBER_PATCHES ...]]

optional arguments:
  -h, --help            show this help message and exit

  --log_file_location LOG_FILE_LOCATION
                        Path to the log file produced during training.
                         (default: None)

  --log_dir_location LOG_DIR_LOCATION
                        Path to log directory to save testing logs (i.e. /path/to/logs/testing/).
                         (default: None)

  --slide_location SLIDE_LOCATION
                        Path to root directory containing all of the slides.
                         (default: None)

  --store_extracted_patches
                        Store extracted patches. Default does not store extracted patches.
                         (default: False)

  --patch_location PATCH_LOCATION
                        Path to root directory to extract patches into.
                         (default: None)

  --generate_heatmap    Generate heatmaps. Default does not generate heatmap.
                         (default: False)

  --heatmap_location HEATMAP_LOCATION
                        Path to directory to save the heatmap H5 files (i.e. /path/to/heatmaps/).
                         (default: None)

  --classification_threshold CLASSIFICATION_THRESHOLD
                        Minimum obtained probability for the most probable class.Default: 0
                         (default: 0)

  --slide_pattern SLIDE_PATTERN
                        '/' separated words describing the directory structure of the slide paths. Normally slides paths look like /path/to/slide/rootdir/subtype/slide.svs and if slide paths are /path/to/slide/rootdir/slide.svs then simply pass ''.
                         (default: subtype)

  --patch_size PATCH_SIZE
                        Patch size in pixels to extract from slide to use in evaluation.
                         (default: 1024)

  --resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]
                        List of patch sizes in pixels to resize the extracted patchs and save. Each size should be at most patch_size. Default does not resize.
                         (default: None)

  --evaluation_size EVALUATION_SIZE
                        The size in pixel to resize patch before passing to model for evaluation. evaluation_size should be one of resize_sizes or set to patch_size. Default uses patch of patch_size for evaluation.
                         (default: None)

  --is_tumor            Only extract tumor patches. Default extracts tumor and normal patches.
                         (default: False)

  --num_patch_workers NUM_PATCH_WORKERS
                        Number of loader worker processes to multi-process data loading. Default uses single-process data loading.
                         (default: 0)

  --gpu_id GPU_ID       The ID of GPU to select. Default uses GPU with the most free memory.
                         (default: None)

  --num_gpus NUM_GPUS   The number of GPUs to use. Default uses a GPU with the most free memory.
                         (default: 1)

  --maximum_number_patches MAXIMUM_NUMBER_PATCHES [MAXIMUM_NUMBER_PATCHES ...]
                        Caution: when you use this flag the code while shuffles the extracted patches from each slide.space separated words describing subtype=maximum_number_of_extracted_patches pairs for each slide. Example: if want to extract 500 Tumor, 0 Normal patches and unlimited POLE patches then the input should be 'Tumor=500 Normal=0 POLE=-1'
                         (default: {})

```

