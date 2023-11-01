# Docker Auto Annotate

## Usage

```
usage: app.py [-h] --log_file_location LOG_FILE_LOCATION --log_dir_location
              LOG_DIR_LOCATION [--patch_location PATCH_LOCATION]
              --slide_location SLIDE_LOCATION [--slide_pattern SLIDE_PATTERN]
              --patch_size PATCH_SIZE
              [--resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]]
              [--evaluation_size EVALUATION_SIZE] [--is_tumor]
              [--num_patch_workers NUM_PATCH_WORKERS] [--gpu_id GPU_ID]

optional arguments:
  -h, --help            show this help message and exit

  --log_file_location LOG_FILE_LOCATION
                        Path to the log file produced during training.
                         (default: None)

  --log_dir_location LOG_DIR_LOCATION
                        Path to log directory to save testing logs (i.e. /path/to/logs/testing/).
                         (default: None)

  --patch_location PATCH_LOCATION
                        Path to root directory containing dataset patches specified in group or split file (i.e. /path/to/patch/rootdir/). Used by Docker to link the directory.
                         (default: None)

  --slide_location SLIDE_LOCATION
                        Path to root directory containing all of the slides.
                         (default: None)

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
```
