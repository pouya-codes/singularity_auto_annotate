import logging

from submodule_utils.logging import logger_factory
from auto_annotate.parser import create_parser
from auto_annotate import *

logger_factory()
logger = logging.getLogger('auto_annotate')

if __name__ == "__main__":
    parser = create_parser()
    config = parser.get_args()
    ape = AutoAnnotator(config)
    ape.run()

# if __name__ == "__main__":

#     args = parser.parse_args()
#     aa = AutoAnnotator.from_log_file(
#             args.log_file_location,
#             args.log_dir_location,
#             args.patch_location,
#             args.slide_location,
#             args.slide_pattern,
#             args.patch_size,
#             resize_sizes=args.resize_sizes,
#             evaluation_size=args.evaluation_size,
#             is_tumor=args.is_tumor,
#             num_patch_workers=args.num_patch_workers,
#             gpu_id=args.gpu_id)
#     aa.run()