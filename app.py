import logging

from submodule_utils.logging import logger_factory
from auto_annotate.parser import create_parser
from auto_annotate import *

logger_factory()
logger = logging.getLogger('auto_annotate')

if __name__ == "__main__":
    parser = create_parser()
    config = parser.get_args()
    app = AutoAnnotator.from_log_file(config)
    app.run()
