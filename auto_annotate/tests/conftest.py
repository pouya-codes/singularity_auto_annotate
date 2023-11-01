import os
import os.path
import shutil
import pytest

from auto_annotate.tests import (OUTPUT_LOG_DIR, OUTPUT_PATCH_DIR, OUTPUT_DIR)

@pytest.fixture
def clean_output():
    """Get the directory to save test outputs. Cleans the output directory before and after each test.
    """

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    os.mkdir(OUTPUT_PATCH_DIR)
    os.mkdir(OUTPUT_LOG_DIR)
    yield None
