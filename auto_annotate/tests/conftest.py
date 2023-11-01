import os
import os.path
import shutil
import pytest

from auto_annotate.tests import (OUTPUT_LOG_DIR, OUTPUT_PATCH_DIR, OUTPUT_DIR)

CLEAN_AFTER_RUN=False

@pytest.fixture
def clean_output():
    """Get the directory to save test outputs. Cleans the output directory before and after each test.
    """
    if os.path.isdir(OUTPUT_PATCH_DIR):
        shutil.rmtree(OUTPUT_PATCH_DIR)
    if os.path.isdir(OUTPUT_LOG_DIR):
        shutil.rmtree(OUTPUT_LOG_DIR)
    for file in os.listdir(OUTPUT_DIR):
        os.unlink(os.path.join(OUTPUT_DIR, file))
    os.mkdir(OUTPUT_PATCH_DIR)
    yield None
    if CLEAN_AFTER_RUN:
        if os.path.isdir(OUTPUT_PATCH_DIR):
            shutil.rmtree(OUTPUT_PATCH_DIR)
        if os.path.isdir(OUTPUT_LOG_DIR):
            shutil.rmtree(OUTPUT_LOG_DIR)
        for file in os.listdir(OUTPUT_DIR):
            os.unlink(os.path.join(OUTPUT_DIR, file))