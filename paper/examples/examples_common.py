"""
Some methods common to all examples.

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""

import sys
import os

# The examples directory
example_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '.'))
project_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../..'))
demo_dir = os.path.join(project_dir, 'demos')
sys.path.insert(0, demo_dir)

# And load everything from the common.py of the scripts dir
scripts_dir= os.path.join(project_dir, 'scripts')
sys.path.insert(0, scripts_dir)
from common import *
