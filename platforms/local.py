# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

from platforms import base

project_root_path =  os.path.join(pathlib.Path.home(), 'Documents/JKU/Seminar/Practical_Work')

class Platform(base.Platform):
    @property
    def root(self):
        return os.path.join(project_root_path, 'open_lth_data')

    @property
    def dataset_root(self):
        return os.path.join(project_root_path, 'open_lth_datasets')

    @property
    def imagenet_root(self):
        raise NotImplementedError
