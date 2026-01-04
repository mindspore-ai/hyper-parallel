# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Transformer-Config dict parse module """

import os
import yaml

class InputConfig(dict):
    """
    A class for configuration that inherits from Python's dict class.
    Can parse configuration parameters from yaml files

    Args:
        args (Any): Extensible parameter list, a yaml configuration file path
        kwargs (Any): Extensible parameter dictionary, a yaml configuration file path

    Returns:
        An instance of the class.

    Examples:
        >>> cfg = InputConfig('./test.yaml')
        >>> cfg.a
    """

    def __init__(self, *args):
        super().__init__()
        cfg_dict = {}

        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('yaml') or arg.endswith('yml'):
                    raw_dict = InputConfig._file2dict(arg)
                    cfg_dict.update(raw_dict)

        InputConfig._dict2config(self, cfg_dict)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    @staticmethod
    def _file2dict(filename=None):
        """Convert config file to dictionary.

        Args:
            filename (str) : config file.
        """
        if filename is None:
            raise NameError(f'This {format(filename)} cannot be empty.')

        filepath = os.path.realpath(filename)
        with open(filepath, encoding='utf-8') as fp:
            # 文件指针重置到文件开头
            fp.seek(0)
            cfg_dict = yaml.safe_load(fp)

        return cfg_dict

    @staticmethod
    def _dict2config(config, dic):
        """Convert dictionary to config.

        Args:
            config : Config object
            dic (dict) : dictionary
        """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = InputConfig()
                    dict.__setitem__(config, key, sub_config)
                    InputConfig._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]
