# Copyright 2025 Huawei Technologies Co., Ltd
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
"""parallel dimensions"""

import os
import copy
import json
import yaml
import toml

from paradise.logger import logger


class YamlObject:
    """Attributed dictionary"""

    def from_dict(self, field):
        """init from dictionary"""
        for k in field:
            if isinstance(field[k], dict):
                setattr(self, k, YamlObject(field[k]))
            else:
                setattr(self, k, field[k])

    def __init__(self, field):
        self.from_dict(field)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            logger.warning(
                "Attribute %s does not exist. Value '0' will be assigned.",
                str(attr),
            )
            return 0
        return self.__dict__[attr]

    def __copy__(self):
        cls = self.__class__
        res = cls.__new__(cls)
        res.__dict__.update(self.__dict__)
        return res

    def __deepcopy__(self, memo):
        cls = self.__class__
        res = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(res, k, copy.deepcopy(v, memo))
        return res

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, field):
        return self.from_dict(field)

    def to_dict(self):
        """Transform from Config to a strict dict class"""
        return_dict = {}
        for key, val in self.__dict__.items():
            if isinstance(val, YamlObject):
                val = val.to_dict()
            return_dict[key] = val
        return return_dict

    def dump(self, file_name, folder=None):
        """Dump to given file in given folder"""
        if not folder:
            folder = os.path.dirname(__file__)
        full_file_name = os.path.join(folder, "config_" + file_name) + ".yaml"
        with open(full_file_name, "w", encoding="utf-8") as outfile:
            yaml.dump(self.to_dict(), outfile, default_flow_style=False)


class Config(YamlObject):
    """Yaml config"""

    def __init__(self, input_config):
        if isinstance(input_config, str):
            with open(input_config, encoding="utf-8") as f:
                if input_config.endswith("yaml"):
                    try:
                        super().__init__(yaml.safe_load(f))
                    except yaml.YAMLError as exc:
                        print(exc)
                        logger.warning(exc)
                elif input_config.endswith("json"):
                    super().__init__(json.load(f))
                elif input_config.endswith("toml"):
                    super().__init__(toml.load(f))
                else:
                    logger.warning("Current handled file formats: YAML, JSON")
        elif isinstance(input_config, Config):
            super().__init__(input_config.__dict__)
        elif isinstance(input_config, dict):
            super().__init__(input_config)
        else:
            raise TypeError("Expecting path string or Config object")

    def __str__(self):
        return str(
            {
                k: (v if not isinstance(v, Config) else vars(v))
                for k, v in vars(self).items()
            }
        )

    # def global_batch_size(self):
    #     """Compute the global batch size"""
    #     pp = Dim.PP.from_config(self)
    #     dp = Dim.DP.from_config(self)
    #     mbs = Dim.MBS.from_config(self)
    #     gbs = dp * mbs
    #     if pp == 1:
    #         logger.debug("global_batch_size = DP(%d) * MBS(%d)", dp, mbs)
    #         # gas = Dim.GAS.from_config(self)
    #         # gbs *= gas
    #     else:
    #         mbn = Dim.MBN.from_config(self)
    #         gbs *= mbn
    #         logger.debug(
    #             "global_batch_size = DP(%d) * MB(%d) * MBS(%d)", dp, mbn, mbs
    #         )
    #     return gbs

    def set_par(self, parallel_dimensions):
        """Set the given parallelization"""
        for dim in parallel_dimensions.dims_val:
            dim.to_config(self, parallel_dimensions.val(dim))

    # def layer_num(self):
    #     """Get the layer number"""
    #     layers = self.model.model_config.num_layers
    #     if layers and layers is not None:
    #         return layers
    #     return self.model.model_config.num_hidden_layers

    # def name(self):
    #     """Get the model name"""
    #     return self.trainer.model_name
