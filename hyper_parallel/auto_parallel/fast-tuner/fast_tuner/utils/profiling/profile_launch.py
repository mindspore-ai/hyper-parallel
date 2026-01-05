"""launch profile"""
import subprocess
import os
import re
import time
from pathlib import Path
from fast_tuner.utils.common import GENERAL_TOML
from fast_tuner.utils.logger import logger

class ProfileLaunch:
    """自动执行profile"""
    def __init__(self, profile_configs, para):
        self.profile_configs = profile_configs
        self.para = para

    def profile_launch(self, profile_file_dir):
        """遍历指定目录下的所有文件"""
        for root, _, files in os.walk(profile_file_dir):
            for file in files:
                # if '_EP' in file:
                #     pattern = MOE_PATTERN_REG_SHELL
                # else:
                #     pattern = LLAMA_PATTERN_REG_SHELL
                # match = re.match(pattern, file)
                # if not match:
                #     continue
                profile_file_path = os.path.join(root, file)
                if file.endswith(".toml"):
                    self.run_torchtitan(profile_file_path)
                elif file.endswith(".sh"):
                    self.run_shell(profile_file_path)
                else:
                    logger.error(f"file type: {file} is not supported.")

    def run_shell(self, profile_file_dir):
        """执行megatron shell脚本"""
        cmd = ["bash", profile_file_dir]
        logger.info(f"profile command: {cmd}")

        process = subprocess.run(
            cmd,
            preexec_fn=os.setpgrp,
            check=False,
        )
        # 为避免profile子进程未结束生成profile文件失败，增加sleep
        time.sleep(60)
        return_code = process.returncode
        logger.info("Last job returns %d.", return_code)


    def run_torchtitan(self, profile_file_toml):
        """执行torch titan"""
        run_file = self.para.TORCHTITAN_PATH
        regex_pattern = GENERAL_TOML.replace('{}', r'(\d+)')
        match = re.match(regex_pattern, Path(profile_file_toml).name)
        if match:
            dp = int(match.group(1))
            tp = int(match.group(2))
            pp = int(match.group(3))
            cp = int(match.group(5))
            world_size = dp * tp * pp * cp
        else:
            raise ValueError(f"Invalid profile_file_toml: {profile_file_toml}")
        cmd = f'CONFIG_FILE={profile_file_toml} NGPU={world_size} {run_file}'
        logger.info(f"run_torchtitan profile command: {cmd}")

        process = subprocess.run(
            cmd,
            preexec_fn=os.setpgrp,
            shell=True,
            check=False,
        )
        # 为避免profile子进程未结束生成profile文件失败，增加sleep
        time.sleep(20)
        return_code = process.returncode
        logger.info("Last job returns %d.", return_code)
