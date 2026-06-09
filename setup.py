#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""setup package."""
import sys
import logging
import os
import shutil
import stat
import platform
import subprocess
from importlib import import_module
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py
from setuptools.command.install import install

ROOT_DIR = os.path.dirname(__file__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def get_readme_content():
    """Read and return the contents of README.md for use as the package long description."""
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, 'README.md'), encoding='UTF-8') as f:
        return f.read()


def get_platform():
    """
    Get platform name.

    Returns:
        str, platform name in lowercase.
    """
    return platform.system().strip().lower()


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    os_info = get_platform()
    cpu_info = platform.machine().strip()

    return f'hyper_parallel platform: {os_info}, cpu: {cpu_info}'


def get_install_requires():
    """
    Get install requirements.

    Returns:
        list, list of dependent packages.
    """
    with open('requirements.txt', encoding='utf-8') as file:
        return file.read().strip().splitlines()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IEXEC | stat.S_IWRITE)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD | stat.S_IWRITE)


def write_commit_id():
    """Write the current git branch name and latest commit hash to the .commit_id file."""
    ret_code = os.system("git rev-parse --abbrev-ref HEAD > ./hyper_parallel/.commit_id "
                         "&& git log --abbrev-commit -1 >> ./hyper_parallel/.commit_id")
    if ret_code != 0:
        sys.stdout.write(
            "Warning: Can not get commit id information. Please make sure git is available.")
        os.system(
            "echo 'git is not available while building.' > ./hyper_parallel/.commit_id")


class EggInfo(egg_info):
    """Egg info."""

    def run(self):
        egg_info_dir = os.path.join(os.path.dirname(
            __file__), 'hyper_parallel.egg-info')
        shutil.rmtree(egg_info_dir, ignore_errors=True)
        super().run()
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """Build py files."""

    def run(self):
        hyper_parallel_lib_dir = os.path.join(
            os.path.dirname(__file__), 'build', 'lib', 'hyper_parallel')
        shutil.rmtree(hyper_parallel_lib_dir, ignore_errors=True)
        self._run_shell_script_optional("scripts/build_symmetric_memory.sh")
        self._run_shell_script_optional("scripts/build_multicore.sh")
        self._run_shell_script_optional("scripts/build_custom_ops.sh")
        super().run()
        update_permissions(hyper_parallel_lib_dir)

    def _run_shell_script(self, script_path, args=None, capture_output=False):
        """Execute specified shell script with error handling"""
        if args is None:
            args = []

        if not os.path.exists(script_path):
            error_msg = f"Warning: Script not found: {script_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        cmd = ["bash", script_path] + args
        logger.info("Executing: %s", ' '.join(cmd))
        try:
            result = subprocess.run(cmd, check=True,capture_output=capture_output, text=True)
            if result.stdout:
                logger.info("Success: %s", result.stdout)
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to execute script: {script_path}, error: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _run_shell_script_optional(self, script_path, args=None):
        """Execute shell script; log a warning on failure instead of raising."""
        try:
            self._run_shell_script(script_path, args=args)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("Optional build step skipped (%s): %s", script_path, e)


class Install(install):
    """Install."""

    def run(self):
        super().run()
        if sys.argv[-1] == 'install':
            pip = import_module('pip')
            hyper_parallel_dir = os.path.join(
                os.path.dirname(pip.__path__[0]), 'hyper_parallel')
            update_permissions(hyper_parallel_dir)


if __name__ == '__main__':
    version_info = sys.version_info
    if (version_info.major, version_info.minor) < (3, 7):
        sys.stderr.write('Python version should be at least 3.7\r\n')
        sys.exit(1)

    write_commit_id()

    setup(
        name='hyper_parallel',
        version='0.1.0',
        author='The MindSpore Authors',
        author_email='contact@mindspore.cn',
        url='https://www.mindspore.cn',
        download_url='https://gitcode.com/mindspore/hyper-parallel/tags',
        project_urls={
            'Sources': 'https://gitcode.com/mindspore/hyper-parallel',
            'Issue Tracker': 'https://gitcode.com/mindspore/hyper-parallel/issues',
        },
        description=get_description(),
        long_description=get_readme_content(),
        long_description_content_type="text/markdown",
        test_suite="tests",
        packages=find_packages(exclude=["*tests*",
                                        "hyper_parallel.auto_parallel.fast-tuner",
                                        "hyper_parallel.auto_parallel.fast-tuner.*"]),
        platforms=[get_platform()],
        include_package_data=True,
        package_data={
            'hyper_parallel': ['.commit_id',
                       'lib/*.so',
                       'lib/*/*.so'],
            'hyper_parallel.core.shard.ops': ['yaml/*.yaml'],
            'hyper_parallel.platform.torch.symmetric_memory': ['*.so'],
            'hyper_parallel.platform.mindspore.symmetric_memory': ['aclshmem_ms/*.so'],
            'hyper_parallel.platform.mindspore.custom_ops': [
                'build/lib/*.so',
            ],
            'hyper_parallel.core.multicore.platform.mindspore': [
                'build/lib/*.so',
                'build/lib/*_auto_generate/*.py',
            ],
            'hyper_parallel.core.multicore.platform.torch': [
                '*.so',
            ],
            'hyper_parallel.auto_parallel.sapp_nd.memory_estimation': [
                'configs_eval/default.yaml',
            ],
            'hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers': [
                'mapping.yaml',
            ],
        },
        cmdclass={
            'egg_info': EggInfo,
            'build_py': BuildPy,
            'install': Install,
        },
        python_requires='>=3.7',
        install_requires=get_install_requires(),
        setup_requires=['ninja >= 1.10.0'],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: Web Environment',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='hyper_parallel',
    )
