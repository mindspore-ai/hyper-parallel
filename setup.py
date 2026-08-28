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
from importlib import import_module
from setuptools import setup, find_packages, Distribution
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py
from setuptools.command.install import install

ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

TORCH26_REQUIRES = [
    "torch==2.6.0",
    "torch-npu==2.6.0.post3",
]

TORCH27_REQUIRES = [
    "torch==2.7.1",
    "torch-npu==2.7.1",
]

TORCH29_REQUIRES = [
    "torch==2.9.1",
    "torch-npu==2.9.1",
]

MINDSPORE_REQUIRES = [
    "mindspore>=2.10",
]

def _read_requirements(requirements_path: str) -> list[str]:
    """Read Python requirement lines from a repository-local file."""
    with open(os.path.join(ROOT_DIR, requirements_path), encoding='utf-8') as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]


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
    return f"{platform.system().strip().lower()}_{platform.machine().strip().lower()}"


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    os_info = get_platform()
    cpu_info = platform.machine().strip()

    return f'hyper_parallel platform: {os_info}, cpu: {cpu_info}'


def get_install_requires() -> list[str]:
    """
    Get install requirements.

    Returns:
        list, list of dependent packages.
    """
    return _read_requirements('requirements.txt')


def get_extra_requires() -> dict[str, list[str]]:
    """
    Get optional framework requirements.

    Returns:
        dict, optional dependency groups keyed by extra name.
    """
    return {
        "torch": list(TORCH29_REQUIRES),
        "torch26": list(TORCH26_REQUIRES),
        "torch27": list(TORCH27_REQUIRES),
        "torch29": list(TORCH29_REQUIRES),
        "mindspore": list(MINDSPORE_REQUIRES),
        "all": TORCH29_REQUIRES + MINDSPORE_REQUIRES,
    }


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
        """Build Python sources and copy the explicitly prepared native payload."""
        shutil.rmtree(self.build_lib, ignore_errors=True)
        super().run()
        target_lib_dir = os.path.join(self.build_lib, 'hyper_parallel')
        native_payload = os.environ.get("HYPER_PARALLEL_NATIVE_OUTPUT_ROOT", "").strip()
        if native_payload:
            if not os.path.isdir(native_payload):
                logger.warning(
                    "[HP-NATIVE-PAYLOAD-MISSING] wheel will contain only successfully built Python sources: %s",
                    native_payload,
                )
            else:
                shutil.copytree(native_payload, target_lib_dir, dirs_exist_ok=True)
                logger.info("Copied optional native payload from %s", native_payload)
        else:
            logger.info("No native payload selected; assembling a core-only wheel.")
        update_permissions(target_lib_dir)


class Install(install):
    """Install."""

    def run(self):
        super().run()
        if sys.argv[-1] == 'install':
            pip = import_module('pip')
            hyper_parallel_dir = os.path.join(
                os.path.dirname(pip.__path__[0]), 'hyper_parallel')
            update_permissions(hyper_parallel_dir)


class BinaryDistribution(Distribution):
    """Force wheel to be tagged as a binary distribution.

    The package ships pre-compiled .so files (built by scripts/build_*.sh and
    bundled via package_data). Without this hint setuptools/wheel would label
    the wheel as pure-python (py3-none-any), which lets pip install it under
    incompatible Python versions or CPU architectures and triggers
    'Python version mismatch' at import time.
    """

    def has_ext_modules(self) -> bool:
        """Return True so wheel is tagged for a specific cpython + platform."""
        return True


if __name__ == '__main__':
    _cmdclass = {
        'egg_info': EggInfo,
        'build_py': BuildPy,
        'install': Install,
    }
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
        scripts=['scripts/hyper_parallel_multicore_set_env.bash'],
        package_data={
            'hyper_parallel.core.shard.ops': ['yaml/*.yaml'],
            'hyper_parallel.platform.torch.custom_ops.gdn': ['LICENSE'],
            'hyper_parallel.platform.mindspore.custom_ops': ['lib/*.so', 'lib/*_auto_generate/*.py'],
            'hyper_parallel.core.multicore': ['lib/**/*'],
            'hyper_parallel.core.symmetric_memory': ['lib/**/*'],
            'hyper_parallel.auto_models.components.datasets.llm': ['*.so'],
            'hyper_parallel.auto_parallel.sapp_nd.memory_estimation': [
                'configs_eval/default.yaml',
            ],
            'hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers': [
                'mapping.yaml',
            ],
        },
        cmdclass=_cmdclass,
        distclass=BinaryDistribution,
        python_requires='>=3.10,<3.13',
        install_requires=get_install_requires(),
        extras_require=get_extra_requires(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: Web Environment',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='hyper_parallel',
    )
