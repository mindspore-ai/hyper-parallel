#!/usr/bin/env python3
# encoding: utf-8
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
"""setup package."""
import sys
import os
import shutil
import stat
import platform
import warnings
import subprocess
from pathlib import Path
from importlib import import_module
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build import build
from setuptools.command.build_py import build_py
from setuptools.command.install import install


def get_readme_content():
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
        super().run()
        update_permissions(hyper_parallel_lib_dir)


class BuildHyperParallelMindSpore(build_py):
    """Custom build command: compile C++ plugin during build phase with version safety"""

    PROJECT_ROOT = Path(__file__).parent.resolve()
    MS_CUSTOM_PASS_DIR = PROJECT_ROOT / "hyper_parallel" / \
        "platform" / "mindspore" / "custom_pass"
    MS_PLUGIN_BUILD_DIR = MS_CUSTOM_PASS_DIR / "build"
    MS_SO_NAME = "libhyper_parallel_mindspore.so"
    MS_SO_DEST = PROJECT_ROOT / "hyper_parallel" / \
        "platform" / "mindspore" / "custom_pass" / MS_SO_NAME

    def run(self):
        try:
            self._build_plugin()

            # Check if SO was actually generated (version skip exits 0 but produces no SO)
            if self.MS_SO_DEST.exists():
                print(f"✓ Mindspore custom pass compiled: {self.MS_SO_DEST}")
            else:
                print(
                    "⚠️ Mindspore custom pass compilation skipped (version requirement or no MindSpore detected)")

        except Exception as e:
            # Only warn if user explicitly requested MindSpore support
            if "MINDSPORE_ROOT" in os.environ:
                warnings.warn(
                    f"⚠️  Build failed despite MINDSPORE_ROOT being set ({os.environ['MINDSPORE_ROOT']}):\n{e}\n"
                    "   Please verify MindSpore development files are installed correctly.",
                    UserWarning
                )
            else:
                warnings.warn(
                    "⚠️  MindSpore custom pass compilation skipped.\n"
                    "   Set MINDSPORE_ROOT to enable graph optimizations for MindSpore backend.\n"
                    "   Example: export MINDSPORE_ROOT=/path/to/mindspore-2.8.1",
                    UserWarning
                )

    def _build_plugin(self):
        """Invoke build_plugin.py to compile the plugin"""
        if not self.MS_CUSTOM_PASS_DIR.exists():
            raise FileNotFoundError(
                f"Mindspore custom pass directory not found: {self.MS_CUSTOM_PASS_DIR}")

        # Clean previous build artifacts
        if self.MS_PLUGIN_BUILD_DIR.exists():
            shutil.rmtree(self.MS_PLUGIN_BUILD_DIR)
        self.MS_PLUGIN_BUILD_DIR.mkdir(parents=True, exist_ok=True)

        # Pass environment variables to build script
        env = os.environ.copy()
        env["MS_PLUGIN_BUILD_DIR"] = str(self.MS_PLUGIN_BUILD_DIR)
        env["MS_SO_OUTPUT"] = str(self.MS_SO_DEST)

        if "MINDSPORE_ROOT" in os.environ:
            env["MINDSPORE_ROOT"] = os.environ["MINDSPORE_ROOT"]
            print(
                f"Using user-specified MINDSPORE_ROOT: {env['MINDSPORE_ROOT']}")

        # Execute build script
        result = subprocess.run(
            [sys.executable, str(self.MS_CUSTOM_PASS_DIR / "build_plugin.py")],
            cwd=str(self.MS_CUSTOM_PASS_DIR),
            env=env,
            capture_output=True,
            text=True,
            check=False
        )

        # build_plugin.py exits 0 on version skip (successful skip), non-0 on actual failure
        if result.returncode != 0:
            print("Build stdout:", result.stdout)
            print("Build stderr:", result.stderr)
            raise RuntimeError(f"Plugin build failed:\n{result.stderr}")

        # If exit 0 but no SO exists, it was a version-based skip - handled in run()
        if not self.MS_SO_DEST.exists():
            raise FileNotFoundError(
                f"Expected SO file not found: {self.MS_SO_DEST}")


class Build(build):
    """
    Orchestrator that conditionally registers backend-specific sub-commands.
    Keeps main build flow pure while enabling optional features.
    """

    # Register sub-commands that run BEFORE standard build steps
    # Format: (command_name, should_run_callable)
    sub_commands = [
        ('build_ms_plugin', lambda self: True),
    ] + build.sub_commands


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
            'hyper_parallel': ['.commit_id'],
            'hyper_parallel.core.shard.ops': ['yaml/*.yaml'],
            "hyper_parallel.platform.mindspore.custom_pass": [BuildHyperParallelMindSpore.MS_SO_NAME],
        },
        cmdclass={
            'egg_info': EggInfo,
            'build': Build,
            'build_py': BuildPy,
            'build_ms_plugin': BuildHyperParallelMindSpore,
            'install': Install,
        },
        python_requires='>=3.7',
        install_requires=get_install_requires(),
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
