#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import os
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "adet", "layers", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "adet._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="AdelaiDet",
    version="0.2.0",
    author="Adelaide Intelligent Machines",
    url="https://github.com/stanstarks/AdelaiDet",
    description="AdelaiDet is AIM's research "
    "platform for instance-level detection tasks based on Detectron2.",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.7",
    install_requires=[
        "termcolor>=1.1",
        # detectron2 v0.6 用到 PIL.Image.LINEAR，Pillow 10+ 已移除该常量
        "Pillow>=6.0,<10.0",
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",
        "rapidfuzz",
        "shapely",
        # 本项目 backbone（cspvig / Lcspvig）依赖 timm 的 DropPath。
        # timm 1.0 起移除了 timm.models.layers.ConvBnAct 等 API，必须锁 <1.0
        "timm>=0.6.12,<1.0",
        # detectron2 v0.6 的 config 依赖
        "omegaconf",
        # BAText 的语法解析依赖，需与 omegaconf 的传递依赖保持一致
        "antlr4-python3-runtime==4.8",
    ],
    extras_require={
        "all": ["psutil"],
        # onnx/ 目录下的导出与推理脚本所需
        "onnx": ["onnx", "onnxruntime"],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
