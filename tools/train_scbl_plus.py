"""向后兼容入口：本脚本已与 ``tools/train_bl+.py`` 合并，内容为空壳。

历史背景
--------
这两个脚本原先各有一份 290 行的 ``Trainer`` 副本，唯一实质差异是
``build_train_loader`` 里是否引用 ``FCPoseDatasetMapper``（该模块在本仓库中
并不存在）。把路径、环境变量、数据集注册抽到 ``base.py`` /
``register_datasets.py`` 之后两者已逐行等价，故只保留一份实现。

两者对应的实验差异**全部在 yaml 里**：本脚本用于
``configs/run-SCBlendMask-plus.yaml``（Lcspvig + LSK 骨干），
``train_bl+.py`` 用于其余配置。骨干、basis 模块、注意力类型都是配置项，
与用哪个脚本无关——所以下面两条命令完全等价：::

    python tools/train_scbl_plus.py --config-file configs/run-SCBlendMask-plus.yaml
    python tools/train_bl+.py      --config-file configs/run-SCBlendMask-plus.yaml

推荐直接使用后者，本文件仅为兼容既有命令与 README 示例而保留。
"""
import importlib.util
import os

_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_bl+.py")

# 文件名含 '+'，不是合法 Python 标识符，不能用普通 import
_spec = importlib.util.spec_from_file_location("_gbadmask_train_impl", _IMPL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

main = _mod.main
cli = _mod.cli

if __name__ == "__main__":
    cli()
