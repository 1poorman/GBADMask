from detectron2.modeling import META_ARCH_REGISTRY

from .blendmask import BlendMask
from .basis_module2 import build_basis_module2

__all__ = ["BlendMask2", "build_basis_module2"]


# 原实现是 BlendMask 的近乎逐行拷贝，唯一区别是 basis 模块走独立的
# Registry("BASIS_MODULE2")，导致"换 basis 模块"与"换 meta 架构"两个维度被绑死，
# 无法正交消融（例如 ViG + 官方 ProtoNet 就配不出来）。
#
# 现在两份 ProtoNet 都注册在同一个 BASIS_MODULE registry 下
# （"ProtoNet"=官方, "ProtoNetV2"=改进版），由 MODEL.BASIS_MODULE.NAME 选择，
# 因此 BlendMask2 与 BlendMask 行为已完全一致，仅作为旧配置的向后兼容别名保留。
#
# 新配置请直接使用 META_ARCHITECTURE: "BlendMask" + MODEL.BASIS_MODULE.NAME: "ProtoNetV2"。
@META_ARCH_REGISTRY.register()
class BlendMask2(BlendMask):
    """Backward-compatible alias of BlendMask (kept for existing configs).

    Kept because configs such as ``configs/run-SCBlendMask-plus.yaml``
    reference this name. Differences in the basis module are now controlled
    via ``MODEL.BASIS_MODULE.NAME`` instead.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
