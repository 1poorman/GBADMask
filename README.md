<div align="center">
    <img src="docs/adel-logo.svg" width="160" >
</div>

# GBADMask

GBADMask 是在 [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) 的 **BlendMask** 实现基础上做的二次开发，目标场景是**植株 / 草莓器官的实例分割**。

核心思路：把 BlendMask 的 bottom-up 基（bases）生成模块与 BiFPN 结合，并引入 **视觉图卷积骨干（Vision GNN / MobileViG）** 与 **全局上下文注意力（GCNet）**、**Focal-Dice-CE 混合损失**，以在保持单阶段实例分割速度的同时提升密集小目标的掩膜质量。

> **上游**：AdelaiDet v0.2.0（BlendMask）+ Detectron2 v0.6
> **主要改动目录**：`adet/modeling/blendmask/`、`adet/modeling/backbone/`

---

## 目录

- [1. 改动总览](#1-改动总览)
- [2. Backbone 侧改动](#2-backbone-侧改动)
- [3. BlendMask 侧改动](#3-blendmask-侧改动)
- [4. 配置说明](#4-配置说明)
- [5. 环境安装](#5-环境安装)
- [6. 数据准备](#6-数据准备)
- [7. 训练 / 评估 / 推理](#7-训练--评估--推理)
- [8. 已知问题](#8-已知问题)
- [9. 优化建议](#9-优化建议)
- [10. 致谢与引用](#10-致谢与引用)

---

## 1. 改动总览

```
adet/modeling/
├── backbone/
│   ├── cspvig.py            ★ 新增  Vision GNN（MobileViG）图卷积骨干
│   ├── Lcspvig.py           ★ 新增  cspvig 变体 + LSKblock 大核选择注意力
│   ├── bifpn.py             ▲ 修改  新增 cspvig / Lcspvig 两个 BiFPN builder
│   ├── __init__.py          ▲ 修改  导出新增的 BiFPN builder
│   └── fpn.py dla.py vovnet.py mobilenet.py resnet_lpf.py resnet_interval.py lpf.py   （官方原样）
└── blendmask/
    ├── basis_module2.py     ★ 新增  低层特征 + GC 全局上下文融合的 ProtoNet
    ├── fdc_loss.py          ★ 新增  Focal-Dice-CrossEntropy 混合分割损失
    ├── GCblock.py           ★ 新增  GCNet 全局上下文块
    ├── cbam.py              ★ 新增  CBAM 注意力（未接入）
    ├── ca.py                ★ 新增  Coordinate Attention（未接入）
    ├── dice_loss.py         ★ 新增  nnUNet 系列分割损失（未接入）
    ├── ND_Crossentropy.py   ★ 新增  nnUNet ND-CE / TopK / WCE（未接入）
    ├── blendmask2.py        ★ 新增  BlendMask1（当前不可运行）
    ├── build.py             ★ 新增  自建 registry（死代码）
    ├── torch-stat.py        ★ 新增  torchstat 统计脚本
    └── basis_module.py blender.py blendmask.py                                        （官方原样）
```

★ = 本项目新增 ；▲ = 在官方文件上修改 ；无标记 = 与官方一致

> 注意：本项目的改动**没有新增 `adet/config/defaults.py` 节点**，全部复用 Detectron2 / AdelaiDet 已有的
> `MODEL.BASIS_MODULE.*`、`MODEL.BiFPN.*`、`MODEL.RESNETS.OUT_FEATURES` 等字段。

---

## 2. Backbone 侧改动

### 2.1 `cspvig.py` —— Vision GNN 骨干（新增）

参考 *Vision GNN (ViG, arXiv:2206.00272)* 与 *MobileViG*，把图像当作图结构来建模：

| 组件 | 说明 |
| --- | --- |
| `Stem` | 两级 3×3 / stride2 卷积，BN + GELU，4 倍下采样 |
| `InvertedResidual` | 局部建模：`MLP(1×1 → DW 3×3 → 1×1)`，`mlp_ratio=4`，带 `DropPath` 与 layer scale |
| `MRConv4d` | **Max-Relative Graph Conv**：按 `K=2` 的步长把特征与其上下 / 左右邻居作差取 `max`，得到"最大相对"图信息，拼接后 1×1 融合 |
| `Grapher` | `fc1 → MRConv4d → fc2` + 残差，图卷积单元 |
| `FFN` | 1×1 升维 → GELU → 1×1 降维 + 残差 |
| `Downsample` | 3×3 / stride2 卷积下采样 |

**CSP 式分半结构**：每个 stage 先 `x.chunk(2, dim=1)`，只让右半 `xb` 过 block 堆叠，再与左半 `xs` 拼接后过 `transition`，在几乎不增加计算量的前提下保留跨层信息流（这也是目录名中 "csp" 的由来）。

输出 4 个尺度（与 ResNet 命名对齐，便于直接挂 BiFPN）：

| 特征名 | 通道 | stride |
| --- | --- | --- |
| `res2` | 42 | 4 |
| `res3` | 84 | 8 |
| `res4` | 224 | 16 |
| `res5` | 400 | 32 |

注册的 builder（`BACKBONE_REGISTRY`）：

| builder | 规模 |
| --- | --- |
| `build_cspvigm_backbone` | `local_blocks=[3,3,9]`、`local_channels=[42,84,224]`、`global_blocks=[3]`、`global_channels=[400]` |
| `build_cspvigm_fpn_backbone` | 上述骨干 + 标准 FPN（`LastLevelMaxPool`） |
| `build_cspvigb_backbone` | 更大规模：`local_blocks=[5,5,15]`、`local_channels=[42,84,240]`、`global_blocks=[5]`、`global_channels=[464]` |

### 2.2 `Lcspvig.py` —— 大核选择注意力变体（新增）

在 `cspvig.py` 的基础上额外实现：

- `LSKblock`（LSKNet 风格）：`DW 5×5` → `DW 7×7(dilation=3)` 两条大核分支，各自 1×1 降维后按通道做 **avg / max 拼接 → 7×7 → sigmoid** 得到空间选择权重，加权融合后 1×1 还原，最后 `x * attn`。
- `BN_LSKb_act`：把 `LSKblock` 嵌进 `Conv-BN-Hardswish`。

> ⚠️ **当前 `BN_LSKb_act` / `LSKblock` 并没有被 `MobileViG.forward` 调用**，也就是说这个文件的"L"（Large-kernel）部分实际未生效，
> `Lcspvig` 目前与 `cspvig` 行为等价。详见 [第 8 节](#8-已知问题)。

### 2.3 `bifpn.py` —— 接入 ViG 骨干（修改）

在官方 `build_fcos_resnet_bifpn_backbone` 之后新增两个 builder，把 BiFPN 的 `bottom_up` 从 ResNet 换成 ViG：

```python
@BACKBONE_REGISTRY.register()
def build_fcos_cspvig_bifpn_backbone(cfg, input_shape): ...   # MobileViG + BiFPN
@BACKBONE_REGISTRY.register()
def build_fcos_Lcspvig_bifpn_backbone(cfg, input_shape): ...  # Lcspvig + BiFPN
```

BiFPN 部分（`SingleBiFPN` / `BiFPN` / `BackboneWithTopLevels`）沿用官方实现，未改动：
可学习的 fast-normalized 融合权重 + `swish` 激活 + 深度可分离 3×3，重复 `MODEL.BiFPN.NUM_REPEATS` 次。

---

## 3. BlendMask 侧改动

### 3.1 `basis_module2.py` —— 改进版 ProtoNet（新增，核心）

官方 ProtoNet（`basis_module.py`）的流程是：多尺度特征各自 3×3 卷积到 `CONVS_DIM` → 双线性相加 → tower → ×2 上采样 → 1×1 输出 `NUM_BASES` 通道 bases。

本项目在 `basis_module2.ProtoNet` 中做了 4 处改动：

1. **低层细节分支**：取 `in_features[0]`（最高分辨率的 `p3`）经 1×1 卷积降到 **24 通道**；
2. **GC 全局上下文**：这 24 通道特征过 `GlobalContextBlock(24, ratio=1/16)`（GCNet，`att` 型空间池化 + `channel_mul`），抑制背景、强化前景器官区域；
3. **深浅特征融合**：低层 GC 特征上采样后与 tower 输出的深层 bases 特征在通道维 `concat`，再 `conv_block(planes+24, planes, 3, 1)` 降维；
4. **bases 输出头**：tower 尾部的 1×1 被移除，改由新增的 `self.conv2 = 3×3-BN-ReLU-1×1` 输出 bases，多一层非线性。

```python
# basis_module2.py:137-144
fre = self.tower(x)                                   # 深层多尺度融合特征
low_feat = self.conv1(features[self.in_features[0]])  # 低层细节 → 24ch
low_feat = F.interpolate(low_feat, fre.size()[2:], mode="bilinear", align_corners=True)
x = torch.cat((self.gc(low_feat), fre), 1)            # GC 加权低层 + 深层
x = self.concat(x)                                    # (planes+24) → planes
outputs = {"bases": [self.conv2(x)]}
```

此外，tower 的**开头**也多插入了一个 `gcblock(planes, ratio=1/16)`。

### 3.2 `fdc_loss.py` —— Focal-Dice-CE 混合损失（新增）

辅助语义分割损失由官方的 `F.cross_entropy` 换成自研的 `DC_and_CE_loss`：

$$
\log p = 0.8 \cdot CE + 0.2 \cdot \bigl(Dice + 1\bigr), \qquad
\mathcal{L} = w \cdot \bigl(1 - e^{-\log p}\bigr)^{\gamma} \cdot \log p
$$

即在 **CE + SoftDice 的加权和**外面再套一层 focal 调制（默认 `gamma=0.75`），把训练重心压向难分像素，
用于缓解植株/草莓场景中"器官像素极少、背景像素极多"的严重类别不平衡。

配套的 `CrossentropyND`、`SoftDiceLoss`、`softmax_helper`、`get_tp_fp_fn` 移植自 nnUNet。

### 3.3 其余新增文件

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| `GCblock.py` | GCNet `GlobalContextBlock`（`avg` / `att` 两种池化，`channel_add` / `channel_mul` 两种融合） | **已使用**（被 `basis_module2` 调用） |
| `cbam.py` | `CBAMLayer`，通道注意力 + 空间注意力串联 | 仅被 import，未使用 |
| `ca.py` | `CA_Block`，Coordinate Attention（沿 H / W 分别池化编码位置信息） | 未使用 |
| `dice_loss.py` | nnUNet 系列损失：`SoftDiceLoss` / `MemoryEfficientSoftDiceLoss` / `TopKLoss` / `DC_and_topk_loss` 等 | 未使用 |
| `ND_Crossentropy.py` | `CrossentropyND` / `TopKLoss` / `WeightedCrossEntropyLoss` / 距离惩罚 CE | 仅被 `dice_loss.py` 依赖 |
| `blendmask2.py` | `BlendMask1` meta-arch | **不可运行**，见第 8 节 |
| `build.py` | 自建 `META_ARCH_REGISTRY` + `build_model` | 死代码，无任何引用 |
| `torch-stat.py` | 3 行 torchstat FLOPs/参数量统计脚本 | 文件名含 `-`，无法 import，只能直接 `python` 运行 |

---

## 4. 配置说明

`configs/` 下 6 个文件，均为 AdelaiDet 的 yacs 配置（`_BASE_` 继承）：

```
configs/
├── Base-BlendMask.yaml            ResNet + FPN，官方基线（COCO）
├── Base-BlendMask-BiFPN.yaml      ResNet + BiFPN，官方基线（COCO，本项目 run-*.yaml 的公共父配置）
├── run-BlendMask+.yaml            ★ BlendMask  + MobileViG + BiFPN
├── run-vig.yaml                   ★ BlendMask  + MobileViG + BiFPN（Plantv2）
├── run-BlendMask2-vig.yaml        ★ BlendMask2 + MobileViG + BiFPN
└── run-SCBlendMask+ .yaml         ★ BlendMask2 + Lcspvig  + BiFPN
```

**两种 meta 架构的区别**（关键，容易踩坑）：

| `MODEL.META_ARCHITECTURE` | basis 模块 | 语义辅助损失 |
| --- | --- | --- |
| `BlendMask` | `basis_module.ProtoNet`（官方） | `F.cross_entropy` |
| `BlendMask2` | `basis_module2.ProtoNet`（低层特征 + GCNet 融合） | `DC_and_CE_loss`（Focal-Dice-CE） |

因此 `run-vig.yaml` 与 `run-BlendMask2-vig.yaml` 构成一对消融实验——**两者唯一区别就是 meta 架构**，可用来单独量化 basis 模块改动的收益。

> ⚠️ `BlendMask2` 的 Focal-Dice-CE 损失**只在 `BASIS_MODULE.LOSS_ON=True` 时生效**，
> 而 `LOSS_ON=True` 需要数据集额外提供 `thing_train2017/*.npz` 监督。
> 自定义数据集上 `LOSS_ON=False`，此时 `BlendMask2` 相对 `BlendMask` 生效的只有 **GC 特征融合**这一项。

### 4.1 关键字段

| 字段 | 含义 | 备注 |
| --- | --- | --- |
| `MODEL.BACKBONE.NAME` | 骨干 builder 名 | ViG 系列见第 2 节表格 |
| `MODEL.BiFPN.IN_FEATURES` | 送进 BiFPN 的特征层 | ViG 骨干输出 `res2..res5` |
| `MODEL.BiFPN.OUT_CHANNELS` | BiFPN 输出通道 | 默认 160 |
| `MODEL.BiFPN.NUM_REPEATS` | BiFPN 堆叠次数 | 默认 6 |
| `MODEL.BASIS_MODULE.NUM_BASES` | bases 数量 | 默认 4 |
| `MODEL.BASIS_MODULE.CONVS_DIM` | ProtoNet 内部通道 | 默认 128 |
| `MODEL.BASIS_MODULE.LOSS_ON` | 是否开启 basis 语义辅助损失 | **自定义数据集必须设为 `False`**，见第 8 节 |
| `MODEL.BASIS_MODULE.NUM_CLASSES` | 辅助语义头类别数 | 必须 = 数据集类别数 |
| `MODEL.BASIS_MODULE.NAME` | 在 basis registry 中查找的类名 | 默认 `"ProtoNet"`；两套 registry 里各注册了一个同名 `ProtoNet`，由 meta 架构决定用哪套 |
| `MODEL.BASIS_MODULE.LOSS_WEIGHT` | basis 语义辅助损失权重 | 默认 0.3，仅 `LOSS_ON=True` 时生效 |
| `MODEL.FCOS.NUM_CLASSES` | 检测头前景类别数 | 必须 = 数据集类别数 |
| `MODEL.FCOS.TOP_LEVELS` | BiFPN 额外生成的顶层数（p6/p7） | 默认 2 |

### 4.2 数据集路径

> ⚠️ `configs/*.yaml` 里只保存**数据集的注册名**（如 `Plantv2_train`），
> **真实磁盘路径不在 yaml 中**，而是写在 `tools/train_bl+.py` / `tools/train_scbl+ .py` 顶部的
> `DATASET_ROOT` 常量里。原先该值硬编码为另一台服务器的 `/mnt/cd/HCH/data/Plantv2/`。

现已改为可通过环境变量覆盖：

```bash
export GBADMASK_DATA_ROOT=/home/huachenghao/codes/GBADMask/datasets/Plantv2
```

未设置时回退到项目内的 `datasets/Plantv2`。

---

## 5. 环境安装

本项目需要编译 `adet._C`（DCNv2 / IoU loss / ML-NMS 等 CUDA 扩展），因此**必须**本地编译，不能只装纯 Python 包。

### 5.1 已创建的 conda 环境

| 项 | 值 |
| --- | --- |
| 环境名 | `gbadmask` |
| Python | 3.9 |
| PyTorch | 2.0.1（CUDA 11.7 构建） |
| torchvision | 0.15.2 |
| CUDA_HOME | `/usr/local/cuda-11.7` |
| GPU | RTX 3090 ×4（Driver 580.126.09，支持到 CUDA 13.0） |
| Detectron2 | v0.6（源码编译） |
| timm | 0.6.12（`cspvig.py` 依赖 `DropPath` / `ConvBnAct`） |

### 5.2 从零复现

```bash
# 1) 建环境
conda create -y -n gbadmask python=3.9 pip
conda activate gbadmask

# 2) PyTorch（与 /usr/local/cuda-11.7 对齐）
pip install torch==2.0.1 torchvision==0.15.2 numpy==1.23.5

# 3) 其余依赖
pip install timm==0.6.12 fvcore==0.1.5.post20221221 iopath==0.1.10 yacs==0.1.8 \
            pycocotools==2.0.6 opencv-python-headless==4.8.1.78 shapely cloudpickle \
            tabulate tensorboard tqdm termcolor matplotlib scipy \
            antlr4-python3-runtime==4.8 ply==3.11 rapidfuzz omegaconf

# 3.1) ⚠️ 必须降级 Pillow 并锁 antlr4
#     - detectron2 v0.6 用到 PIL.Image.LINEAR，Pillow ≥10 已移除该常量
#     - omegaconf 会把 antlr4 顶到 4.9.x，而 BAText 要求 4.8
pip install "Pillow==9.5.0" "antlr4-python3-runtime==4.8"

# 4) 编译环境（以下 5 个环境变量两个包都要设）
export CUDA_HOME=/usr/local/cuda-11.7     # ← 不能用默认的 13.0
export PATH=/usr/local/cuda-11.7/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6"         # ← RTX 3090；不加会为所有架构编译，极慢
export CC=/usr/bin/gcc-11                 # ← 系统默认 g++ 14.1 超过 CUDA 11.7 的上限
export CXX=/usr/bin/g++-11

# 5) Detectron2 v0.6（源码）
cd /home/huachenghao/codes
git clone -b v0.6 --depth 1 https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install --no-build-isolation --no-deps -e .

# 6) GBADMask（本仓库）
cd /home/huachenghao/codes/GBADMask
python -m pip install --no-build-isolation --no-deps -e .
```

> **为什么用 `pip install --no-build-isolation` 而不是 `python setup.py build develop`？**
> 后者会走 PEP 517 的 build isolation，pip 在临时目录建一个干净环境来解析 `setup.py`，
> 而 detectron2/adet 的 `setup.py` 开头就 `import torch` 读版本号 —— 隔离环境里没有 torch，
> 直接 `ModuleNotFoundError: No module named 'torch'`。加 `--no-build-isolation` 让它用当前环境。
> `--no-deps` 是为了避免 pip 顺手把 `Pillow` / `antlr4` 又升回去。

### 5.3 编译过程中需要修改的源码

`adet` 源自 2022 年的 AdelaiDet v0.2.0，在新工具链下有 1 处必须改：

**`adet/layers/csrc/ml_nms/ml_nms.cu`** —— `THC/THC.h` 在 PyTorch 1.11+ 已被移除，
该文件仍在使用 `THCState` / `THCudaMalloc` / `THCudaFree` / `THCudaCheck` / `THCCeilDiv`。
本项目已迁移到 ATen/c10 等价 API：

```cpp
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#define THCCeilDiv(a, b) (((a) + (b) - 1) / (b))
#define THCudaCheck(err) C10_CUDA_CHECK(err)
// THCudaMalloc(state, n)  →  c10::cuda::CUDACachingAllocator::raw_alloc(n)
// THCudaFree(state, ptr)  →  c10::cuda::CUDACachingAllocator::raw_delete(ptr)
// THCState *state = ...   →  直接删除
```

验证：

```bash
conda activate gbadmask
python -c "
import torch, detectron2, adet
from adet import _C
print('detectron2', detectron2.__version__, '| adet', adet.__version__)
print('adet._C ops:', sorted(a for a in dir(_C) if not a.startswith('_')))
"
# 期望输出：
#   detectron2 0.6 | adet 0.1.1
#   adet._C ops: ['bezier_align_backward', 'bezier_align_forward',
#                 'def_roi_align_backward', 'def_roi_align_forward', 'ml_nms']
```

跑一遍完整冒烟测试（构建模型 + 推理出 mask + 训练反向）：

```bash
cd /home/huachenghao/codes/GBADMask
python tests/tmp/smoke_test.py
```

该脚本会依次验证 4 个 run 配置能否：① 构建模型 ② GPU 推理并输出与原图同尺寸的
`pred_masks` ③ 前向算出 5 个 loss ④ 反向得到非零梯度。全部通过时输出 `passed 4/4`。

---

## 6. 数据准备

数据尚未下载到本机。目录需按 COCO 格式组织：

```
datasets/
└── Plantv2/                        # ← GBADMASK_DATA_ROOT 指向这里
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/                  # 训练图片
    └── val2017/                    # 验证图片
```

JSON 为标准 COCO `instances` 格式（`images` / `annotations` / `categories`），
`categories` 的条目会被自动读取并注册为 `thing_classes`。

> 若开启 `MODEL.BASIS_MODULE.LOSS_ON: True`，还需要额外生成 `thing_train2017/*.npz` 的
> basis 语义监督（见 `adet/data/dataset_mapper.py:195-214`），自定义数据集一般没有，建议直接关闭。

---

## 7. 训练 / 评估 / 推理

```bash
conda activate gbadmask
cd /home/huachenghao/codes/GBADMask
export CUDA_HOME=/usr/local/cuda-11.7
export GBADMASK_DATA_ROOT=$PWD/datasets/Plantv2

# 单卡训练
OMP_NUM_THREADS=1 python tools/train_bl+.py \
    --config-file configs/run-BlendMask+.yaml \
    --num-gpus 1

# 多卡训练（DDP）
OMP_NUM_THREADS=1 python tools/train_bl+.py \
    --config-file configs/run-BlendMask+.yaml \
    --num-gpus 4

# 评估
OMP_NUM_THREADS=1 python tools/train_bl+.py \
    --config-file configs/run-BlendMask+.yaml \
    --eval-only --num-gpus 1 \
    MODEL.WEIGHTS output/blendmask-plus/model_final.pth

# 推理
python demo/demo.py \
    --config-file configs/run-BlendMask+.yaml \
    --input datasets/Plantv2/val2017/*.jpg \
    --opts MODEL.WEIGHTS output/blendmask-plus/model_final.pth
```

可用的训练脚本：

| 脚本 | 数据集 | 对应配置 |
| --- | --- | --- |
| `tools/train_net.py` | 官方（命令行 `--opts DATASETS.TRAIN`） | 通用 |
| `tools/train_bl+.py` | `Plantv2_*` / `Strawberry_*`（自动扫描） | `run-BlendMask+.yaml`、`run-vig.yaml`、`run-BlendMask2-vig.yaml` |
| `tools/train_scbl+ .py` | `Plantv2_*` / `Strawberry_*`（自动扫描） | `run-SCBlendMask+ .yaml` |

两个脚本数据集注册逻辑相同，可互换；区别只在 `OUTPUT_DIR` 与历史用途。
二者都会在启动时扫描 `GBADMASK_DATA_ROOT`（默认 `datasets/Plantv2`）及其同级目录，
把存在的 COCO `instances_*.json` 注册成 `<目录名>_train` / `<目录名>_val`。

典型实验流程（ablation）：

```bash
# ① 官方 basis 模块 + ViG 骨干（基线）
OMP_NUM_THREADS=1 python tools/train_bl+.py --config-file configs/run-vig.yaml --num-gpus 1

# ② 改进 basis 模块（GC 融合）+ ViG 骨干
OMP_NUM_THREADS=1 python tools/train_bl+.py --config-file configs/run-BlendMask2-vig.yaml --num-gpus 1

# ③ 改进 basis 模块 + Lcspvig 骨干
OMP_NUM_THREADS=1 python tools/train_bl+.py --config-file "configs/run-SCBlendMask+ .yaml" --num-gpus 1
```

> 注意文件名含空格，命令行要加引号（详见 [P1-8](#p1--工程健壮性)）。

---

## 8. 已知问题

> 以下问题都是当前代码里**真实存在**的，按严重程度排序。前 5 条会导致训练直接跑不起来。

| # | 位置 | 问题 | 影响 |
| --- | --- | --- | --- |
| 1 | `adet/modeling/backbone/bifpn.py:402-424`（**已修复**） | `build_fcos_cspvig_bifpn_backbone` 的 docstring `"""` 没有闭合，导致其后所有内容被吞进字符串，最终 `unterminated triple-quoted string literal` | **`import adet` 直接 SyntaxError**，连环境都装不完。现已补上结尾 `"""` 并修掉 `Lcspvig` 函数体首行多出的缩进空格 |
| 2 | `adet/modeling/blendmask/basis_module2.py:18`（**已缓解**） | `ProtoNet` 注册在独立的 `Registry("BASIS_MODULE2")`，而 `blendmask.py` 用的是 `basis_module.build_basis_module`（查 `Registry("BASIS_MODULE")`） | 改进版**只能**通过 `META_ARCHITECTURE="BlendMask2"` 生效。已新增 `configs/run-BlendMask2-vig.yaml`，`run-SCBlendMask+ .yaml` 也已还原为 `BlendMask2`；但 `run-BlendMask+.yaml` / `run-vig.yaml` 仍是官方 ProtoNet。两套 registry 并存的写法仍建议统一（见 P0-2） |
| 3 | `configs/run-BlendMask+.yaml` 等（**已修复**） | `MODEL.BACKBONE.NAME` 拼错：`build_fcos_csp_vig_bifpn_backbone` / `build_fcos_Lcsp_vig_bifpn_backbone` / `build_fcos_cspvigm_bifpn_backbone` 均未注册 | registry 报 `KeyError`。已全部改为实际注册名 `build_fcos_cspvig_bifpn_backbone` / `build_fcos_Lcspvig_bifpn_backbone` |
| 4 | `configs/run-SCBlendMask+ .yaml`、`run-vig.yaml`（**已修复**） | `META_ARCHITECTURE` 原为 `"BlendMask2"` / `"BlendMask3"`，当时均未注册。`BlendMask3` 无对应实现，已改为 `"BlendMask"`；`BlendMask2` 现已由 `blendmask/class BlendMask2` 补齐注册，配置已还原 | 构建模型时报 `KeyError` |
| 5 | `configs/Base-BlendMask*.yaml` | `MODEL.BASIS_MODULE.LOSS_ON: True` 需要 `thing_train2017/*.npz` 的 basis 语义监督 | 自定义数据集上 dataloader 直接 `FileNotFoundError` |
| 6 | `adet/modeling/blendmask/blendmask2.py`（**已修复**） | 原 `from .blenders import build_blenders`（实际文件是 `blender.py`，函数是 `build_blender`）、`build_basis_module3(...)`（不存在）、`__all__` 写 `"BlendMask2"` 但类名是 `BlendMask1` | 该文件 import 即失败。现已逐项修正并注册为 `BlendMask2` |
| 7 | `tools/train_bl+.py:47,53,62`、`tools/train_scbl+ .py` | `CUDA_VISIBLE_DEVICES="0,1,2"`、数据集根目录、以及**模块加载时**的 `json.load(open(TRAIN_JSON))` 全部硬编码 | 换机器 / 数据未就绪时 import 阶段就崩 |
| 8 | `setup.py:74-87` | `install_requires` 缺 `timm`，而 `cspvig.py` / `Lcspvig.py` 必须 import `timm` | 干净环境下 `import adet` 失败 |
| 9 | `adet/modeling/backbone/Lcspvig.py:305-323` | `BN_LSKb_act` / `LSKblock` 定义后从未被 `MobileViG.forward` 调用 | "SC"（大核选择）分支实际未生效，`Lcspvig` ≡ `cspvig` |
| 10 | `adet/modeling/blendmask/basis_module2.py:78-79` | `nn.Conv2d(..., kernel_size=1, stride=1, padding=1)`，1×1 卷积配 `padding=1` 会改变空间尺寸且无意义 | 分辨率出现 off-by-2，靠后续插值掩盖 |
| 11 | `adet/modeling/backbone/bifpn.py:401` | `@BACKBONE_REGISTRY.register()` 与 `def` 之间缺空行、`backbone/__init__.py` 里 3 个 import 挤在一行 | 纯风格问题 |
| 12 | `adet/modeling/backbone/cspvig.py:6,8-10,343` | `import numpy` 未使用；`IMAGENET_DEFAULT_MEAN/STD`、`register_model`、`ConvBnAct` 均为无效 import（timm 1.x 已移除 `timm.models.layers.ConvBnAct`） | 在 timm ≥ 1.0 下 import 失败；依赖必须锁 `timm==0.6.12` |
| 13 | `tools/train_scbl+ .py` | 文件名含空格 | 命令行需引号，易踩坑 |
| 14 | `configs/run-*.yaml` | `MODEL.RESNETS.DEPTH: 50` 对 ViG 骨干无效 | 误导性配置 |
| 15 | `adet/modeling/blendmask/blendmask.py:48`（**已修复**） | `top_layer` 的输入通道写死读 `cfg.MODEL.FPN.OUT_CHANNELS`（默认 256），但用 BiFPN 时实际通道由 `MODEL.BiFPN.OUT_CHANNELS` 决定（默认 160） | **所有 BiFPN 配置一做前向就崩**：`expected input to have 256 channels, but got 160`。已改为向 `self.backbone.output_shape()` 查询真实通道数，FPN / BiFPN 自动适配 |
| 16 | `adet/layers/csrc/ml_nms/ml_nms.cu`（**已修复**） | 使用 `THC/THC.h` 系列 API，而 THC 在 PyTorch 1.11+ 已被移除 | 编译 `adet._C` 时 `fatal error: THC/THC.h: No such file`。已迁移到 ATen/c10，见 5.3 节 |
| 17 | 环境依赖 | `detectron2 v0.6` 依赖 `PIL.Image.LINEAR`（Pillow ≥10 已移除）；`omegaconf` 会把 `antlr4-python3-runtime` 顶到 4.9.x（BAText 要求 4.8） | 需 `pip install Pillow==9.5.0 antlr4-python3-runtime==4.8`，见 5.2 节 |

---

## 9. 优化建议

### P0 —— 先让代码能跑通

1. ~~**修复 `bifpn.py` 未闭合的 docstring**（第 8 节 #1）~~ — 已修掉（含 `Lcspvig` 函数体首行的多余缩进）。
2. ~~**修正 3 个 run 配置的 `MODEL.BACKBONE.NAME` 与 `META_ARCHITECTURE`**（第 8 节 #3、#4）~~ — 已全部改为实际注册名。
3. ~~**修复 `blendmask2.py` 的三处断链**（第 8 节 #6）~~ — `blenders` → `blender`、`build_basis_module3` → `build_basis_module2`、`BlendMask1` → `BlendMask2`。
4. ~~**修复 `top_layer` 通道硬编码**（第 8 节 #15）~~ — 改为向 backbone 查询真实通道数，BiFPN 配置现在能跑前向了。
5. ~~**把 `ml_nms.cu` 从 THC 迁到 ATen/c10**（第 8 节 #16）~~ — torch 2.0 下 `adet._C` 已能编译。
6. **统一两套 basis registry**：目前 `basis_module.py`（`Registry("BASIS_MODULE")`）与 `basis_module2.py`（`Registry("BASIS_MODULE2")`）各自注册了一个同名 `ProtoNet`，只能靠 `META_ARCHITECTURE` 在 `BlendMask` / `BlendMask2` 之间整体切换，**无法单独替换 basis 模块**。建议删掉第二个 registry，把改进版改名 `ProtoNetV2` 注册进第一个，再用 `MODEL.BASIS_MODULE.NAME: "ProtoNet"/"ProtoNetV2"` 自由组合——这样"官方 CE vs FDC 损失""ViG vs Lcspvig"才能正交消融，而不是现在的 2×2 只能跑对角。
7. **自定义数据集关掉 `BASIS_MODULE.LOSS_ON`**，或写一个离线脚本把 COCO 的 instance mask 预先烘成 `thing_train2017/*.npz`。目前所有 run 配置都已显式设为 `False`。

### P1 —— 工程健壮性

8. **数据集与设备全部外置**：用 `--opts` 或环境变量（已支持 `GBADMASK_DATA_ROOT`）注入路径；把 `json.load(open(TRAIN_JSON))` 从模块级挪进 `register_dataset()`，做到"数据不在也能 import"；`CUDA_VISIBLE_DEVICES` 改由命令行 / 环境控制。
9. **补 `setup.py` 依赖**：缺 `timm`（必须锁 `<1.0`，否则 `cspvig.py` 的 `ConvBnAct` import 失败）、`rapidfuzz`（`adet/evaluation/text_eval_script.py` 依赖，训练脚本会 `from adet.evaluation import TextEvaluator`）、`omegaconf`（detectron2 v0.6 依赖）。建议补一份 `requirements.txt` 与 `environment.yml`，把本文 5.2 节测通的版本组合固化下来。
10. **清理死代码**：`blendmask/build.py`、`ca.py`、`cbam.py`、`dice_loss.py`、`ND_Crossentropy.py`、`simam_module`、`cspvig.py` 里未使用的 `numpy` / timm import。`torch-stat.py` 重命名为 `torch_stat.py` 或移入 `util/`。
    另外 `adet/data/builtin.py` 与 `util/`（11 个脚本）里还留着大量上游 **ABCNet / 越南语文本检测**的数据集注册与处理代码，与本项目完全无关，建议一并清理，避免 `import adet` 时注册一堆用不到的数据集。
11. **修文件名空格**：`tools/train_scbl+ .py` → `tools/train_scbl_plus.py`，`configs/run-SCBlendMask+ .yaml` → `configs/run-SCBlendMask-plus.yaml`，避免 shell 转义问题。

### P2 —— 效果与性能

12. **真正接入 LSK**：把 `BN_LSKb_act` 用到 `Lcspvig.MobileViG` 的 stage 输出或 BiFPN 的融合节点，否则 SC 变体与基线无差异；同时补一组 cspvig vs Lcspvig 的消融实验确认收益。
13. **GC block 的位置与数量需要消融**：目前 `basis_module2` 里插了 2 个 GC block（低层 24ch 分支 + tower 开头），但没有任何对照实验。建议拆成"仅低层 GC""仅 tower GC""都加"三档，另外把 `cbam.py` / `ca.py` 作为可替换项做横向对比（`MODEL.BASIS_MODULE.ATTN: "gc" / "cbam" / "ca" / "none"` 配置化）。
14. **FDC 损失的超参需要标定**：`gamma=0.75`、`0.8*CE + 0.2*(Dice+1)` 这组权重目前是拍的；`SoftDiceLoss` 默认 `batch_dice=False`、`do_bg=True`，在实例分割的极端不平衡场景下建议试 `do_bg=False`；另外 `DC_and_CE_loss.forward` 里 `target.view((-1,1))` 在 `weight` 为 tensor 的分支下才用到，目前 `weight=1` 恒为标量，该分支是死代码。
15. **修正 `basis_module2.py:78` 的 `padding=1`**，改为 `padding=0`，避免无谓的分辨率偏移。
16. **BiFPN 通道数过大**：`OUT_CHANNELS=160` + `NUM_REPEATS=6` 对单卡 3090 的显存压力不小。建议先试 `OUT_CHANNELS=96~128`、`NUM_REPEATS=3~4` 的组合做吞吐/精度权衡；另外 `SingleBiFPN.forward` 里 `torch.stack(input_nodes, dim=-1)` 会显式物化 `[B,C,H,W,n]` 的五维张量，是显存热点，可改成逐路加权累加或 `w1*x1 + w2*x2` 的展开写法。
17. **ViG 骨干的 `dpr` 分配有 bug**：`cspvig.py:326-331` 中 global stage 的循环里 `dpr_idx += 1` 在 `for j in range(global_blocks[0])` **循环体内部**，但每次迭代用的是同一个 `dpr[dpr_idx]`（因为 `+=1` 在 append 之后），导致同一个 stage 内所有 block 共享同一个 drop rate，且后续 stage 索引计算偏移。应改为每个 block 递增。
18. **加载 ImageNet 预训练**：`cspvig` 目前是随机初始化（`_initialize_weights`），`run-vig.yaml` 里 `MODEL.WEIGHTS` 指向 MobileViG 预训练权重的那行被注释掉了。解析 MobileViG 的 `state_dict` 做骨干初始化，通常比从头训练收敛快很多、精度也更高。
19. **训练策略**：`SOLVER.STEPS=(60000, 80000)` + `MAX_ITER=90000` 是 COCO 的 3× 配置，对小数据集（草莓/植株）严重过配，建议 `MAX_ITER=20000~30000`、`STEPS` 按 0.6/0.8 比例调整，并启用 `INPUT.MIN_SIZE_TRAIN` 的多尺度增强；`IMS_PER_BATCH=6` + `BASE_LR=0.001` 也不是 8 卡线性缩放后的值（官方 16 图对应 0.01），单卡 6 图建议 `BASE_LR≈0.0025` 起调。
20. **评估与可视化**：`TEST.EVAL_PERIOD=5000` 偏大，调试期建议调到 1000；`tools/visualize_data.py` 已存在，建议训练前先跑一遍确认标注与 `basis_sem` 正确。
21. **推理部署**：`onnx/` 目录已有导出脚本，但未见针对 ViG 骨干（`MRConv4d` 的动态 `max` 图卷积）的算子验证。ONNX 导出前建议先把 `MRConv4d` 的循环展开成固定数次 `torch.max`，否则 `loop` 节点在很多推理后端上不支持。

---

## 10. 致谢与引用

- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) —— 本项目的基础框架
- [Detectron2](https://github.com/facebookresearch/detectron2) —— 底层检测库
- [Vision GNN (ViG)](https://arxiv.org/abs/2206.00272) / MobileViG —— 图卷积骨干
- [GCNet](https://arxiv.org/abs/1904.14294) —— 全局上下文块
- [CBAM](https://arxiv.org/abs/1807.06521)、[Coordinate Attention](https://arxiv.org/abs/2103.02907)、[LSKNet](https://arxiv.org/abs/2303.09030) —— 注意力模块
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) —— Dice / ND-Crossentropy 损失实现

```BibTeX
@inproceedings{chen2020blendmask,
  title     = {{BlendMask}: Top-Down Meets Bottom-Up for Instance Segmentation},
  author    = {Chen, Hao and Sun, Kunyang and Tian, Zhi and Shen, Chunhua and Huang, Yongming and Yan, Youliang},
  booktitle = {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}

@article{han2022vision,
  title   = {Vision GNN: An Image is Worth Graph of Nodes},
  author  = {Han, Kai and Wang, Yunhe and Guo, Jianyuan and Tang, Yehui and Wu, Enhua},
  journal = {arXiv preprint arXiv:2206.00272},
  year    = {2022}
}

@misc{tian2019adelaidet,
  author       = {Tian, Zhi and Chen, Hao and Wang, Xinlong and Liu, Yuliang and Shen, Chunhua},
  title        = {{AdelaiDet}: A Toolbox for Instance-level Recognition Tasks},
  howpublished = {\url{https://git.io/adelaidet}},
  year         = {2019}
}
```

## License

上游 AdelaiDet 采用 2-clause BSD License（学术用途），详见 [LICENSE](LICENSE)。商业用途请联系原作者 [Chunhua Shen](mailto:chhshen@gmail.com)。
本仓库中新增的 `cspvig.py` / `Lcspvig.py` 等文件源自对应开源实现，请同时遵守其原始许可。
