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
    ├── basis_module2.py     ★ 新增  改进版 ProtoNetV2（低层特征 + 注意力融合 + FDC 损失）
    ├── fdc_loss.py          ★ 新增  Focal-Dice-CrossEntropy 混合分割损失
    ├── GCblock.py           ★ 新增  GCNet 全局上下文块（被 ATTN="gc" 使用）
    ├── cbam.py              ★ 新增  CBAM 注意力（被 ATTN="cbam" 使用）
    ├── ca.py                ★ 新增  Coordinate Attention（被 ATTN="ca" 使用，已改为支持动态分辨率）
    ├── blendmask2.py        ★ 新增  BlendMask2（已改为继承自 BlendMask 的兼容别名）
    └── basis_module.py blender.py blendmask.py                                        （官方原样）
```

★ = 本项目新增 ；▲ = 在官方文件上修改 ；无标记 = 与官方一致

已删除的死代码：`blendmask/build.py`、`dice_loss.py`、`ND_Crossentropy.py`、`torch-stat.py`、
`simam_module`（`fdc_loss.py` 已自带所需实现，其余全仓库无引用）。

**本项目新增的配置节点**（`adet/config/defaults.py`）：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL.VIG.USE_LSK` | `True` | 是否在 Lcspvig 的 transition 层启用 LSK |
| `MODEL.BASIS_MODULE.ATTN` | `"gc"` | basis 内部注意力类型：`none`/`gc`/`cbam`/`ca` |
| `MODEL.BASIS_MODULE.LOW_LEVEL_DIM` | `24` | `ProtoNetV2` 低层细节分支通道数 |

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
├── run-BlendMask+.yaml            ★ 官方 basis + MobileViG + BiFPN
├── run-vig.yaml                   ★ 官方 basis + MobileViG + BiFPN（Plantv2，基线）
├── run-BlendMask2-vig.yaml        ★ 改进 basis + MobileViG + BiFPN
└── run-SCBlendMask-plus.yaml      ★ 改进 basis + Lcspvig(含 LSK) + BiFPN
```

**两个可正交组合的维度**（原本被绑死，现已解耦）：

| 维度 | 控制字段 | 取值 |
| --- | --- | --- |
| basis 模块 | `MODEL.BASIS_MODULE.NAME` | `ProtoNet`（官方）/ `ProtoNetV2`（改进：低层特征 + 注意力融合 + Focal-Dice-CE） |
| 骨干 | `MODEL.BACKBONE.NAME` | `build_fcos_cspvig_bifpn_backbone`（MobileViG）/ `build_fcos_Lcspvig_bifpn_backbone`（+LSK） |

因此 4 个 run 配置可以拆成 2×2 网格：
`run-vig`（官方+ViG）× `run-BlendMask2-vig`（改进+ViG）× `run-SCBlendMask-plus`（改进+Lcspvig），
缺的"官方 basis + Lcspvig"只需在 `run-SCBlendMask-plus.yaml` 上覆盖
`--opts MODEL.BASIS_MODULE.NAME ProtoNet` 即可。

> ⚠️ `META_ARCHITECTURE` 的 `BlendMask` 与 `BlendMask2` 现在**行为完全一致**
> （`BlendMask2` 仅为向后兼容保留的子类别名）。basis 模块的差异请一律用
> `MODEL.BASIS_MODULE.NAME` 控制，不要再依赖 meta 架构名。

> ⚠️ `ProtoNetV2` 的 Focal-Dice-CE 损失**只在 `BASIS_MODULE.LOSS_ON=True` 时生效**，
> 而 `LOSS_ON=True` 需要数据集额外提供 `thing_train2017/*.npz` 监督。
> 自定义数据集上 `LOSS_ON=False`，此时 `ProtoNetV2` 相对 `ProtoNet` 生效的只有
> **GC 特征融合 + 输出头多一层非线性**这两项。

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
| `MODEL.BASIS_MODULE.NAME` | basis 模块类名 | `ProtoNet`（官方）/ `ProtoNetV2`（改进），见上一节 |
| `MODEL.BASIS_MODULE.ATTN` | basis 内部注意力类型 | `none` / `gc`（默认）/ `cbam` / `ca`，仅 `ProtoNetV2` 支持 |
| `MODEL.BASIS_MODULE.LOW_LEVEL_DIM` | `ProtoNetV2` 低层细节分支通道数 | 默认 24 |
| `MODEL.BASIS_MODULE.LOSS_WEIGHT` | basis 语义辅助损失权重 | 默认 0.3，仅 `LOSS_ON=True` 时生效 |
| `MODEL.VIG.USE_LSK` | 是否在 Lcspvig 的 transition 层启用 LSK | 默认 `True`；设 `False` 则退化为普通 cspvig，用于消融 |
| `MODEL.FCOS.NUM_CLASSES` | 检测头前景类别数 | 必须 = 数据集类别数 |
| `MODEL.FCOS.TOP_LEVELS` | BiFPN 额外生成的顶层数（p6/p7） | 默认 2 |

### 4.2 数据集配置

推荐只写**数据集名称**，磁盘路径由脚本自动解析：

```yaml
DATASETS:
  NAME: "wheat_seg"     # 只写名字；自动注册 wheat_seg_{train,val,test}
                        # 并把 TRAIN/TEST 指向 wheat_seg_train / wheat_seg_val
```

`NAME` 的目录查找顺序（`tools/base.py:resolve_dataset_dir`）：绝对路径 >
`datasets/<名称>` > `GBADMASK_DATA_ROOT` 同级目录 > 扫描 `datasets/`
（因此 `datasets/HBueHxOW/wheat_seg` 这类嵌套布局也能找到）。

数据集来源优先级：**命令行 `--dataset <名称>`** > **配置 `DATASETS.NAME`** >
`DATASETS.TRAIN/TEST` 注册名反推。前两者会覆盖 `TRAIN/TEST`；最后一种不改配置，
只按其注册名反推需要注册哪些数据集。

若需自定义划分（例如用 train+val 训练），把 `NAME` 留空并显式写注册名：

```yaml
DATASETS:
  NAME: ""
  TRAIN: ("wheat_seg_train", "wheat_seg_val")
  TEST:  ("wheat_seg_test",)
```

> `MODEL.FCOS.NUM_CLASSES` 与 `MODEL.BASIS_MODULE.NUM_CLASSES` 必须等于数据集类别数。
> 换数据集忘了改会在训练中途抛 CUDA index 越界，因此启动时若检测到不一致会先打出
> WARNING 指明该改成多少（只提示，不改配置）。

#### 环境变量

需要指向非默认位置时可用环境变量覆盖：

```bash
export GBADMASK_DATA_ROOT=/home/huachenghao/codes/GBADMask/datasets/Plantv2
```

未设置且 `DATASETS.NAME` 也为空时，回退到项目内的 `datasets/Plantv2`。
有了 `DATASETS.NAME` 后一般不再需要设它。

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

跑一遍端到端验证。推荐用 6.2 节的 coco128-seg 小数据集，它会完整走通
「数据加载 → 训练 → 评估」链路：

```bash
export GBADMASK_DATA_ROOT=$PWD/datasets/coco128-seg
CUDA_VISIBLE_DEVICES=1 python tools/train_bl+.py \
    --config-file configs/run-coco128-test.yaml --num-gpus 1
```

若只想快速验证模型能否构建并前向/反向（不需要数据），可临时写一个脚本，
对 4 个 run 配置依次调用 `build_model` + `model(inputs)` + `loss.backward()`。
注意构造输入时：`gt_boxes` 必须是 `Boxes` 对象、`gt_masks` 必须是 `BitMasks`、
`basis_sem` 必须是 2D 的 `(H, W)`（内部会补 batch 维并再 `unsqueeze(1)`）。

---

## 6. 数据准备

### 6.1 目标项目的自有数据（草莓 / 植株）

目录需按 COCO 格式组织：

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

### 6.2 用公开小数据集验证训练链路（coco128-seg）

目标数据集尚未就绪时，可用 Ultralytics 的 **coco128-seg**（128 张 COCO 原图，
带真实 polygon mask）先验证整条训练链路。仓库内已附转换脚本
`datasets/prepare_coco_seg.py`（原始格式为 YOLO-seg，需转成 COCO instances）。

```bash
conda activate gbadmask
cd /home/huachenghao/codes/GBADMask

# 1) 下载（约 7 MB）
mkdir -p datasets/_dl
curl -L -o datasets/_dl/coco128-seg.zip \
     https://ultralytics.com/assets/coco128-seg.zip

# 2) 转换（自动按 25% 划分 val）
python datasets/prepare_coco_seg.py coco128-seg
#   -> datasets/coco128-seg/{annotations,train2017,val2017}
#      96 train / 32 val / 929 实例 / 69 类

# 3) 训练 600 次迭代（单卡 3090 约 1.5 分钟）
export GBADMASK_DATA_ROOT=$PWD/datasets/coco128-seg
CUDA_VISIBLE_DEVICES=1 python tools/train_bl+.py \
    --config-file configs/run-coco128-test.yaml --num-gpus 1
```

**实测结果**（`run-coco128-test.yaml`，600 iter 从头训练）：

| 指标 | bbox | segm |
| --- | --- | --- |
| AP | 0.181 | 0.089 |
| AP50 | 0.444 | 0.312 |
| APl | 0.651 | 0.347 |

峰值显存约 1.8 GB，速度约 0.145 s/iter。128 张图不足以训练出可用模型，
但 AP50 从 0 涨到 0.44 说明**数据加载、梯度流、mask 分支、评估链路全部正确**。

> 注意：本项目 `.gitignore` 默认忽略 `datasets/`，该目录仅在本地用于验证，不会被提交。
> 网络受限时 `images.cocodataset.org` 与 GitHub 常不可达，`ultralytics.com` 通常可用
> （实测约 100~675 KB/s）。

### 6.3 小麦病害分割数据集（wheat_seg）

`datasets/HBueHxOW/` 下有三份内容，只有 `zzy_dataset` 带分割标注：

| 目录 | 内容 | 是否可用 |
| --- | --- | --- |
| `zzy_dataset/` | YOLO-seg，`images/{train,val,test}` + `labels/{train,val,test}`，905 张图 | ✅ 转换源 |
| `zzy_wheat/` | 同批图的 LabelMe 源标注，用于核对类别名 | ✅ 辅助 |
| `wheatData_split910/` | 12 类文件夹、7653 张图的**纯分类**数据，**无 mask** | ❌ 跳过 |

```bash
cd /home/huachenghao/codes/GBADMask

# 转换（默认用相对软链接引用图片，省 458 MB；--copy 可改为复制）
python datasets/prepare_wheat_seg_coco.py
#   -> datasets/HBueHxOW/wheat_seg/{annotations,train2017,val2017,test2017}
#      train 631 张/1601 实例，val 90/261，test 178/429，12 类

# 训练（数据集名已写在 yaml 的 DATASETS.NAME 里，无需环境变量）
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python tools/train_bl+.py \
    --config-file configs/run-wheat-seg.yaml --num-gpus 1
```

12 个类别（YOLO `class_id` 顺序，COCO `category_id` 从 1 起）：
根冠腐烂 / 健康小麦 / 叶锈病 / 白粉病 / 散黑穗病 / 蚜虫病 / 孢囊线虫病 /
红蜘蛛 / 赤霉病 / 纹枯病 / 茎基腐 / 全蚀病。

类别名由 `zzy_wheat/*.json` 的 LabelMe 标签与 `labels/*.txt` 的 class_id 交叉比对得出
（原始包内没有 `classes.txt`）。转换时还做了两处清洗：6 个同时出现在两个 split 的
文件名按 train>val>test 去重（避免验证集泄漏）；多边形坐标反归一化后裁剪到图像范围内。

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
    MODEL.WEIGHTS output/blendmask-plus/model_final.pth
```

> **命令行覆盖配置的正确写法**：detectron2 的 `opts` 是**位置参数**
> （`nargs=argparse.REMAINDER`），**不是** `--opts` 标志。写成 `--opts A B` 会报
> `unrecognized arguments`。正确形式是把键值对直接放在命令末尾。

可用的训练脚本：

| 脚本 | 说明 | 对应配置 |
| --- | --- | --- |
| `tools/train_bl+.py` | **主脚本**。数据集从 `DATASETS.NAME` 或 `--dataset` 取 | 全部 |
| `tools/train_scbl_plus.py` | 兼容壳，内容为空，逻辑全部转发到 `train_bl+.py` | `run-SCBlendMask-plus.yaml` |
| `tools/train_net.py` | 官方版，**无数据集注册**（只能用 `adet/data/builtin.py` 里已注册的名字） | 通用 |

> 前两个脚本原是两份 290 行的 `Trainer` 副本，抽走路径与注册逻辑后已逐行等价，
> 故合并为一个实现。它们对应的实验差异**全在 yaml 里**，因此用哪个脚本都一样：
> `--config-file configs/run-SCBlendMask-plus.yaml` 才是决定因素。

全部实验变量都在配置里，换实验**不需要改代码**：

| 想调什么 | 改哪个配置键 |
| --- | --- |
| 数据集 | `DATASETS.NAME`（或命令行 `--dataset`） |
| 骨干 | `MODEL.BACKBONE.NAME`：`build_fcos_cspvig_bifpn_backbone` / `build_fcos_Lcspvig_bifpn_backbone` |
| basis 模块 | `MODEL.BASIS_MODULE.NAME`：`ProtoNet` / `ProtoNetV2` |
| 注意力类型 | `MODEL.BASIS_MODULE.ATTN`：`none` / `gc` / `cbam` / `ca` |
| LSK 开关 | `MODEL.VIG.USE_LSK` |
| 训练规模 | `SOLVER.MAX_ITER` / `IMS_PER_BATCH` / `BASE_LR` 等 |

典型实验流程（ablation）：

```bash
# ① 官方 basis 模块 + MobileViG（基线）
OMP_NUM_THREADS=1 python tools/train_bl+.py --config-file configs/run-vig.yaml --num-gpus 1

# ② 改进 basis 模块 + MobileViG
OMP_NUM_THREADS=1 python tools/train_bl+.py --config-file configs/run-BlendMask2-vig.yaml --num-gpus 1

# ③ 改进 basis 模块 + Lcspvig（含 LSK）
OMP_NUM_THREADS=1 python tools/train_bl+.py --config-file configs/run-SCBlendMask-plus.yaml --num-gpus 1

# ④ 消融：关掉 LSK，单独量化 LSK 的收益
OMP_NUM_THREADS=1 python tools/train_bl+.py \
    --config-file configs/run-SCBlendMask-plus.yaml --num-gpus 1 \
    MODEL.VIG.USE_LSK False

# ⑤ 消融：对比注意力类型（gc / cbam / ca / none）
OMP_NUM_THREADS=1 python tools/train_bl+.py \
    --config-file configs/run-BlendMask2-vig.yaml --num-gpus 1 \
    MODEL.BASIS_MODULE.ATTN cbam

# ⑥ 消融：官方 vs 改进 basis 模块（在同一骨干上）
OMP_NUM_THREADS=1 python tools/train_bl+.py \
    --config-file configs/run-BlendMask2-vig.yaml --num-gpus 1 \
    MODEL.BASIS_MODULE.NAME ProtoNet

# ⑦ 换数据集（只改一处，配合 DATASETS.NAME 使用）
OMP_NUM_THREADS=1 python tools/train_bl+.py --config-file configs/run-wheat-seg.yaml \
    --num-gpus 1 --dataset coco128-seg SOLVER.MAX_ITER 600
```

上述 ④~⑥ 均已在 coco128-seg 上实测跑通（各 20 iter）。其中 ⑥ 需要注意
`SOLVER.MAX_ITER` 会沿用配置里的值，调试时可一并覆盖为较小的数。

> ④~⑥ 说明消融**不需要改代码**：骨干、basis 模块、注意力、LSK 全是配置项。
> 实测把 5 个变量（`META_ARCHITECTURE` / `BACKBONE.NAME` / `BASIS_MODULE.NAME` /
> `BASIS_MODULE.ATTN` / `VIG.USE_LSK`）全部通过 yaml 切换后训练正常。
> ⑦ 用 `--dataset` 临时换数据集时，记得两个 `NUM_CLASSES` 也要跟着改
> （不一致时启动时会有 WARNING 提示）。

---

## 8. 已知问题

> 下表按严重程度排序。**除 #5、#11、#14 外均已修复**（加删除线或标注"已修复"的条目）。
> #5 属于使用方式问题（自定义数据集必须关 `LOSS_ON`），#11 是纯风格，#14 已无残留。

| # | 位置 | 问题 | 状态与影响 |
| --- | --- | --- | --- |
| 1 | `adet/modeling/backbone/bifpn.py` | `build_fcos_cspvig_bifpn_backbone` 的 docstring `"""` 没有闭合，导致其后所有内容被吞进字符串 | **已修复**。原本 `import adet` 直接 SyntaxError |
| 2 | `adet/modeling/blendmask/basis_module2.py` | 自建 `Registry("BASIS_MODULE2")`，与其他模块不在同一 registry | **已修复**。改为复用 `BASIS_MODULE_REGISTRY`，改进版注册为 **`ProtoNetV2`**，由 `MODEL.BASIS_MODULE.NAME` 选择；basis 模块与 backbone 现可正交消融 |
| 2b | `adet/modeling/blendmask/blendmask2.py` | 原本是 `BlendMask` 的近乎逐行拷贝（约 150 行重复） | **已修复**。改为继承 `BlendMask`，仅作向后兼容别名 |
| 3 | `configs/run-BlendMask+.yaml` 等 | `MODEL.BACKBONE.NAME` 拼错（3 处） | **已修复**。改为实际注册名 |
| 4 | `configs/run-SCBlendMask-plus.yaml`、`run-vig.yaml` | `META_ARCHITECTURE` 原为 `"BlendMask2"` / `"BlendMask3"`，均未注册 | **已修复**。`BlendMask3` 无实现，落到 `BlendMask`；`BlendMask2` 已注册 |
| 5 | `configs/Base-BlendMask*.yaml` | `MODEL.BASIS_MODULE.LOSS_ON: True` 需要 `thing_train2017/*.npz` 监督 | **需注意**（非缺陷）。自定义数据集必须显式设为 `False`，所有 run 配置已设置。详见第 4 节 |
| 6 | `adet/modeling/blendmask/blendmask2.py` | `from .blenders import build_blenders`、`build_basis_module3(...)`、`__all__` 与类名不符 | **已修复** |
| 7 | `tools/train_bl+.py`、`tools/train_scbl_plus.py` | `CUDA_VISIBLE_DEVICES`、数据集根目录硬编码；`json.load` 在 import 时执行 | **已修复**。改为 `setdefault` + `GBADMASK_DATA_ROOT` + 延迟加载。现已进一步抽到 `tools/base.py` / `tools/register_datasets.py`，脚本内零路径硬编码 |
| 7b | `fvcore` + `configs/*.yaml` | `_open_cfg` 用 `g_pathmgr.open(filename, "r")`，**未指定 encoding**，编码取决于进程 locale | **已修复**（`adet/config/defaults.py` 显式 UTF-8）。原本在 locale 为 C/POSIX 的服务器上读中文 yaml 直接 `UnicodeDecodeError` |
| 8 | `setup.py` | `install_requires` 缺 `timm` | **已修复**。已补全并新增 `requirements.txt`（含 `Pillow<10`、`timm<1.0` 等约束） |
| 9 | `adet/modeling/backbone/Lcspvig.py` | `LSKblock` / `BN_LSKb_act` 定义后从未被调用 | **已修复**。4 个 stage 的 transition 层已改用 `BN_LSKb_act`，参数量 4.14M → **4.53M**；新增 `MODEL.VIG.USE_LSK` 开关 |
| 10 | `adet/modeling/blendmask/basis_module2.py` | 1×1 卷积误用 `padding=1` | **已修复**。改为 `padding=0` |
| 11 | `adet/modeling/backbone/bifpn.py`、`backbone/__init__.py` | 装饰器与 `def` 之间缺空行；多个 import 挤在一行 | 纯风格问题，未改 |
| 12 | `adet/modeling/backbone/cspvig.py` | `import numpy` 未使用；`IMAGENET_DEFAULT_MEAN/STD`、`register_model`、`ConvBnAct` 均无效（timm 1.x 已移除 `ConvBnAct`） | **已修复**。仅保留实际使用的 `DropPath` |
| 13 | `tools/train_scbl+ .py`、`configs/run-SCBlendMask+ .yaml` | 文件名含空格 | **已修复**。已用 `git mv` 重命名为 `train_scbl_plus.py` / `run-SCBlendMask-plus.yaml` |
| 14 | `configs/run-*.yaml` | `MODEL.RESNETS.DEPTH` 对 ViG 骨干无效 | **已清理** |
| 15 | `adet/modeling/blendmask/blendmask.py` | `top_layer` 写死读 `MODEL.FPN.OUT_CHANNELS`(256)，而 BiFPN 实际是 160 | **已修复**。改为向 `backbone.output_shape()` 查询真实通道 |
| 16 | `adet/layers/csrc/ml_nms/ml_nms.cu` | 使用 `THC/THC.h` 系列 API，PyTorch 1.11+ 已移除 | **已修复**。迁移到 ATen/c10，见 5.3 节 |
| 17 | 环境依赖 | detectron2 v0.6 依赖 `PIL.Image.LINEAR`（Pillow ≥10 已移除）；`omegaconf` 会把 `antlr4` 顶到 4.9.x | **已在 `requirements.txt` 中锁定** `Pillow==9.5.0`、`antlr4-python3-runtime==4.8` |
| 18 | `adet/modeling/blendmask/ca.py` | `CA_Block` 构造时需固定 `h` / `w`，无法用于分辨率可变网络 | **已修复**。改为运行时推断尺寸 + `AdaptiveAvgPool2d((None, 1))`，现已可通过 `ATTN="ca"` 启用 |
| 19 | `adet/modeling/backbone/cspvig.py`、`Lcspvig.py` | global stage 的 `Grapher` 与 `FFN` 共用同一个 drop rate，而 local stage 每个 block 一个 | **已修复**。改为各占一个，`n_blocks` 同步按 `2 * sum(global_blocks)` 计 |
| 20 | `adet/modeling/backbone/bifpn.py` | 融合时 `torch.stack(nodes, -1)` 显式物化 `[B,C,H,W,n]` 五维张量 | **已优化**。改为逐路加权累加（数值等价，实测偏差 0.0），峰值显存省约 10% |
| 21 | `tools/train_bl+.py:42`、`tools/train_scbl_plus.py:42` | `from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper` —— 该模块在本仓库中**不存在** | **已修复**。改为本项目实际需要的 `DatasetMapperWithBasis`（BlendMask 需要它产出 `basis_sem`）。原本一运行训练脚本就 `ModuleNotFoundError` |
| 22 | `adet/evaluation/text_eval_script.py:10` | `from rapidfuzz import string_metric`，而 rapidfuzz 3.x 已移除该子模块 | **已修复**。改为优先用 `rapidfuzz.distance.Levenshtein`，并保留旧版回退分支 |
| 23 | `adet/modeling/blendmask/ca.py` | `split([w, h], 3)` 顺序写反；cat 时先放 `x_h`（长度 h），应按 `[h, w]` 切 | **已修复**。原写法在**非正方形**特征图上会崩（`expanded size (64) must match (80)`）。正方形输入时不报错，所以此前冒烟测试未暴露 |

---

## 9. 优化建议

### 已完成

| # | 内容 | 结果 |
| --- | --- | --- |
| **新增** | **数据集注册与路径外置** | 新增 `tools/base.py`（路径/环境/COCO 目录约定）与 `tools/register_datasets.py`（给名字即注册）。训练脚本不再含任何路径硬编码 |
| **新增** | **合并重复训练脚本** | `train_scbl_plus.py` 原是 `train_bl+.py` 的 290 行副本，抽走公共逻辑后逐行等价，现改为转发壳。实验差异全在 yaml |
| **新增** | **数据集名写入 yaml** | 新增 `DATASETS.NAME`，只写名字即可自动注册并填充 `TRAIN/TEST`；优先级 `--dataset` > `DATASETS.NAME` > `TRAIN/TEST` 反推。换数据集还会校验 `NUM_CLASSES` 一致性 |
| P0-1~5 | 修复阻断性缺陷（docstring / registry 名 / 断链 / top_layer 通道 / THC 迁移） | `import adet` 与全部配置前向+反向均通过 |
| P0-6 | **统一两套 basis registry** | 改进版注册为 `ProtoNetV2`，由 `MODEL.BASIS_MODULE.NAME` 选择。basis 模块与 backbone 现可**正交组合**（原来只能沿对角线跑 2×2） |
| P0-7 | 自定义数据集关 `LOSS_ON` | 所有 run 配置已显式设为 `False` |
| P1-8 | 数据集与设备外置 | `GBADMASK_DATA_ROOT` 环境变量 + 类别 json 延迟加载 + `setdefault` |
| P1-9 | 补 `setup.py` 依赖 + `requirements.txt` | 已补 `timm<1.0` / `rapidfuzz` / `omegaconf` 等，并固化实测版本组合 |
| P1-10 | 清理死代码 | 删除 `blendmask/build.py`、`dice_loss.py`、`ND_Crossentropy.py`、`torch-stat.py`、`simam_module`；清理无效 import。**注意 `ca.py` / `cbam.py` 未被删除**，而是改造成可用模块（见 P2-13） |
| P1-11 | 文件名去空格 | `git mv` 重命名为 `train_scbl_plus.py` / `run-SCBlendMask-plus.yaml`（保留历史） |
| P2-12 | **真正接入 LSK** | Lcspvig 的 4 个 transition 层改用 `BN_LSKb_act`，参数量 4.14M → **4.53M**；新增 `MODEL.VIG.USE_LSK` 开关 |
| P2-13 | **注意力配置化** | 新增 `MODEL.BASIS_MODULE.ATTN: none/gc/cbam/ca`。`ca.py` 已从"需固定 h/w 的死代码"改造为支持动态分辨率 |
| P2-15 | 修 `padding=1` | 改为 `padding=0` |
| P2-16 | **BiFPN 显存优化** | `torch.stack(nodes, -1)` 改为逐路加权累加；实测数值偏差 **0.0**，峰值显存省约 **10%** |
| P2-17 | 修 `dpr` 分配 | 原实现 global stage 的 `Grapher` 与 `FFN` **共用**同一个 drop rate（local stage 则每个 block 一个，两者不一致）。现改为各占一个，`n_blocks` 同步按 `2 * sum(global_blocks)` 计 |
| 新增 | **修复配置文件编码** | fvcore 读 yaml 未指定 encoding，依赖 locale。已在 `adet/config/defaults.py` 显式 UTF-8，locale 为 C/POSIX 的服务器也能读中文注释的 yaml |
| 新增 | **修复训练脚本导入** | `tools/train_bl+.py` 与 `train_scbl_plus.py` 导入不存在的 `adet.data.fcpose_dataset_mapper`，一运行就 `ModuleNotFoundError`。已改为 BlendMask 实际需要的 `DatasetMapperWithBasis` |
| 新增 | **兼容 rapidfuzz 3.x** | `adet/evaluation/text_eval_script.py` 用 `rapidfuzz.string_metric`，3.x 已移除。改为 `rapidfuzz.distance.Levenshtein` 并保留旧版回退 |
| 新增 | **修复 CA 在非正方形特征图上崩溃** | `ca.py` 的 `split` 顺序写反。正方形输入不报错，非正方形会崩；此前冒烟测试用正方形输入未暴露，真实训练（多尺度）才触发 |

### 待办 —— 需要训练数据或实验才能定论

13. **消融实验验证收益**：现在开关都已就位，建议按以下顺序跑（每组只需改 1~2 个字段）：
    - **basis 模块**：`run-vig.yaml`（ProtoNet）vs `run-BlendMask2-vig.yaml`（ProtoNetV2）
    - **骨干**：`run-BlendMask2-vig.yaml`（MobileViG）vs `run-SCBlendMask-plus.yaml`（Lcspvig，含 LSK）
    - **LSK 本身**：`run-SCBlendMask-plus.yaml` 配 `MODEL.VIG.USE_LSK: False` 与 `True` 对比
    - **注意力类型**：`MODEL.BASIS_MODULE.ATTN` 取 `none` / `gc` / `cbam` / `ca` 四档
    四组都可通过 `--opts` 覆盖，无需新增配置文件。
14. **FDC 损失的超参需要标定**：`gamma=0.75`、`0.8*CE + 0.2*(Dice+1)` 这组权重目前是拍的；`SoftDiceLoss` 默认 `batch_dice=False`、`do_bg=True`，在实例分割的极端不平衡场景下建议试 `do_bg=False`；另外 `DC_and_CE_loss.forward` 里 `target.view((-1,1))` 在 `weight` 为 tensor 的分支下才用到，目前 `weight=1` 恒为标量，该分支是死代码。
15. **BiFPN 通道数**：`OUT_CHANNELS=160` + `NUM_REPEATS=6` 显存压力不小，可试 `96~128` / `3~4` 的组合作吞吐-精度权衡（已实测：512×512 + LOSS_ON 训练峰值约 0.9 GB，余量充足，实际可先按大 batch 跑）。
16. **加载 ImageNet 预训练**：`cspvig` 目前是随机初始化（`_initialize_weights`），`run-vig.yaml` 里 `MODEL.WEIGHTS` 指向 MobileViG 权重的那行被注释掉了。解析 MobileViG 的 `state_dict` 做骨干初始化，通常收敛更快、精度更高。
17. **训练策略**：`SOLVER.STEPS=(60000, 80000)` + `MAX_ITER=90000` 是 COCO 的 3× 配置，对小数据集（草莓/植株）可能过配，建议 `MAX_ITER=20000~30000`、`STEPS` 按 0.6/0.8 比例调整；`IMS_PER_BATCH=6` + `BASE_LR=0.001` 也不是线性缩放值（官方 16 图对应 0.01），单卡 6 图建议 `BASE_LR≈0.0025` 起调。**建议在数据集就绪后按实际收敛曲线调**。
18. **评估与可视化**：`TEST.EVAL_PERIOD=5000` 偏大，调试期建议 1000；训练前先跑 `tools/visualize_data.py` 确认标注正确。
19. **推理部署**：`onnx/` 目录已有导出脚本，但未见针对 ViG 骨干（`MRConv4d` 的动态 `max` 图卷积）的算子验证。ONNX 导出前建议先把 `MRConv4d` 的循环展开成固定数次 `torch.max`，否则 `loop` 节点在很多推理后端上不支持。
20. **可选清理**：`adet/data/builtin.py` 与 `util/`（11 个脚本）里还留着上游 **ABCNet / 越南语文本检测**的数据集注册与处理代码，与本项目无关。清理可减少 `import adet` 时注册的无用数据集，但涉及面较广，建议单独一次改动后重新编译验证。

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
