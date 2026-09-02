# 小麦病害实例分割数据集（wheat_seg / wheat_seg_clean）

`datasets/HBueHxOW/` 下小麦病害数据集的**带分割标注**部分，已转换为 COCO instances
格式。现有**两份产物**，推荐使用 `wheat_seg_clean`：

| 数据集 | 生成方式 | 图片 | 实例 | 验证集泄漏 | 用途 |
| --- | --- | --- | --- | --- | --- |
| `wheat_seg` | 按**文件名**去重 | 899 | 2291 | ⚠️ 有（30 组） | 与历史基线对齐 |
| **`wheat_seg_clean`** | 按**内容 MD5**去重 + 移除类别冲突 | **818** | **2121** | ✅ 无 | **正式实验推荐** |

注册名分别为 `wheat_seg_{train,val,test}` 与 `wheat_seg_clean_{train,val,test}`。

转换脚本：[`datasets/prepare_wheat_seg_coco.py`](datasets/prepare_wheat_seg_coco.py)
训练配置：[`configs/run-wheat-seg.yaml`](configs/run-wheat-seg.yaml)

> **两版差异详见第 7 节**。简单说：`wheat_seg` 只按文件名去重，漏掉了 30 组
> 「异名同图」（如 `train/CrownAndRootRot_11.jpg` 与 `val/CrownAndRootRot_504.jpg`
> 字节完全相同），会让 val/test 指标虚高。

---

## 1. 概览

| 项 | `wheat_seg` | `wheat_seg_clean` |
| --- | --- | --- |
| 任务 | 实例分割（12 类小麦病害 + 健康） | 同左 |
| 图片总数 | 899 | **818** |
| 实例总数 | 2291 | **2121** |
| 类别数 | 12 | 12 |
| 平均实例/图 | 2.55 | 2.59 |
| 跨 split 内容泄漏 | 30 组 | **0** |
| 标注格式 | 多边形（polygon），COCO instances | 同左 |
| 许可证 | CC0 | CC0 |
| 来源 | AI Studio 公开数据集（上传者 HBueHxOW） | 同左 |
| 磁盘占用（软链接） | 1.3 MB | 1.2 MB |

原始 README 说明数据构成为：约 50% 来自 IMU 的 LWDCD2020 公开数据集，
30% 大田采集，20% 网络爬虫。原始发布页称"近 7000 张"，但**其中带分割标注的只有
905 张**，其余 7653 张为纯分类数据（见第 6 节）。

---

## 2. 目录结构

### 2.1 源数据

```
datasets/HBueHxOW/
├── zzy_dataset/            ← 转换源（YOLO-seg 格式）
│   ├── images/{train,val,test}/*.jpg      905 张
│   └── labels/{train,val,test}/*.txt      905 个（与图片一一对应）
├── zzy_wheat/              ← LabelMe 源标注，用于核对类别名
│   ├── *.json   899 个
│   ├── *.txt    899 个
│   └── *.jpg    884 个（缺 15 张）
├── wheatData_split910/     ← 纯分类数据，无 mask，不参与转换
│   └── {train,val,test}/<12 类文件夹>/    7653 张
├── wheat_seg/              ← 转换输出 A：按文件名去重（899 张）
└── wheat_seg_clean/        ← 转换输出 B：按内容去重 + 去冲突（818 张，推荐）
```

### 2.2 转换输出

```
datasets/HBueHxOW/{wheat_seg | wheat_seg_clean}/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_test2017.json
├── train2017/   val2017/   test2017/
└── (图片为指向 ../../zzy_dataset/images/<split>/*.jpg 的相对软链接)
```

图片默认用**相对软链接**，省约 450 MB 磁盘。需要独立副本时用 `--copy`。

---

## 3. 划分统计

源数据共 905 张，三份口径的规模对比：

| split | 源图片 | `wheat_seg`<br>(文件名去重) | `wheat_seg_clean`<br>(内容去重+去冲突) |
| --- | --- | --- | --- |
| train | 631 | 631 | **584** |
| val | 91 | 90 | **82** |
| test | 183 | 178 | **152** |
| **合计** | **905** | **899** | **818** |

实例数：

| split | `wheat_seg` | `wheat_seg_clean` |
| --- | --- | --- |
| train | 1601 | **1489** |
| val | 261 | **248** |
| test | 429 | **384** |
| **合计** | **2291** | **2121** |

两版均无无标注图（每张图至少 1 个实例）。

> **test 集当前不参与训练**：`DATASETS.NAME` 只把 `TRAIN/TEST` 指向 `_train`/`_val`。
> 如需用 test 评估，改 yaml 里的 `DATASETS.TEST` 即可。

---

## 4. 类别与分布

类别顺序沿用 YOLO `class_id`（0 起），COCO `category_id` 为 `class_id + 1`（1 起）。

下表为 `wheat_seg`（899 张）的分布，`wheat_seg_clean`（818 张）见右侧合计列。

| id | 英文名 | 中文名 | train | val | test | 899 合计 | **818 合计** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CrownAndRootRot | 根冠腐烂 | 202 | 18 | 48 | 268 | **210** |
| 2 | HealthyWheat | 健康小麦 | 209 | 28 | 45 | 282 | **268** |
| 3 | LeafRust | 叶锈病 | 98 | 10 | 23 | 131 | **116** |
| 4 | PowderyMildew | 白粉病 | 170 | 77 | 76 | 323 | **308** |
| 5 | WheatLooseSmut | 散黑穗病（散斑） | 127 | 14 | 30 | 171 | **169** |
| 6 | WheatAphids | 蚜虫病 | 119 | 20 | 34 | 173 | **160** |
| 7 | WheatCystNematode | 孢囊线虫病 | 74 | 3 | 26 | 103 | **95** |
| 8 | WheatRedSpider | 红蜘蛛 | 154 | 12 | 16 | 182 | **181** |
| 9 | WheatScab | 赤霉病 | 209 | 36 | 55 | 300 | **300** |
| 10 | WheatSharpEyespot | 纹枯病 | 67 | 14 | 24 | 105 | **95** |
| 11 | WheatStalkRot | 茎基腐 | 91 | 18 | 27 | 136 | **122** |
| 12 | WheatTake-all | 全蚀病 | 81 | 11 | 25 | 117 | **97** |
| | **合计** | | **1601** | **261** | **429** | **2291** | **2121** |

（train/val/test 三列为 `wheat_seg` 的划分；`wheat_seg_clean` 各 split 为
train 584 / val 82 / test 152，划分比例相近。）

**类别不均衡约 3.2 倍**（PowderyMildew 308 vs WheatCystNematode、WheatSharpEyespot 95）。
`WheatCystNematode` 在 val 中仅 3 个实例，该类在验证集上的 AP 基本不可信。

> 原始包内**没有 `classes.txt`**。上表类别名由 `zzy_wheat/*.json` 的 LabelMe 标签与
> `zzy_dataset/labels/*.txt` 的 `class_id` 交叉比对得出，每类 40~110 张图、一一对应无歧义。

---

## 5. 标注特征

### 5.1 图片尺寸

| 项 | 宽 | 高 |
| --- | --- | --- |
| 最小 | 124 | 122 |
| 中位 | 640 | 480 |
| 最大 | 7360 | 4912 |

899 张图有 **501 种不同尺寸**（非统一分辨率）。最常见：800×506（51 张）、
640×480（28 张）、259×194（20 张）。69 张图长边超过 2000 px。

### 5.2 多边形

顶点数：最小 4，中位 11，均值 12.6，最大 85。

### 5.3 实例面积（像素）

| 分位 | 面积 |
| --- | --- |
| min | 134 |
| 25% | 2 869 |
| 中位 | 8 010 |
| 75% | 26 042 |
| max | 20 148 987 |

按占原图面积比例分桶：

| 占比 | 实例数 | 百分比 |
| --- | --- | --- |
| < 1% | 372 | 16.2% |
| 1–5% | 1099 | 48.0% |
| 5–20% | 630 | 27.5% |
| 20–50% | 171 | 7.5% |
| > 50% | 19 | 0.8% |

**19 个实例覆盖原图 50% 以上**，其中 3 个超过 90%：

| split | 文件 | 类别 | 原图 | 占比 | 顶点 |
| --- | --- | --- | --- | --- | --- |
| val | LeafRust_1.jpg | LeafRust | 2541×1049 | 99.1% | 4 |
| train | LeafRust_94.jpg | LeafRust | 1666×624 | 96.9% | 10 |
| test | LeafRust_23.jpg | LeafRust | 2541×1049 | 96.9% | 4 |

仅 4 个顶点就框住整张图，形态上更像"整片叶子"而非"病斑"，**疑为标注噪声**。
由于只占 0.8%，对整体训练影响有限，但会拉低大目标（`APl`）的可信度。

### 5.4 每图实例数

| 实例数 | 图片数 | 占比 |
| --- | --- | --- |
| 1 | 203 | 32.2% |
| 2 | 111 | 17.6% |
| 3 | 84 | 13.3% |
| 4 | 56 | 8.9% |
| 5 | 50 | 7.9% |
| 6–10 | 96 | 15.2% |
| > 10 | 30 | 4.8% |

单图最多 23 个实例。约 1/3 的图只有 1 个实例。

---

## 6. 未参与转换的数据

| 目录 | 内容 | 原因 |
| --- | --- | --- |
| `wheatData_split910/` | 7653 张，12 个类别文件夹 | **纯分类数据，无 mask**，无法用于实例分割 |
| `zzy_wheat/*.jpg` | 884 张 | 与 `zzy_dataset` 是同一批图，但缺 15 张，仅取用其 json 标核对类别名 |

`zzy_dataset` 的 905 张图与 905 个标签**一一对应**（`imgs-labs = labs-imgs = 0`），
包括 15 张 `DSC17_*` / `IMG18_*` 非常规命名的图，已全部进入转换结果。

---

## 7. 数据质量问题与清洗

> ✅ **`wheat_seg_clean` 已修复 7.1 与 7.2**，即内容去重 + 移除类别冲突。
> 若有正式实验在跑，建议切到 `wheat_seg_clean` 重跑，否则 val AP 不可信。

### 7.1 跨 split 内容重复（验证集泄漏）

按 **MD5 内容哈希**统计，源数据 905 张中只有 **824 张是唯一内容**，
共 **75 组重复**，其中 **36 组跨 split**（会泄漏）、39 组在 split 内（浪费样本）。

只按**文件名**去重只能抓到 6 组，因为多数重复是**异名同图**：

```
train/CrownAndRootRot_11.jpg  train/CrownAndRootRot_344.jpg  val/CrownAndRootRot_504.jpg
train/CrownAndRootRot_395.jpg train/CrownAndRootRot_87.jpg   test/CrownAndRootRot_87.jpg
train/LeafRust_0.jpg          train/LeafRust_48.jpg          test/LeafRust_5.jpg
val/LeafRust_1.jpg            test/LeafRust_23.jpg
train/WheatAphids_49.jpg      test/WheatAphids_3.jpg         test/WheatAphids_96.jpg
... 共 36 组
```

`wheat_seg_clean` 按内容去重，同组保留 **split 优先级最高**（train > val > test）、
其次文件名的一个副本，结果确定可复现。

### 7.2 标注冲突：同一张图被标成不同类别

6 组内容相同的图片，`*.txt` 里的类别**互相矛盾**（不只是文件名前缀不同）：

| 图片 A | 图片 B | 冲突类别 |
| --- | --- | --- |
| train/WheatAphids_87 | test/WheatRedSpider_10 | 蚜虫病 vs 红蜘蛛 |
| train/WheatSharpEyespot_31 | val/WheatStalkRot_27 | 纹枯病 vs 茎基腐 |
| train/WheatSharpEyespot_58 | train/WheatStalkRot_43 | 纹枯病 vs 茎基腐 |
| train/WheatSharpEyespot_86 | train/WheatStalkRot_91 | 纹枯病 vs 茎基腐 |
| train/WheatStalkRot_40 | test/WheatTake-all_96 | 茎基腐 vs 全蚀病 |
| train/WheatStalkRot_77 | train/WheatTake-all_15 | 茎基腐 vs 全蚀病 |

纹枯病 / 茎基腐 / 全蚀病 三者症状接近（均为茎基部病变），是主要混淆来源。

`wheat_seg_clean` 把这 6 组**整组丢弃**（共 12 张副本，去重后占 6 张）：
无法判断哪个标注是对的，保留任一都会引入噪声标签。

> 另有 **65 组**内容相同的图，类别一致但**多边形顶点数不同**（如
> `CrownAndRootRot_105` 32 顶点 vs `CrownAndRootRot_412` 29 顶点），
> 属于"同图重复标注"而非"类别冲突"，按去重保留一份即可，不丢弃。

### 7.3 清洗结果汇总

| 问题 | `wheat_seg` | `wheat_seg_clean` |
| --- | --- | --- |
| 跨 split 异名同图（30 组） | ❌ 残留 | ✅ 已去重 |
| split 内重复（39 组） | ❌ 残留 | ✅ 已去重 |
| 跨 split 重名重复（6 张） | ✅ 已去重 | ✅ 已去重 |
| 类别冲突（6 组 / 12 张） | ❌ 残留 | ✅ 整组丢弃 |
| 多边形越界坐标 | ✅ 已裁剪 | ✅ 已裁剪 |
| 19 个占比 >50% 的异常大实例 | ⚠️ 保留（见 5.3） | ⚠️ 保留 |

### 7.4 `wheat_seg_clean` 的验证结果

| 检查项 | 结果 |
| --- | --- |
| 图片总数 | 818（train 584 / val 82 / test 152） |
| **跨 split 内容泄漏** | **0 组** ✅ |
| **split 内重复** | **0** ✅ |
| labelme 与 txt 类别不一致 | 0 张（818 张可比对）✅ |
| 无标注图 | 0 ✅ |
| 类别数 | 12，id 连续 1~12 ✅ |
| 训练冒烟 | 5 iter 正常，loss 3.60 ✅ |

---

## 8. 使用方法

### 8.1 重新生成

```bash
cd /home/huachenghao/codes/GBADMask

# 推荐：内容去重 + 移除类别冲突（818 张）
python datasets/prepare_wheat_seg_coco.py \
    --out datasets/HBueHxOW/wheat_seg_clean --dedup hash --drop-conflict

# 与历史基线对齐：按文件名去重（899 张）
python datasets/prepare_wheat_seg_coco.py \
    --out datasets/HBueHxOW/wheat_seg --dedup name

# 其它选项
python datasets/prepare_wheat_seg_coco.py --copy            # 复制图片而非软链接
python datasets/prepare_wheat_seg_coco.py --dedup none      # 不去重（905 张，仅调试用）
```

`--dedup` 三档对比：

| 档位 | 判据 | 结果 |
| --- | --- | --- |
| `none` | 不去重 | 905 张（含全部重复与冲突） |
| `name` | 文件名相同 | 899 张（默认，历史行为） |
| `hash` | 内容 MD5 相同 | 824 张（推荐） |

配合 `--drop-conflict` 可再移除 6 组类别冲突 → 818 张。

### 8.2 检查可用数据集

```bash
python tools/register_datasets.py                    # 列出所有
python tools/register_datasets.py wheat_seg_clean    # 注册并打印统计
```

### 8.3 训练

数据集名已写在配置里，无需设置环境变量：

```bash
# 推荐：清洁版
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python tools/train_bl+.py \
    --config-file configs/run-wheat-seg.yaml --num-gpus 1 --dataset wheat_seg_clean
```

也可以直接改 yaml 里的 `DATASETS.NAME`：

```yaml
DATASETS:
  NAME: "wheat_seg_clean"   # 自动注册 wheat_seg_clean_{train,val,test} 并填充 TRAIN/TEST
MODEL:
  FCOS:
    NUM_CLASSES: 12
  BASIS_MODULE:
    NUM_CLASSES: 12
    LOSS_ON: False      # 无 thing_train2017/*.npz 语义监督，必须关闭
```

命令行 `--dataset` 优先级高于 yaml 的 `DATASETS.NAME`，适合临时切换
（如 `--dataset coco128-seg`），切换时会校验 `NUM_CLASSES` 是否匹配。

---

## 9. 训练注意事项

1. **优先用 `wheat_seg_clean`**。`wheat_seg` 的 val/test 里有 30 组图已在 train
   出现过（见 7.1），AP 虚高且不可比。若需与历史结果对齐再用 `wheat_seg`。
2. **规模偏小**。818 张图 / 2121 个实例，远小于 COCO（33 万张图）。
   `run-wheat-seg.yaml` 默认 36000 iter，容易过拟合，建议盯紧 val AP 曲线，
   必要时提前停或加强数据增强。
3. **类别不均衡**约 3.2 倍，少样本类（WheatCystNematode、WheatSharpEyespot 各 95）
   可考虑加权或过采样。
4. **尺寸差异大**（124×122 ~ 7360×4912）。多尺度训练已开启
   （`MIN_SIZE_TRAIN: (512, 576, 640, 704, 768)`，`MAX_SIZE_TRAIN: 1024`），
   但超大图会显著拉长单步耗时。
5. **`WheatCystNematode` 在 val 中仅 3 个实例**，该类的 val AP 基本不可信，
   建议以 test（26 个实例）或整体 mAP 为准。
