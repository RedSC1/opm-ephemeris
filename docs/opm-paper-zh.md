# OPM：一种面向长时段星历分发的紧凑多项式轨道文件格式（占位稿）

> 状态：论文占位稿。本文先固定结构、术语和主线叙述，数值结果表格后续用完整 600 年矩阵与 dense validation 结果补齐。

## 摘要

本文提出 OPM（Orbital Polynomial Map）文件格式及其生成方法，用于将 JPL DE441 等高精度星历在长时间跨度内压缩为可随机访问、可校验、跨平台读取的二进制轨道多项式文件。OPM 使用分段 Chebyshev 多项式、整数化残差系数、按轴按阶位宽打包、CRC64 校验和显式模型描述表，在保持亚毫角秒级方向精度的同时显著降低文件体积，并避免运行时依赖大型 BSP 文件。

当前实现采用 600 年分片作为主要分发粒度。生成流程分为 raw OPM 拟合与一次 no-size-increase polish 两阶段：Sun 使用 km 误差度量；Mercury、Venus、Moon 使用各自 native 相对向量角误差；EMB、Mars、Jupiter、Saturn、Uranus、Neptune、Pluto 使用 polished Sun anchor 的 heliocentric composite 角误差。对 SSB-centered bodies，当前收敛的 polish 策略为 `capped_lex_guarded_ceiling`，结合 `cheb-center-uniform-endpoints` active grid、shifted guard grid、endpoint-band guard nodes、局部峰值细化与 `pmax_first` 接受策略。

后续结果表明，OPM 在随机访问速度、文件体积和长时段验证误差之间取得了适合软件分发的工程折中。[TODO: 填入最终 size、p99、pmax、speedup 表格。]

## 1. 背景与目标

高精度天体位置计算通常直接依赖 SPK/BSP 星历文件。以 DE441 为例，其覆盖时间长、精度高，但文件体积较大，且随机访问和部署成本对轻量客户端、移动端、浏览器端或嵌入式场景并不理想。OPM 的目标不是替代 DE441 作为权威源数据，而是从 DE441 派生出一种面向应用分发的紧凑表示。

OPM 的设计目标包括：

1. **长时段覆盖**：以 600 年分片覆盖较长历史与未来区间；
2. **随机访问**：给定 Julian Date 可直接定位 segment 并重建位置；
3. **紧凑存储**：通过 reference shape、residual Chebyshev、量化与 bit packing 降低体积；
4. **精度稳定**：关注 p99、p99.9 与 worst-case pmax，而不是只看均方误差；
5. **格式自描述**：header 与 model table 记录中心、向量类型、模型类型、时间边界、量化步长、位宽等信息；
6. **可校验**：header 与 payload 分别使用 CRC64/ECMA；
7. **可复现生成**：固定一条 raw-only + one-pass polish 的生产路线，避免多候选人工挑选。

## 2. OPM1 文件格式概述

当前格式 magic 为：

```text
OPM1
```

文件扩展名为：

```text
.opm
```

OPM1 文件由固定 header、模型描述表、量化参数、位宽表、打包整数系数、可选 clock/model table 等部分组成。header 固定为 320 bytes，并包含：

- magic 与 endian tag；
- source ephemeris 与 coverage JD range；
- body id / center id；
- storage vector kind；
- model kind；
- segment count 与 segment addressing；
- residual degree；
- reference shape degree；
- quantization steps；
- block offsets / sizes；
- header CRC64 与 payload CRC64。

当前 canonical storage vector 包括：

```text
SSB -> Body
Sun -> Body
Earth -> Moon
raw XYZ Chebyshev
```

其中不同 body 会根据物理结构选择不同存储向量。例如 Mercury/Venus 使用 `Sun -> Body`，Moon 使用 `Earth -> Moon`，outer major bodies 多使用 `SSB -> Body`。

## 3. 分段模型

OPM 将覆盖区间划分为多个 segment，每个 segment 内使用 Chebyshev 多项式重建三维位置。根据 body 特性，模型分为几类：

1. **raw XYZ Chebyshev**：直接拟合三轴位置；
2. **fixed frame / apsis frame reference shape**：先用参考形状描述主要二维轨道，再拟合三轴 residual；
3. **Mercury / Moon clock correction**：对周期边界或相位使用专门 clock table 修正。

每个 segment 内重建流程可概括为：

```text
JD
  -> segment index
  -> normalized tau in expanded segment domain
  -> evaluate reference shape / residual Chebyshev
  -> frame unalignment if needed
  -> Cartesian position vector
```

## 4. 量化与打包

OPM 不直接存储 float64 Chebyshev 系数，而是使用按 degree 的 quantization step 将 residual coefficients 映射为整数：

```text
qcoeff = round(coeff / quant_step)
coeff  = qcoeff * quant_step
```

随后统计所有 segment 中每个 axis/degree 所需 zigzag 位宽，并按轴打包。这样可以保留简单、快速的解码逻辑，同时显著降低 payload 体积。

当前 polish 阶段遵守 no-size-increase 原则：局部调整整数 qcoeff 时，不允许超过原始全局 width table，因此不会增加文件体积。

## 5. 生产生成路线

当前生产路线固定为：

```text
600 年分片 + --polish + --validate
```

更具体地说：

```text
.raw/*.opm
  -> 一次 robust polish
  -> final *.opm
```

原则：

- 输入必须是 raw OPM；
- polish 只做一轮生产路径；
- 不对 old polished artifact 继续 polish；
- 不做 `min(raw, polished)` 或多候选人工挑选；
- 当前目标是稳定、可复现、整体 tail 明显改善，而不是对每个 shard 搜索数学最优。

## 6. 按 body 路由的 polish 策略

`generate_range.py --polish` 内部根据 body 类型路由到不同 optimizer。

### 6.1 Sun

Sun 使用 km error metric：

```text
Sun raw OPM
  -> optimize_opm_global_tail.py --error-metric km
  -> Sun polished OPM
```

原因是 `SSB -> Sun` 的角误差度量在几何上不稳定，不适合作为主要优化目标。

### 6.2 Mercury / Venus / Moon

Mercury、Venus、Moon 使用 native relative vector angular metric：

```text
Sun -> Mercury
Sun -> Venus
Earth -> Moon
```

当前生成脚本默认使用 native guarded/refined pmax polish：在保持 native metric 不变的前提下，复用 SSB pmax optimizer 中更强的取点、guard 与 peak refinement 机制。

候选升级路线为：

```text
native metric
+ cheb-center-uniform-endpoints active grid
+ refine-peaks = 3
+ shifted guard grid
+ endpoint-band guard nodes
+ capped_lex_guarded_ceiling
+ pmax_first
```

600 年 J2000 shard 的 dense validation 结果显示，native guarded/refined pmax polish 在 Mercury、Venus、Moon 上均降低了 p99 与 pmax；代价是 p50 略升，Mercury/Venus 的 p95 也有极小上升。由于当前生产目标更重视 worst-case 与 tail，初步结论是 native bodies 可以采用 strong native guarded/refined polish 作为后续默认路线。

2048 nodes/segment dense validation 对比如下：

| Body | Variant | p50 arcsec | p95 arcsec | p99 arcsec | pmax arcsec |
|---|---|---:|---:|---:|---:|
| Mercury | raw | 0.000206128584 | 0.000402412448 | 0.000500330603 | 0.000773990804 |
| Mercury | current native | 0.000233756373 | 0.000352030910 | 0.000390402806 | 0.000534298092 |
| Mercury | strong native guarded | 0.000238036422 | 0.000354586633 | 0.000388594958 | 0.000515211355 |
| Venus | raw | 0.000189601188 | 0.000345478393 | 0.000417035045 | 0.000611725244 |
| Venus | current native | 0.000213923137 | 0.000319972510 | 0.000363224329 | 0.000544429024 |
| Venus | strong native guarded | 0.000221232437 | 0.000321904401 | 0.000350862583 | 0.000430904066 |
| Moon | raw | 0.000227957727 | 0.000414730085 | 0.000500911459 | 0.000860586859 |
| Moon | current native | 0.000255803528 | 0.000381630886 | 0.000429923905 | 0.000642295686 |
| Moon | strong native guarded | 0.000262290292 | 0.000380800801 | 0.000414210210 | 0.000522351993 |

### 6.3 SSB-centered bodies with polished Sun anchor

EMB、Mars、Jupiter、Saturn、Uranus、Neptune、Pluto 的文件存储向量为：

```text
SSB -> Body
```

但 polish 目标使用 heliocentric composite angular error：

```text
(Body_OPM - Sun_OPM) vs (Body_DE441 - Sun_DE441)
```

其中 `Sun_OPM` 是已 polish 的 Sun OPM。这样优化目标更接近实际使用中的 heliocentric 方向误差。

当前收敛参数为：

```text
nodes-per-segment = 32
node-grid         = cheb-center-uniform-endpoints
refine-peaks      = 3
guard-grid        = shifted
objective         = capped_lex_guarded_ceiling
pmax-cap          = 0.00070 arcsec
accept-policy     = pmax_first
```

`pmax-cap` 是 writer-side soft target / brake line，不是文件格式参数，也不是硬保证。当当前 pmax 高于 cap 时，优化器优先压低 max；当 pmax 已低于 cap 时，优化器转向保护 topK、p99 和整体 tail 稳定性。

## 7. Guarded/refined pmax optimizer

当前 SSB polish 的核心思想是用更丰富的取点集合近似 segment 内 worst-case error，而不是只依赖单一 Chebyshev grid。

active scoring grid：

```text
Chebyshev nodes
+ center-dense nodes
+ uniform nodes
+ segment endpoints
```

guard grid：

```text
shifted center-dense nodes
+ shifted uniform nodes
+ endpoint-near nodes
+ endpoint-band nodes
+ shifted endpoint-band nodes
```

此外，`refine-peaks = 3` 会对当前 active grid 上发现的局部峰值做一维局部最大值搜索，并把 refined peak JD 加入评分集合。

优化器按 segment 处理 qcoeff 的小步长整数扰动，尝试 `±1` 变化，在不增加 width table 的前提下接受能改善局部 pmax/tail 且不触发 guard regression 的 candidate。

## 8. 验证方法

OPM 的验证分为确定性 dense grid validation 与随机 JD benchmark 两类。

### 8.1 Dense validation

Dense validation 按 segment 生成大量 Chebyshev nodes，并与 DE441 truth position 比较。输出指标包括：

```text
p50 / p95 / p99 / max
```

对 polish 实验，不能只看 optimizer 内部的 active/guard grid，需要再用更密的独立 validation grid 复扫，例如：

```text
nodes-per-segment = 1024
nodes-per-segment = 2048
```

### 8.2 Random JD benchmark

随机 JD benchmark 用于模拟运行时随机访问，比较 OPM reconstruction 与 DE441 的速度和误差分布。其作用是评估实际使用场景中的吞吐与延迟，不替代 dense validation，因为随机采样可能漏掉窄 pmax spike。

## 9. 当前结果占位

### 9.1 文件体积

| Body | Raw/Polished | Segments | Size | Notes |
|---|---:|---:|---:|---|
| Sun | TODO | TODO | TODO | km polish |
| Mercury | TODO | TODO | TODO | native angular |
| Venus | TODO | TODO | TODO | native angular |
| Moon | TODO | TODO | TODO | native angular |
| EMB | TODO | TODO | TODO | Sun-anchor composite |
| Mars | TODO | TODO | TODO | Sun-anchor composite |
| Jupiter | TODO | TODO | TODO | Sun-anchor composite |
| Saturn | TODO | TODO | TODO | Sun-anchor composite |
| Uranus | TODO | TODO | TODO | Sun-anchor composite |
| Neptune | TODO | TODO | TODO | Sun-anchor composite |
| Pluto | TODO | TODO | TODO | Sun-anchor composite |

### 9.2 Dense validation accuracy

| Body | Metric | p50 | p95 | p99 | pmax | Validation nodes |
|---|---|---:|---:|---:|---:|---:|
| Sun | km | TODO | TODO | TODO | TODO | TODO |
| Mercury | arcsec | TODO | TODO | TODO | TODO | TODO |
| Venus | arcsec | TODO | TODO | TODO | TODO | TODO |
| Moon | arcsec | TODO | TODO | TODO | TODO | TODO |
| Mars | composite arcsec | TODO | TODO | TODO | TODO | TODO |
| Jupiter | composite arcsec | TODO | TODO | TODO | TODO | TODO |
| Saturn | composite arcsec | TODO | TODO | TODO | TODO | TODO |
| Uranus | composite arcsec | TODO | TODO | TODO | TODO | TODO |
| Neptune | composite arcsec | TODO | TODO | TODO | TODO | TODO |
| Pluto | composite arcsec | TODO | TODO | TODO | TODO | TODO |

### 9.3 Runtime benchmark

| Body | Samples | OPM us/eval | DE441 us/eval | Speedup | Notes |
|---|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## 10. 工程取舍

OPM 的重点是工程稳定性，而不是对单个 shard 的局部最优追逐。实际 tuning 中存在某些 raw_ep 或实验 candidate 在个别 body/century 上偶然优于 guarded polish 的情况，但这不代表全局路线应该切换。当前采用的策略是：

- 与 true raw 相比必须显著改善；
- 关注 dense validation 下的 worst-case 与 tail；
- 避免每个 body、每个 shard 单独死循环调参；
- 使用固定默认参数保证可复现；
- 将历史实验脚本归档到 `legacy/`，active tree 只保留当前生产路径。

## 11. 局限与后续工作

后续需要补齐：

1. 完整 600 年分片的最终 dense validation 表格；
2. Mercury/Venus/Moon native guarded/refined optimizer 的 full-matrix 结果；
3. OPM 与 DE441 的随机访问速度对比；
4. 多 shard 边界附近的误差行为统计；
5. 长期 API 与直接 reader 的封装；
6. 若需要发布旧产物，可使用 legacy byte-level migration，但正式路线应重新生成 OPM1。

## 12. 结论

OPM1 提供了一条从高精度 DE441 星历到紧凑、可随机访问、可校验分发文件的工程路径。当前生产路线已经收敛为 600 年分片、raw-only 生成、一次 robust polish 与 dense validation。SSB-centered bodies 使用 polished Sun anchor composite metric 与 guarded/refined pmax optimizer；inner native bodies 保持 native metric，并正在验证是否采用同一套更强 optimizer 机制。

本文当前为占位稿，后续将用完整 dense validation 与 benchmark 数据替换 TODO 表格。