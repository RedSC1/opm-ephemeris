# OPM 生成方法说明（中文草稿）

> 这是给论文 / demo 代码用的中文技术说明草稿。重点解释算法思想和生成流程，而不是逐字描述文件格式字段。后面可以据此翻译成英文版。

## 1. 目标

OPM（Planetary Ephemeris File）这个 demo 的目标是：

1. 从 DE441 这样的高精度 NASA BSP 星历中抽取行星 / 月亮位置；
2. 用比较小的自定义二进制文件保存这些位置；
3. 保持读取端足够简单、快速；
4. 精度控制在约 `0.001 arcsec` 以内；
5. 文件体积比直接保存高阶原始 Chebyshev 系数小很多。

核心思想不是“直接压缩 DE441”，而是：

```text
轨道的主要形状很规律，可以用一个低成本参考模型表达；
每个短段只保存相对参考模型的残差；
残差再用 Chebyshev 多项式 + 量化整数保存。
```

换句话说：

```text
位置 = 参考轨道形状 + 小残差
```

参考轨道形状负责吃掉大部分几何结构，残差负责补细节。

---

## 2. 文件覆盖范围和分片策略

OPM 文件本身只描述一个 Julian Date 覆盖范围：

```text
coverage_start_jd
coverage_span_days
```

“一个世纪一片”或者“600 年一片”不是格式要求，只是打包策略。

现在推荐的发布策略是：

```text
600 年一片 = 6 个 J2000 对齐世纪一片
```

原因：

1. full-range 一个大文件覆盖约 3 万年，会让全文件共享的 width table 被长期漂移拖大；
2. 单世纪文件局部性最好，但文件数量太多；
3. 600 年分片保留了局部 reference / width table 的优点，同时文件数量比单世纪少 6 倍；
4. 初步测试显示，600 年分片 + pmax polish 的体积和精度都接近单世纪，且明显比 full-range 大包更适合作为发布颗粒度。

### 2.1 为什么不能随便贴着 DE441 边界生成

用户请求的覆盖范围和生成器内部实际需要采样的范围不是一回事。

例如用户想生成：

```text
[A, B]
```

但是内部拟合可能需要：

```text
[A - margin, B + margin]
```

原因包括：

- Chebyshev 拟合节点会做 segment domain expansion；
- 水星、金星、月亮需要找到完整的近日点 / 近地点到下一次近日点 / 近地点的区间；
- 有些 frame / reference shape 需要边界外的完整轨道段。

所以 `generate_range.py` 默认使用 strict range safety：

```text
如果内部拟合需要的采样范围超出 BSP 覆盖，就提前报错。
```

这比生成到一半才出现 `Missing SPK coverage` 更清楚。

### 2.2 关于 reference shape 的可复现性

OPM 的 reference shape 不依赖外部解析星历或第三方压缩格式。writer 直接从源星历本身构造文件级 reference shape：

```text
DE441 samples
  -> local orbital / lunar apsis frame
  -> aligned segment Chebyshev coefficients
  -> average over segments
  -> file-level mean reference shape
  -> per-segment residual
```

这样 writer 是自包含的：给定同一个 BSP、同一个 body config 和同一个 coverage，就能确定地生成 reference shape、residual coefficients、quant table 和 bit-packed payload。

---

## 3. 不同天体的存储向量

OPM 不是所有天体都用同一种向量存储。当前 demo 大致分成几类。

### 3.1 Sun

Sun 存的是：

```text
SSB -> Sun
```

即太阳系质心到太阳的向量。

Sun 比较特殊，因为 `SSB -> Sun` 这个向量本身很短，如果用角秒误差评估，会出现病态放大。因此 Sun 的 pmax polish 用的是线性距离误差：

```text
km metric
```

而不是 angular metric。

### 3.2 Mercury / Venus

Mercury 和 Venus 存的是：

```text
Sun -> body
```

也就是日心向量。

这两个内行星使用基于近日点的轨道分段和旋转参考形状。

### 3.3 EMB / Mars / Jupiter / Saturn / Uranus / Neptune / Pluto

这些天体在文件中存的是：

```text
SSB -> body
```

但是很多最终用户场景更关心的是相对太阳的视方向，所以调 pmax 的时候不能只看：

```text
Body_opm vs Body_DE441
```

而应该看带 OPM Sun anchor 的 composite heliocentric metric：

```text
(Body_opm - Sun_opm) vs (Body_DE441 - Sun_DE441)
```

这样 Sun 的量化误差会被真实地纳入指标。

### 3.4 Moon

Moon 存的是：

```text
Earth -> Moon
```

Moon 用的是近地点到近地点的分段，以及月球专用的轨道平面 / apsis frame 模型。

---

## 4. Chebyshev 多项式表示

每个 segment 里的位置或者残差都用 Chebyshev 多项式表示。

对一个 segment：

```text
jd in [a, b]
```

先把时间归一化到：

```text
tau in [-1, 1]
```

如果启用了 expansion，则实际拟合域是：

```text
[a - f * (b-a), b + f * (b-a)]
```

然后在 Chebyshev 节点上采样真实位置，拟合：

```text
x(tau) = sum c_x[k] T_k(tau)
y(tau) = sum c_y[k] T_k(tau)
z(tau) = sum c_z[k] T_k(tau)
```

其中 `T_k` 是 Chebyshev 多项式。

Chebyshev 的好处：

1. 在固定区间内拟合稳定；
2. 给定 degree 后计算快；
3. segment 内随机访问很方便；
4. 读取端只需要保存少量系数并做多项式求值。

---

## 5. 最简单模型：raw_xyz_cheb

Sun 使用最直接的模型：

```text
raw_xyz_cheb
```

即每个固定时间 segment 直接拟合：

```text
SSB -> Sun 的 x/y/z
```

没有轨道旋转，没有 reference shape。

流程：

```text
DE441 真实位置
  -> 每段 Chebyshev 拟合
  -> 系数量化
  -> 打包
```

Sun 的量化单位是 km，pmax polish 也使用 km metric。

---

## 6. 轨道旋转 frame：为什么要旋转

行星轨道大致是一个平面里的椭圆。如果直接在惯性坐标系里拟合 `x/y/z`，很多变化其实只是：

```text
轨道平面方向
近日点方向
椭圆形状
```

混在三个坐标轴里，会让残差更大。

所以 demo 对很多天体使用一个局部轨道 frame。

对每个 segment，先从真实位置点拟合出一个最合适的轨道平面和 apsis 方向：

```text
plane_u
plane_v
apsis_angle
```

这里 `plane_u / plane_v` 是轨道平面法向量的 stereographic 参数，`apsis_angle` 是平面内的近日点 / 近地点方向角。

然后把惯性坐标下的位置旋转到这个局部轨道 frame：

```text
inertial xyz
  -> align_positions(...)
  -> local aligned xyz
```

在这个 aligned 坐标系里：

- `x/y` 主要描述轨道平面内的椭圆形状；
- `z` 通常变得很小；
- 不同 segment 的形状更接近，可以取平均 reference shape；
- 残差系数变小，量化后 bit width 也变小。

读取时反过来做：

```text
local aligned xyz
  -> unalign_positions(...)
  -> inertial xyz
```

这就是“轨道旋转还原”。

---

## 7. frame 参数的时间模型

每个 segment 都有自己的最佳 frame 参数：

```text
plane_u[i]
plane_v[i]
apsis_angle[i]
```

但是文件里不直接逐段保存这三个值，否则开销会变大。

当前 demo 用低阶时间模型描述这些 frame 参数随时间的变化。不同参数 / 不同 body 可以选择不同复杂度：有的用一阶 Chebyshev 就够，有的可以升到二阶；事件时间本身还可以叠加离散修正表，例如 Mercury 的 Cheb8 近日点修正和 Moon 的 century i16 近地点修正。

```text
plane_u(T)      ~= low-degree Chebyshev(T)
plane_v(T)      ~= low-degree Chebyshev(T)
apsis_angle(T)  ~= low-degree Chebyshev(T)
```

其中 `T` 是按 segment midpoint 在文件内部归一化出来的长期时间坐标。

也就是说，文件里保存的是一组平滑的 frame / phase 时间模型，而不是每段原样保存 frame。

生成时：

```text
每段找 best frame
  -> 对 frame 参数随时间拟合低阶时间模型
  -> 必要时叠加事件时间离散修正
  -> 用平滑后的 frame 参数重新对每段做 residual fit
```

读取时：

```text
根据 segment index 求 midpoint
  -> 计算 T
  -> eval 低阶时间模型得到 plane_u/plane_v/apsis_angle
  -> 应用事件时间修正后的 segment phase / tau
  -> 还原坐标
```

### 7.1 单 segment 特殊情况

任意 JD 小包可能只有一个 segment。此时 `T` 的归一化区间长度为 0，不能做普通 cheb1 拟合。

当前实现会把这种情况作为常量 frame：

```text
coeff = [value, 0]
```

这样小包也能正常生成。

---

## 8. reference shape：平均轨道形状

旋转到 aligned frame 后，每个 segment 的轨道形状会非常相似。

于是可以对每个 segment 的 aligned `x/y` 分别做较高阶 Chebyshev 拟合：

```text
aligned_x_i(tau) ~= sum a_i[k] T_k(tau)
aligned_y_i(tau) ~= sum b_i[k] T_k(tau)
```

然后对所有 segment 的 `x/y` 系数取平均，得到 reference shape：

```text
shape_x[k] = mean_i a_i[k]
shape_y[k] = mean_i b_i[k]
```

这个 `shape_x / shape_y` 是文件级别的平均轨道形状。

后续每段不再直接保存完整 aligned x/y，而是保存：

```text
aligned_x_i(tau) - shape_x(tau)
aligned_y_i(tau) - shape_y(tau)
aligned_z_i(tau)
```

也就是：

```text
残差 = 当前 segment 的真实形状 - 文件平均 reference shape
```

读取时再加回来：

```text
aligned_x = shape_x(tau) + residual_x(tau)
aligned_y = shape_y(tau) + residual_y(tau)
aligned_z = residual_z(tau)
```

这一步非常关键。它让大部分“轨道长得像一个椭圆”的信息只存一次，逐段 payload 只需要存小残差。

---

## 9. 残差 Chebyshev 拟合

对每个 segment，生成器会拟合残差多项式：

```text
residual_x(tau) ~= sum r_x[k] T_k(tau)
residual_y(tau) ~= sum r_y[k] T_k(tau)
residual_z(tau) ~= sum r_z[k] T_k(tau)
```

不同 body 有不同的 residual degree。例如：

- Mercury / Venus / Moon：短周期、段数多；
- Mars / outer planets：段更长；
- Uranus / Neptune / Pluto 为了压低 600 年分片的 tail，当前 demo 提高到了 `residual_degree = 30`。

残差系数越小，量化后整数越小，最终 bit width 越小。

---

## 10. 系数量化

残差 Chebyshev 系数是浮点数，不能直接高效保存。OPM 的量化不是对位置样本做 min/max 映射，而是对 **Chebyshev residual coefficient** 做物理单位量化。

对每个 segment、每个 axis、每个 degree：

```text
q[axis, k, seg] = round(coeff[axis, k, seg] / step[k])
coeff_recon[axis, k, seg] = q[axis, k, seg] * step[k]
```

这里的 `step[k]` 只随 degree `k` 变化，不随 segment 变化；当前实现里也不随 axis 变化。也就是说：

```text
同一个 degree 的 x/y/z 和所有 segment 共用一个 quant step；
不同 degree 可以有不同 quant step。
```

这样做的原因是，不同阶 Chebyshev coefficient 对误差和 bit width 的影响不一样。低阶 coefficient 往往影响整体形状，高阶 coefficient 更多影响局部细节，所以有些 body 会让高阶 step 稍微变粗，以减少整数幅度和 bit width。

### 10.1 degree-dependent quant schedule

OPM 用一个 `base` 和一个 `pattern` 生成每一阶的量化步长。

设：

```text
degree = residual_degree
x_k = k / degree,  k = 0..degree
```

当前支持三种 pattern。

#### flat

```text
step[k] = base
```

所有 degree 使用同一个量化步长。

#### linear:a

```text
step[k] = base * (1 + a * k / degree)
```

从低阶到高阶线性变粗。最高阶是：

```text
step[degree] = base * (1 + a)
```

例子：

```text
Mercury 0.032 linear:0.65
最低阶 0.032 km，最高阶 0.032 * 1.65 = 0.0528 km
```

#### growth:a

```text
step[k] = base * a^(k / degree)
```

从低阶到高阶按几何曲线平滑变粗。注意因为指数是 `k / degree`，所以最高阶不是 `base * a^degree`，而是：

```text
step[degree] = base * a
```

例子：

```text
Jupiter 0.5 growth:1.25
最低阶 0.5 km，最高阶 0.625 km
```

### 10.2 当前 generator 推荐参数

这些是 `opm_demo/body_configs.py` 里的当前推荐 demo 生成参数。full-range tuning 文档里的某些最终 artifact 可能有额外实验选择，写论文 / demo 时需要区分“生成器默认推荐参数”和“某次最终调参产物”。

| Body | residual_degree | Quant base | Pattern | Lowest step | Highest step | Notes |
|---|---:|---:|---|---:|---:|---|
| Sun | 25 | `0.01 km` | `flat` | `0.01 km` | `0.01 km` | SSB->Sun anchor；Sun 自身用 km metric 调 pmax。 |
| Mercury | 24 | `0.032 km` | `linear:0.65` | `0.032 km` | `0.0528 km` | 高阶稍粗，减小 bit width；Mercury 另有 Cheb8 事件时间修正。 |
| Venus | 24 | `0.06 km` | `flat` | `0.06 km` | `0.06 km` | Sun->Venus native vector。 |
| EMB | 28 | `0.02 km` | `growth:1.25` | `0.02 km` | `0.025 km` | SSB->EMB；最终用户指标通常看 EMB-Sun composite。 |
| Mars | 28 | `0.04 km` | `flat` | `0.04 km` | `0.04 km` | SSB->Mars；full-range final 曾选择 d30 以降低 tail。 |
| Jupiter | 24 | `0.5 km` | `growth:1.25` | `0.5 km` | `0.625 km` | 外行星 residual 幅度大，km step 也更大。 |
| Saturn | 24 | `1.0 km` | `growth:1.25` | `1.0 km` | `1.25 km` | 外行星；使用 Sun-anchor composite polish。 |
| Uranus | 30 | `1.6 km` | `linear:0.5` | `1.6 km` | `2.4 km` | 600 年分片中 d30 用来压低 outer-planet tail。 |
| Neptune | 30 | `3.5 km` | `flat` | `3.5 km` | `3.5 km` | d30/q3.5 让 600 年 tail 约落到 0.0005 arcsec 附近。 |
| Pluto | 30 | `3.5 km` | `growth:1.25` | `3.5 km` | `4.375 km` | 600 年分片候选；full-range final 文档里另有 d28/q4.0 选择。 |
| Moon | 24 | `0.00025 km` | `flat` | `0.00025 km` | `0.00025 km` | Earth->Moon；Moon 对 km step 极敏感。 |

### 10.3 和 width table 的关系

量化越粗：

```text
step[k] 更大
q = round(coeff / step[k]) 更小
需要的 bit width 更小
文件更小
误差更大
```

量化越细：

```text
step[k] 更小
q 整数幅度更大
需要的 bit width 更大
文件更大
误差更小
```

量化完成后，OPM 不把整数统一存成 `int16` 或 `int32`，而是统计整个文件所有 segment 的：

```text
axis × degree
```

所需 bit width。因此 quant schedule 和 width table 是一起工作的：

```text
reference shape 让 coeff 变小；
quant step 控制 qcoeff 的整数幅度；
width table 只给每个 axis × degree 分配必要 bits；
pmax polish 在不增大这些 bit width 的前提下微调整数尾部误差。
```

---

## 11. width table 和 bit packing

OPM payload 不是把每个量化整数都用固定 32-bit 保存。

生成器会统计全文件所有 segment 的量化残差整数，得到每个：

```text
axis × degree
```

需要的最小 bit width。

例如：

```text
x degree 0 需要 18 bits
x degree 1 需要 20 bits
...
y degree 0 需要 19 bits
...
```

这个表就是 width table。

然后 payload 按 axis-major / degree-major 的方式紧密 bit-pack：

```text
axis -> degree -> segment
```

这样读取时可以按固定 width 快速解包。

### 11.1 为什么 full-range 文件会变胖

width table 是“一个文件内共享”的。

如果一个 full-range 文件覆盖 3 万年，只要某个年代某个 residual coefficient 特别大，那么整个文件所有 segment 都要按那个最大 bit width 付成本。

这就是 full-range 大包变胖的主要原因之一。

600 年分片可以避免这个问题：

```text
每 600 年有自己的 reference shape 和 width table
```

长期漂移不会污染整个 3 万年文件。

---

## 12. 不同模型类型

当前 demo 主要有四种模型。

### 12.1 raw_xyz_cheb

用于 Sun。

```text
直接保存 SSB->Sun 的 xyz Chebyshev 系数
```

没有 reference shape，没有 frame。

### 12.2 mean_apsis_frame_shape

用于 Mercury / Venus。

```text
Sun->body
近日点到近日点分段
轨道 frame 对齐
mean x/y reference shape
残差 Chebyshev
```

Mercury 还使用了持久化的 Cheb8 事件时间修正表，让全局 mean anomalistic clock 更贴近日点事件。

### 12.3 mean_lunar_apsis_frame_shape

用于 Moon。

```text
Earth->Moon
近地点到近地点分段
月球轨道 frame 对齐
mean x/y reference shape
残差 Chebyshev
```

Moon 使用 century i16 线性表修正近地点事件时间，避免每个文件都重新保存大量 perigee boundary。

### 12.4 fixed_frame_shape

用于 EMB / Mars / Jupiter / Saturn / Uranus / Neptune / Pluto。

这些 body 使用固定时间长度或者固定周期 segment，而不是每段重新搜索事件边界。

仍然会：

```text
每段拟合 best frame
拟合 cheb1 frame time model
生成 mean x/y reference shape
保存 residual Chebyshev
```

---

## 13. pmax polish：写入端尾部误差修正

生成 raw OPM 后，文件已经能通过验证，但 max error 可能有孤立尖峰。

pmax polish 的目标是：

```text
不改变模型结构
不改变 quant step
不增加 width table
只微调每段的整数 qcoeff
压低 p99 / max tail
```

也就是说，polish 是 writer-side 的整数微调：

```text
qcoeff[axis, degree] += -1 / 0 / +1 / ...
```

每次候选调整都要满足：

1. 当前 segment 的 tail error 变好；
2. 不超过原来的 bit width 限制；
3. 不破坏全局 p99 budget；
4. 对 SSB bodies，要用 composite Sun-anchor 指标。

### 13.1 Sun polish

Sun 用：

```text
optimize_opm_global_tail.py --error-metric km
```

因为 `SSB -> Sun` 角秒误差会病态。

### 13.2 native bodies polish

Mercury / Venus / Moon 使用 native angular metric：

```text
OPM vector vs DE441 vector
```

因为它们的存储向量本身就是最终关心的相对向量：

```text
Sun->Mercury
Sun->Venus
Earth->Moon
```

但 optimizer 不再使用早期较简单的 global-tail Cheb-only 路线，而是使用 native 版 guarded/refined pmax polish：

```text
tools/optimize_opm_native_guarded_pmax.py
```

即保持 native metric 不变，但采用与 SSB pmax optimizer 同源的强取点策略：

```text
active grid   = cheb-center-uniform-endpoints
refine peaks  = 3
guard grid    = shifted，并包含 endpoint-band guard nodes
objective     = capped_lex_guarded_ceiling
pmax_cap      = 0.00070 arcsec
accept policy = pmax_first
```

600 年 J2000 shard 的 2048 nodes/segment dense validation 显示，这条路线相对旧 native polish 进一步降低了 Mercury / Venus / Moon 的 p99 和 pmax，代价是 p50 或 p95 有轻微上升。当前生产目标优先保护 tail / worst-case，因此 native bodies 默认切换到这条 strong native polish 路线。

### 13.3 SSB bodies polish

EMB / Mars / Jupiter / Saturn / Uranus / Neptune / Pluto 用 composite metric：

```text
(Body_opm - Sun_opm) vs (Body_DE441 - Sun_DE441)
```

其中 `Sun_opm` 必须是已经 polish 过的 Sun 文件。

这一步很重要，因为否则 body 自己看起来误差小，不代表和 OPM Sun 相减后的用户-facing 方向误差小。

---

## 14. `generate_range.py --polish` 的正式流程

现在推荐使用：

```bash
python3 generate_range.py \
  --de441 /path/to/de441.bsp \
  --all \
  --jd-start 2451545 \
  --days 219150 \
  --output-root out/opm600/c+0000 \
  --polish \
  --validate
```

其中：

```text
219150 days = 6 * 36525 days = 600 years
```

加 `--polish` 后，内部流程是：

```text
1. 生成 raw OPM 到 <output-root>/.raw/
2. polish Sun -> <output-root>/sun.opm
3. polish Mercury / Venus / Moon -> <output-root>/...
4. polish SSB bodies with polished Sun anchor -> <output-root>/...
5. validate polished output
6. 日志写入 <output-root>/logs/
```

所以正式用户不需要手动跑一堆 optimizer。

底层工具仍然保留，方便实验调参：

```text
tools/optimize_opm_global_tail.py
tools/optimize_opm_native_guarded_pmax.py
tools/optimize_opm_ssb_sun_anchor_pmax.py
```

---

## 15. 当前推荐的 600 年分片参数

J2000 附近 600 年样本测试中，`--polish` 后外三颗曾经略微冒头：

```text
Uranus / Neptune / Pluto p99,max 接近 0.0006~0.0007 arcsec
```

因此当前 demo 把外三颗调成：

```text
Uranus:  residual_degree=30, quant=1.6 linear:0.5
Neptune: residual_degree=30, quant=3.5 flat
Pluto:   residual_degree=30, quant=3.5 growth:1.25
```

这样 600 年分片 + polish 后大致为：

```text
Uranus   p99 ~0.000421, max ~0.000456
Neptune  p99 ~0.000469, max ~0.000498
Pluto    p99 ~0.000474, max ~0.000527
```

体积成本每 600 年分片只有约 `0.5 KiB`，基本可以忽略。

### 15.1 当前收敛的 SSB polish 方案

目前不再继续比较 `0.00065` / `0.00060` 这类 pmax cap 微调。当前收敛目标不是逐个 shard 搜数学最优，而是保持一条稳定、可复现、raw-only 的生产路径：

```text
.raw/*.opm
  -> 一次 robust polish
  -> final *.opm
```

SSB-centered bodies 的当前推荐 polish 参数是：

```text
active grid   = cheb-center-uniform-endpoints
refine peaks  = 3
guard grid    = shifted，并包含 endpoint-band guard nodes
objective     = capped_lex_guarded_ceiling
pmax_cap      = 0.00070 arcsec
accept policy = pmax_first
```

其中 `pmax_cap = 0.00070` 只是 writer-side 搜索策略里的 soft target / 刹车线：

```text
高于 0.00070 arcsec 时，优先压局部 max；
低于 0.00070 arcsec 后，不再死磕单点 max，优先保护 topK / p99 / 整体 tail。
```

它不是 OPM 文件格式字段，也不是硬保证。个别 shard 上某个 raw 结果偶然比 polished 结果更低，不作为失败条件；正式验收看的是相对 `.raw` 是否整体明显改善、全矩阵 worst-case 是否稳定、以及是否保持单路径不爆炸。

因此当前推荐发布形态仍然是：

```text
600 年分片 + --polish + --validate
```

不要在正式流程中做 `min(raw, polished)` 或多候选选择；这会把调参复杂度带回生产路径。

---

## 16. 读取端重建公式

读取一个时间 `jd` 时，大致流程是：

1. 根据 header coverage 检查 `jd` 是否在文件范围内；
2. 根据 segment addressing 找到 segment index；
3. 解包该 segment 的 qcoeff；
4. 用 quant step 还原 Chebyshev residual coeff；
5. 计算 segment 内归一化时间 `tau`；
6. 如果有 reference shape：

   ```text
   aligned_x = shape_x(tau) + residual_x(tau)
   aligned_y = shape_y(tau) + residual_y(tau)
   aligned_z = residual_z(tau)
   ```

7. 如果有 frame model：

   ```text
   根据 segment midpoint 算 frame 参数
   unalign_positions(aligned_xyz, frame_params)
   ```

8. 得到文件存储向量，例如：

   ```text
   SSB->body
   Sun->body
   Earth->Moon
   ```

上层 apparent pipeline 再根据需要组合这些向量。

---

## 17. 为什么这个方法适合 demo / paper

这个 demo 的重点不是发明最复杂的压缩算法，而是展示一种清晰、可解释、可快速读取的星历表示方法：

```text
物理/几何先验 + Chebyshev 残差 + 整数量化 + writer-side tail polish
```

优点：

1. 轨道旋转和 reference shape 有明确几何意义；
2. Chebyshev 残差容易验证和实现；
3. 读取端不需要复杂优化，只做解包、多项式求值、旋转还原；
4. pmax polish 是写入端离线步骤，不增加读取端复杂度；
5. 600 年分片让长期漂移局部化，避免 full-range width table 被极端年代拖大。

---

## 18. 附录：full-range selected 大包的随机 JD 抽样参考

除了 per-segment deterministic validation，还可以做一个随机 JD 抽样 benchmark，作为实际随机访问场景下的 sanity check。

这个测试不是 worst-case 有效性证明。它只表示：

```text
在随机抽到的这些浮点 JD 上，OPM 和 DE441 的差异是多少。
```

很窄的 pmax spike 可能被随机采样漏掉；反过来，随机采样也可能踩到固定 validation nodes 没踩到的中间 tail。因此它适合放在附录或补充材料中，和正式 validation 互补。

### 18.1 测试对象

这次抽样测试用的是 full-range selected tuning artifacts，也就是当前最终选择的大包产物：

```text
out/body-packed/tuning/sun/q010-pmax/sun.opm
out/body-packed/tuning/mercury/d26-q032-global-tail-full/mercury.opm
out/body-packed/tuning/venus/q08-global-tail-full/venus.opm
out/body-packed/tuning/emb/sun-q010-pmax-auto-revisit/emb.opm
out/body-packed/tuning/mars/sun-q010-pmax-d30-auto-revisit-slack1e-6/mars.opm
out/body-packed/tuning/jupiter/sun-q010-pmax-auto-revisit-slack1e-6/jupiter.opm
out/body-packed/tuning/saturn/sun-q010-pmax-auto-revisit-slack1e-6/saturn.opm
out/body-packed/tuning/uranus/sun-q010-pmax-auto-revisit-slack1e-6/uranus.opm
out/body-packed/tuning/neptune/sun-q010-pmax-auto-revisit-slack1e-6/neptune.opm
out/body-packed/tuning/pluto/sun-q010-pmax-auto-revisit-slack1e-6/pluto.opm
out/body-packed/tuning/moon/q034-global-tail-full/moon.opm
```

为了避免 `out/body-packed/tuning/` 里多个候选文件互相重叠，测试时先把最终选择的 11 个文件放到一个临时 root：

```text
/tmp/opm-final-selected-big
```

再运行：

```bash
python3 benchmark_random_jd.py \
  --de441 /path/to/de441.bsp \
  --opm-root /tmp/opm-final-selected-big \
  --samples 10000 \
  --seed 1 \
  --timing-repeats 1 \
  --csv /tmp/opm-final-selected-big/random-final-selected.csv
```

采样方式是 stratified random JD sampling：在各 body 的 coverage 内分层随机抽取 `10000` 个浮点 JD。

### 18.2 指标定义

- Sun 使用线性距离误差：

  ```text
  km
  ```

- Mercury / Venus / Moon 使用 native angular metric：

  ```text
  OPM vector vs DE441 vector
  ```

- EMB / Mars / Jupiter / Saturn / Uranus / Neptune / Pluto 使用带 OPM Sun anchor 的 composite metric：

  ```text
  (Body_opm - Sun_opm) vs (Body_DE441 - Sun_DE441)
  ```

表中的 `random max` 只是这 `10000` 个随机样本里的最大值，不是全局连续时间 maximum。

### 18.3 full-range selected 大包随机抽样结果

| Body | Metric | random p50 | random p95 | random p99 | random max |
|---|---|---:|---:|---:|---:|
| Sun | km | 0.0201155 | 0.0308991 | 0.0352645 | 0.0448217 |
| EMB | composite arcsec | 0.000108418 | 0.000338257 | 0.000452040 | 0.000624999 |
| Jupiter | composite arcsec | 0.000254565 | 0.000411848 | 0.000482011 | 0.000710424 |
| Mars | composite arcsec | 0.000143911 | 0.000289325 | 0.000379403 | 0.000733972 |
| Mercury | native arcsec | 0.000272407 | 0.000415335 | 0.000468880 | 0.000592435 |
| Moon | native arcsec | 0.000338888 | 0.000504031 | 0.000567263 | 0.000685853 |
| Neptune | composite arcsec | 0.000283893 | 0.000427602 | 0.000487882 | 0.000607437 |
| Pluto | composite arcsec | 0.000247191 | 0.000447412 | 0.000529021 | 0.000726026 |
| Saturn | composite arcsec | 0.000223883 | 0.000397103 | 0.000481278 | 0.000701157 |
| Uranus | composite arcsec | 0.000263129 | 0.000417596 | 0.000488629 | 0.000872971 |
| Venus | native arcsec | 0.000322337 | 0.000481749 | 0.000548705 | 0.000691978 |

大致结论：

```text
full-range selected 大包在随机 10000 JD 抽样下，
大多数 body 的 random p99 约为 0.00038~0.00057 arcsec，
random max 仍低于 0.001 arcsec。
```

外行星中 Uranus / Pluto / Saturn 的 random max 相对更高，这说明 full-range 单文件的 isolated tail 仍然比 600 年分片更难压。原因是 full-range 文件用一个 reference shape 和一个 width table 覆盖接近整个 DE441 长跨度，长期漂移和极端年代会更容易污染 tail。

因此这组数据更适合作为：

```text
正式 full-range selected 大包的随机访问参考数据
```

而不是：

```text
推荐发布形态的最终 worst-case 证明
```

正式推荐的发布形态仍然是：

```text
600 年分片 + --polish + --validate
```

速度字段暂时不作为正式结论。当前 benchmark 调用的是 validator 里的重建函数；它会扫描 segment，不能代表未来 direct random-access reader 的真实性能。正式速度 benchmark 应在独立 reader 实现后重新测。

---

## 19. 一句话总结

OPM demo 的核心可以概括为：

```text
先把轨道转到更自然的局部轨道坐标系，
用文件级平均 Chebyshev reference shape 表示主要轨道形状，
每段只保存小的 Chebyshev 残差，
残差量化成整数并按最小 bit width 打包，
最后在不增大 width 的前提下微调整数系数压低 pmax。
```

推荐发布形态是：

```text
600 年分片 + --polish + --validate
```

这样兼顾：

```text
体积、精度、文件数量、读取复杂度
```
