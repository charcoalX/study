"""
h5py 三项对比实验
=====================================
1: chunks=(1,H,W) vs chunks=(10,H,W) — 随机访问性能
2: compression="lzf" vs compression="gzip" — 文件大小与速度
3: 在 create_streaming_hdf5 中加入 exposure_time dataset

运行方式: python h5py_exercises.py
"""

import h5py
import numpy as np
import time
from pathlib import Path

out_dir = Path("output_exercises")
out_dir.mkdir(exist_ok=True)

H, W = 256, 256
N_FRAMES = 20

SEP = "=" * 60

# =============================================================================
#  1: chunks=(1,H,W) vs chunks=(10,H,W)
# 重点：chunk 是 HDF5 读取和压缩的最小单元
#       读单帧时，必须把整个 chunk 解压，多余的帧直接丢弃
# =============================================================================

def exercise_1():
    print(f"\n{SEP}")
    print(" 1: chunks 大小对随机访问性能的影响")
    print(SEP)

    results = {}

    for chunk_size in [1, 10]:
        fpath = str(out_dir / f"ex1_chunks_{chunk_size}.h5")

        # --- 写入 ---
        t0 = time.time()
        with h5py.File(fpath, "w") as f:
            ds = f.create_dataset(
                "data",
                shape=(0, H, W),
                maxshape=(None, H, W),
                dtype=np.float32,
                chunks=(chunk_size, H, W),   # ← 唯一变化
                compression="gzip",
                shuffle=True,
            )
            for i in range(N_FRAMES):
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = np.random.poisson(50 + i * 10, size=(H, W)).astype(np.float32)
        write_time = time.time() - t0
        file_size_kb = Path(fpath).stat().st_size / 1024

        # --- 随机访问第 13 帧（重复 50 次取均值）---
        with h5py.File(fpath, "r") as f:
            t0 = time.time()
            for _ in range(50):
                _ = f["data"][13][()]
            rand_read_ms = (time.time() - t0) / 50 * 1000

        results[chunk_size] = {
            "write_time": write_time,
            "file_size_kb": file_size_kb,
            "rand_read_ms": rand_read_ms,
        }

        print(f"\n  chunks=({chunk_size}, H, W):")
        print(f"    写入时间 (20帧) : {write_time:.3f} s")
        print(f"    文件大小        : {file_size_kb:.1f} KB")
        print(f"    单帧随机读取    : {rand_read_ms:.3f} ms")

    # --- 对比 ---
    r1 = results[1]["rand_read_ms"]
    r10 = results[10]["rand_read_ms"]
    ratio = r10 / r1
    print(f"\n  【结论】")
    print(f"    chunks=10 的随机读取比 chunks=1 慢 {ratio:.1f}x")
    print(f"    原因：读第13帧时 chunks=10 必须解压包含它的整个10帧 chunk，")
    print(f"          但只使用其中1帧，其余9帧的解压工作全部浪费。")
    print(f"    适用场景：逐帧访问 → chunks=1；批量顺序读取 → chunks=10")


# =============================================================================
#  2: compression="lzf" vs compression="gzip"
# 重点：不同算法在压缩率和速度上的权衡
#       Poisson 噪声（高熵）对 lzf 不友好，对 gzip 影响较小
# =============================================================================

def exercise_2():
    print(f"\n{SEP}")
    print(" 2: lzf vs gzip vs 无压缩")
    print(SEP)

    # 检查 lzf 是否可用
    lzf_available = True
    try:
        with h5py.File(str(out_dir / "_test_lzf.h5"), "w") as f:
            f.create_dataset("t", data=np.ones((4, 4)), compression="lzf")
    except Exception:
        lzf_available = False

    compressions = []
    if lzf_available:
        compressions.append("lzf")
    else:
        print("  注意: lzf 插件未安装，跳过 lzf 测试（pip install hdf5plugin）")
    compressions += ["gzip", None]

    results = {}

    for comp in compressions:
        label = comp if comp else "none"
        fpath = str(out_dir / f"ex2_comp_{label}.h5")

        kwargs = {"chunks": (1, H, W), "shuffle": True}
        if comp:
            kwargs["compression"] = comp

        # 写入
        t0 = time.time()
        with h5py.File(fpath, "w") as f:
            ds = f.create_dataset(
                "data",
                shape=(0, H, W),
                maxshape=(None, H, W),
                dtype=np.float32,
                **kwargs,
            )
            for i in range(N_FRAMES):
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = np.random.poisson(50 + i * 10, size=(H, W)).astype(np.float32)
        write_time = time.time() - t0
        file_size_kb = Path(fpath).stat().st_size / 1024

        # 顺序读取全部帧
        with h5py.File(fpath, "r") as f:
            t0 = time.time()
            for i in range(N_FRAMES):
                _ = f["data"][i][()]
            read_ms = (time.time() - t0) / N_FRAMES * 1000

        results[label] = {
            "write_time": write_time,
            "file_size_kb": file_size_kb,
            "read_ms": read_ms,
        }

        print(f"\n  compression='{label}':")
        print(f"    写入时间  : {write_time:.3f} s")
        print(f"    文件大小  : {file_size_kb:.1f} KB")
        print(f"    读取速度  : {read_ms:.3f} ms / 帧")

    # 无压缩作为基准对比
    base_size = results["none"]["file_size_kb"]
    print(f"\n  【结论】（基准：无压缩 = {base_size:.0f} KB）")
    for label in [c if c else "none" for c in compressions]:
        if label == "none":
            continue
        ratio = base_size / results[label]["file_size_kb"]
        print(f"    {label:5s} 压缩率: {ratio:.1f}x  |  读取开销: +{results[label]['read_ms'] - results['none']['read_ms']:.2f} ms/帧")
    print()
    print(f"    gzip 在随机（高熵）数据上通常比 lzf 压得更小，")
    print(f"    因为 gzip(deflate) 含 Huffman 编码，能处理高熵序列；")
    print(f"    lzf 只做 LZ 字节匹配，随机数据几乎找不到重复串。")
    print(f"    真实探测器图像（有大量接近零的背景）lzf 速度优势才会显现。")


# =============================================================================
#  3: 在流式写入中加入 exposure_time dataset
# 重点：与 frame_number、timestamp 完全相同的三步模式
#       create(shape=(0,)) → resize(+1) → ds[-1] = value
# =============================================================================

def exercise_3():
    print(f"\n{SEP}")
    print(" 3: 新增 exposure_time dataset")
    print(SEP)

    fpath = str(out_dir / "ex3_with_exposure.h5")
    n_frames = 5

    with h5py.File(fpath, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        data_grp = entry.create_group("data")
        data_grp.attrs["NX_class"] = "NXdata"
        data_grp.attrs["signal"] = "data"

        det_ds = data_grp.create_dataset(
            "data",
            shape=(0, H, W),
            maxshape=(None, H, W),
            dtype=np.float32,
            chunks=(10, H, W),
            compression="gzip",
            shuffle=True,
        )
        frame_ds = data_grp.create_dataset(
            "frame_number",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
        )
        timestamp_ds = data_grp.create_dataset(
            "timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
        )

        # ── 新增 exposure_time ──────────────────────────────
        exposure_ds = data_grp.create_dataset(
            "exposure_time",          # 新 dataset
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
        )
        exposure_ds.attrs["units"] = "s"   # NeXus 要求标注物理单位
        # ────────────────────────────────────────────────────

        for i in range(n_frames):
            frame = np.random.poisson(50 + i * 10, size=(H, W)).astype(np.float32)

            det_ds.resize(det_ds.shape[0] + 1, axis=0)
            det_ds[-1] = frame

            frame_ds.resize(frame_ds.shape[0] + 1, axis=0)
            frame_ds[-1] = i

            timestamp_ds.resize(timestamp_ds.shape[0] + 1, axis=0)
            timestamp_ds[-1] = time.time()

            # 新增：写入曝光时间（模拟变化的曝光，0.1s 递增）
            exposure_ds.resize(exposure_ds.shape[0] + 1, axis=0)
            exposure_ds[-1] = 0.1 * (i + 1)

            print(f"  Written frame {i}  |  exposure={0.1*(i+1):.1f}s")

    # --- 验证 ---
    print(f"\n  【验证文件结构】")
    with h5py.File(fpath, "r") as f:
        def print_tree(name, obj):
            depth = name.count("/")
            indent = "    " + "  " * depth
            kind = "GROUP  " if isinstance(obj, h5py.Group) else "DATASET"
            shape = str(obj.shape) if hasattr(obj, "shape") else ""
            print(f"{indent}{kind}  {name}  {shape}")
        f.visititems(print_tree)

        et = f["entry/data/exposure_time"][()]
        fn = f["entry/data/frame_number"][()]
        units = f["entry/data/exposure_time"].attrs["units"]
        print(f"\n    frame_number  : {fn}")
        print(f"    exposure_time : {et} ({units})")
        print(f"\n  【结论】")
        print(f"    exposure_time 与其他 dataset 完全对齐（同帧索引）。")
        print(f"    .attrs['units']='s' 满足 NeXus 规范，读者无需查文档知道单位。")
        print(f"    三步模式（create → resize → ds[-1]=value）适用于任何新 dataset。")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'#' * 60}")
    print("  h5py 对比实验")
    print(f"  输出目录: {out_dir.resolve()}")
    print(f"{'#' * 60}")

    exercise_1()
    exercise_2()
    exercise_3()

    print(f"\n{SEP}")
    print("全部完成。生成的 .h5 文件在 output_exercises/ 目录。")
    print(SEP)
