"""
nexus_structure.py
"""

import h5py
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


# =============================================================================
# PART 1: 完整的 NeXus SAXS 文件
# 重点：理解每个 NXclass 的含义和用途
#       NXentry > NXinstrument > NXdetector
#                              > NXmonochromator
#                > NXsample
#                > NXdata (主数据 group，包含指向 detector 的链接)
# =============================================================================

def create_nexus_saxs_file(filepath: str, n_frames: int = 20) -> None:
    """
    创建符合 NeXus NXsas 应用定义的 SAXS 数据文件。

    NeXus 应用定义 NXsas 规定了 SAXS 数据必须包含哪些字段。
    参考：https://manual.nexusformat.org/classes/applications/NXsas.html
    """
    H, W = 512, 512

    with h5py.File(filepath, "w") as f:

        # NeXus 根属性
        f.attrs["NX_class"] = "NXroot"
        f.attrs["file_name"] = filepath
        f.attrs["HDF5_Version"] = h5py.version.hdf5_version
        f.attrs["NeXus_version"] = "4.3.0"

        # -----------------------------------------------------------------------
        # NXentry：一次实验 = 一个 entry
        # 重点：一个 HDF5 文件可以有多个 entry（多次扫描），
        #       但 entry/definition 必须声明遵循哪个应用定义
        # -----------------------------------------------------------------------
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("title", data="SAXS scan of lysozyme in buffer")
        entry.create_dataset("definition", data="NXsas")   # 应用定义声明
        entry.create_dataset("start_time",
            data=datetime.now(timezone.utc).isoformat())
        entry.create_dataset("end_time",
            data=datetime.now(timezone.utc).isoformat())

        # -----------------------------------------------------------------------
        # NXinstrument：描述实验仪器参数
        # -----------------------------------------------------------------------
        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"
        inst.create_dataset("name", data="MTEST CoSAXS")

        # NXmonochromator：单色器，定义 X 射线能量和波长
        # 重点：energy 和 wavelength 是做 q 轴校准的关键参数
        #       q = 4π·sin(θ) / λ
        mono = inst.create_group("monochromator")
        mono.attrs["NX_class"] = "NXmonochromator"
        mono.create_dataset("energy", data=12400.0)       # eV
        mono["energy"].attrs["units"] = "eV"
        mono.create_dataset("wavelength", data=1.0)       # Ångström
        mono["wavelength"].attrs["units"] = "angstrom"

        # NXdetector：探测器参数，PyFAI 积分需要这些参数
        det = inst.create_group("detector")
        det.attrs["NX_class"] = "NXdetector"
        det.create_dataset("description", data="Dectris Eiger2 4M")
        det.create_dataset("distance", data=1500.0)        # 样品到探测器距离
        det["distance"].attrs["units"] = "mm"
        det.create_dataset("x_pixel_size", data=0.075)     # 像素尺寸
        det["x_pixel_size"].attrs["units"] = "mm"
        det.create_dataset("y_pixel_size", data=0.075)
        det["y_pixel_size"].attrs["units"] = "mm"
        det.create_dataset("beam_center_x", data=256.0)    # 束流中心，像素坐标
        det.create_dataset("beam_center_y", data=256.0)
        det.create_dataset("count_time", data=0.1)         # 曝光时间
        det["count_time"].attrs["units"] = "s"

        # 探测器数据（可扩展）
        # 重点：这里存的是原始 2D 图像的时间序列
        det_data = det.create_dataset(
            "data",
            shape=(0, H, W),
            maxshape=(None, H, W),
            dtype=np.float32,
            chunks=(1, H, W),
            compression="lzf",
        )
        det_data.attrs["units"] = "counts"
        det_data.attrs["long_name"] = "Detector counts"

        # -----------------------------------------------------------------------
        # NXsample：样品信息
        # -----------------------------------------------------------------------
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("name", data="lysozyme_10mg_ml")
        sample.create_dataset("description", data="Lysozyme in 50mM HEPES pH 7.5")
        sample.create_dataset("concentration", data=10.0)
        sample["concentration"].attrs["units"] = "mg/ml"

        # -----------------------------------------------------------------------
        # NXdata：主数据 group — NeXus 最重要的约定
        # 重点：NXdata 是数据可视化工具（如 SILX）查找数据的入口
        #       @signal 告诉工具"主数据在哪"
        #       @axes  告诉工具"坐标轴是什么"
        #       实际数据通过 NeXus 硬链接指向 detector/data，避免数据重复
        # -----------------------------------------------------------------------
        data_grp = entry.create_group("data")
        data_grp.attrs["NX_class"] = "NXdata"
        data_grp.attrs["signal"] = "data"     # 主信号：data dataset
        data_grp.attrs["axes"] = ["frame_number", "y_pixel", "x_pixel"]

        # 帧号和时间戳（可扩展）
        fn_ds = data_grp.create_dataset(
            "frame_number", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        fn_ds.attrs["long_name"] = "Frame number"

        ts_ds = data_grp.create_dataset(
            "timestamp", shape=(0,), maxshape=(None,), dtype=np.float64
        )
        ts_ds.attrs["units"] = "s"
        ts_ds.attrs["long_name"] = "Unix timestamp"

        # 像素坐标轴（静态，整个扫描不变）
        data_grp.create_dataset("x_pixel", data=np.arange(W, dtype=np.float32))
        data_grp["x_pixel"].attrs["units"] = "pixel"
        data_grp.create_dataset("y_pixel", data=np.arange(H, dtype=np.float32))
        data_grp["y_pixel"].attrs["units"] = "pixel"

        # NeXus 硬链接：data/data 指向 instrument/detector/data
        # 重点：这样不复制数据，data_grp["data"] 和 det_data 是同一块数据
        f["entry/data/data"] = f["entry/instrument/detector/data"]

        # -----------------------------------------------------------------------
        # 写入帧数据（模拟采集）
        # -----------------------------------------------------------------------
        import time
        for i in range(n_frames):
            # 生成合成 SAXS 图案（真实数据是从 ZeroMQ 流接收的）
            frame = _generate_saxs_frame(H, W, frame_idx=i)

            # 追加到 detector/data
            det_data.resize(det_data.shape[0] + 1, axis=0)
            det_data[-1] = frame

            fn_ds.resize(fn_ds.shape[0] + 1, axis=0)
            fn_ds[-1] = i

            ts_ds.resize(ts_ds.shape[0] + 1, axis=0)
            ts_ds[-1] = time.time()

        # 写入结束时间
        entry["end_time"][()] = datetime.now(timezone.utc).isoformat()

    size_mb = Path(filepath).stat().st_size / 1e6
    print(f"✓ NeXus file created: {filepath} ({n_frames} frames, {size_mb:.2f} MB)")


def _generate_saxs_frame(H: int, W: int, frame_idx: int = 0) -> np.ndarray:
    """生成合成 SAXS 衍射图案（同心环 + 噪声）。"""
    cy, cx = H / 2, W / 2
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # 背景散射（中心强，向外衰减）
    frame = 1000.0 * np.exp(-r / 100)

    # 衍射环（高斯峰）
    for r0, sigma, A in [(60, 8, 3000), (110, 6, 1500), (160, 5, 800)]:
        frame += A * np.exp(-(r - r0)**2 / (2 * sigma**2))

    # 光束挡板（beam stop）遮住中心
    frame[r < 15] = 0

    # 泊松噪声（X 射线的量子统计本质）
    frame = np.random.poisson(frame * (1 + frame_idx * 0.02)).astype(np.float32)
    return frame


# =============================================================================
# PART 2: 读取 NeXus 文件 — 提取物理参数
# 重点：从 NeXus 文件中读取 PyFAI 积分所需的几何参数
# =============================================================================

def read_nexus_geometry(filepath: str) -> dict:
    """
    从 NeXus 文件读取探测器几何参数。
    这些参数是 PyFAI 方位角积分的必要输入。
    """
    params = {}
    with h5py.File(filepath, "r") as f:
        det = f["entry/instrument/detector"]
        mono = f["entry/instrument/monochromator"]

        params["distance_mm"] = det["distance"][()]
        params["pixel_size_mm"] = det["x_pixel_size"][()]
        params["beam_center_x"] = det["beam_center_x"][()]
        params["beam_center_y"] = det["beam_center_y"][()]
        params["energy_eV"] = mono["energy"][()]
        params["wavelength_A"] = mono["wavelength"][()]
        params["n_frames"] = f["entry/data/data"].shape[0]
        params["frame_shape"] = f["entry/data/data"].shape[1:]

    return params


def read_frame(filepath: str, frame_idx: int) -> np.ndarray:
    """读取单帧探测器图像。"""
    with h5py.File(filepath, "r") as f:
        # 重点：只读一帧，不加载整个数据集
        return f["entry/data/data"][frame_idx][:]


# =============================================================================
# PART 3: 在 NeXus 文件上做简单分析
# =============================================================================

def radial_profile_numpy(frame: np.ndarray, center: tuple) -> tuple:
    """
    用纯 NumPy 计算径向强度分布（简化版方位角积分）。
    目的：理解积分的物理含义，Week 2 会用 PyFAI 做真正的积分。

    原理：把每个像素按到中心的距离分组，对每组求平均。
    """
    cy, cx = center
    H, W = frame.shape
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    r_max = min(H, W) // 2
    radii = np.arange(r_max)
    intensity = np.array([
        frame[r == ri].mean() if (r == ri).any() else 0
        for ri in radii
    ])
    return radii, intensity


if __name__ == "__main__":
    out_dir = Path("output_week1")
    out_dir.mkdir(exist_ok=True)

    print("=== NeXus Structure Demo ===\n")

    # 创建文件
    nexus_file = str(out_dir / "saxs_scan_001.h5")
    create_nexus_saxs_file(nexus_file, n_frames=10)

    # 读取几何参数
    print("\n=== Geometry Parameters ===")
    params = read_nexus_geometry(nexus_file)
    for k, v in params.items():
        print(f"  {k}: {v}")

    # 读取一帧并做径向分布
    frame = read_frame(nexus_file, frame_idx=0)
    print(f"\n=== Frame 0 ===")
    print(f"  Shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"  Min: {frame.min():.0f}, Max: {frame.max():.0f}, "
          f"Mean: {frame.mean():.1f}")

    radii, intensity = radial_profile_numpy(frame, center=(256, 256))
    print(f"\n=== Radial Profile (first 10 bins) ===")
    for r, I in list(zip(radii[:10], intensity[:10])):
        bar = "█" * int(I / intensity.max() * 20)
        print(f"  r={r:3d} px: {bar} {I:.0f}")

    # 验证 NeXus 链接
    with h5py.File(nexus_file, "r") as f:
        print(f"\n=== NeXus Link Verification ===")
        d1 = f["entry/data/data"]
        d2 = f["entry/instrument/detector/data"]
        print(f"  data/data id: {d1.id.id}")
        print(f"  detector/data id: {d2.id.id}")
        print(f"  Same object: {d1.id == d2.id}")  # True = 硬链接成功
