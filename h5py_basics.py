"""
h5py 基础
=========================
目的：掌握 HDF5 文件的读写，理解层级结构。
这是 所有数据存储的基础格式。

运行方式：python 01_h5py_basics.py
"""

import h5py
import numpy as np
from pathlib import Path


# =============================================================================
# PART 1: 创建 HDF5 文件 — 理解层级结构
# 重点：HDF5 就像一个文件系统，Group = 目录，Dataset = 文件
# =============================================================================

def create_basic_hdf5(filepath: str) -> None:
    """创建一个基础 HDF5 文件，演示所有核心概念。"""

    with h5py.File(filepath, "w") as f:

        # --- Group（组）：类似文件夹 ---
        # 重点：用 create_group() 创建层级，路径用 "/" 分隔
        instrument = f.create_group("instrument")
        detector = instrument.create_group("detector")
        sample = f.create_group("sample")

        # --- Attributes（属性）：附加到 Group 或 Dataset 的元数据 ---
        # 重点：NeXus 标准要求每个 group 有 NX_class 属性
        # 这是 W 数据文件的核心约定
        f.attrs["NX_class"] = "NXroot"
        f.attrs["file_name"] = filepath
        instrument.attrs["NX_class"] = "NXinstrument"
        instrument.attrs["name"] = "W CoSAXS"
        detector.attrs["NX_class"] = "NXdetector"
        sample.attrs["NX_class"] = "NXsample"
        sample.attrs["name"] = "lysozyme_001"

        # --- Dataset（数据集）：实际存储数据的地方 ---
        # 重点：shape 定义维度，dtype 定义数据类型
        # float32 比 float64 节省一半空间，精度对科学数据够用

        # 模拟一张探测器图像（512×512 像素）
        fake_image = np.random.poisson(100, size=(512, 512)).astype(np.float32)
        detector.create_dataset("data", data=fake_image)
        detector.create_dataset("x_pixel_size", data=0.075)   # mm
        detector.create_dataset("y_pixel_size", data=0.075)   # mm
        detector.create_dataset("distance", data=1500.0)       # mm

        # 标量数据
        instrument.create_group("monochromator").create_dataset(
            "energy", data=12400.0  # eV
        )
        instrument["monochromator"].attrs["NX_class"] = "NXmonochromator"

        print(f"✓ Created: {filepath}")


# =============================================================================
# PART 2: 读取 HDF5 文件 — 理解懒加载
# 重点：h5py 默认不把数据读入内存，只有 [()] 或切片时才读取
#       这对 GB 级文件非常重要
# =============================================================================

def read_hdf5_demo(filepath: str) -> None:
    """演示 HDF5 文件的读取方式。"""

    with h5py.File(filepath, "r") as f:

        # --- 遍历结构 ---
        print("\n=== File Structure ===")
        def print_structure(name, obj):
            indent = "  " * name.count("/")
            kind = "GROUP" if isinstance(obj, h5py.Group) else "DATASET"
            shape = obj.shape if hasattr(obj, "shape") else ""
            print(f"{indent}{name}  [{kind}] {shape}")
        f.visititems(print_structure)

        # --- 读取 Dataset ---
        # 重点：f["path/to/dataset"] 返回的是 Dataset 对象，不是数组
        #       加 [()] 才把数据读入 numpy 数组
        dataset_obj = f["instrument/detector/data"]
        print(f"\nDataset object type: {type(dataset_obj)}")   # h5py.Dataset
        print(f"Shape: {dataset_obj.shape}, dtype: {dataset_obj.dtype}")

        # 方式1：全部读入内存
        full_array = dataset_obj[()]   # 等价于 dataset_obj[...]
        print(f"Full array loaded: {full_array.shape}, {full_array.nbytes/1e6:.1f} MB")

        # 方式2：只读一部分（内存高效！）
        # 重点：这是处理大文件的关键技巧，只读 ROI
        roi = dataset_obj[100:200, 100:200]  # 只读100×100的区域
        print(f"ROI loaded: {roi.shape}")  # 只用了 1/26 的内存

        # --- 读取 Attributes ---
        print(f"\nNX_class: {f['instrument'].attrs['NX_class']}")
        print(f"Beamline: {f['instrument'].attrs['name']}")
        print(f"Energy: {f['instrument/monochromator/energy'][()]} eV")


# =============================================================================
# PART 3: 可扩展数据集 — 流式写入的关键
# 重点：W 的 stream_receiver 就是这样往 HDF5 里追加帧的
#       maxshape=(None, H, W) 允许第一个维度无限扩展
# =============================================================================

def create_streaming_hdf5(filepath: str, n_frames: int = 10) -> None:
    """演示流式写入：逐帧追加到 HDF5，模拟探测器数据采集。"""

    H, W = 256, 256  # 使用小尺寸方便演示

    with h5py.File(filepath, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        data_grp = entry.create_group("data")
        data_grp.attrs["NX_class"] = "NXdata"
        data_grp.attrs["signal"] = "data"   # NeXus 约定：告诉读者主数据在哪

        # 重点：初始 shape=(0, H, W)，maxshape=(None, H, W)
        # None 表示该维度可以无限增长
        # chunks=(1, H, W) 表示每次读写的最小单位是 1 帧
        # 这样随机访问第 N 帧时，只需要读 1 个 chunk
        det_ds = data_grp.create_dataset(
            "data",
            shape=(0, H, W),
            maxshape=(None, H, W),
            dtype=np.float32,
            chunks=(10, H, W),
            compression="gzip",    # LZF：快速无损压缩，科学数据推荐  #lzf  gzip
            shuffle=True,          # shuffle 提高压缩率
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

        # 模拟逐帧写入（模拟 ZeroMQ 接收到一帧就写一帧）
        import time
        for i in range(n_frames):
            frame = np.random.poisson(50 + i*10, size=(H, W)).astype(np.float32)

            # 重点：resize 扩展一帧，然后写入最后一个位置
            det_ds.resize(det_ds.shape[0] + 1, axis=0)
            det_ds[-1] = frame   # [-1] 表示最后一帧

            frame_ds.resize(frame_ds.shape[0] + 1, axis=0)
            frame_ds[-1] = i

            timestamp_ds.resize(timestamp_ds.shape[0] + 1, axis=0)
            timestamp_ds[-1] = time.time()

            print(f"  Written frame {i}, total size: {det_ds.shape}")

    print(f"✓ Streaming HDF5 created: {filepath} ({n_frames} frames)")


# =============================================================================
# PART 4: 验证文件
# =============================================================================

def verify_file(filepath: str) -> None:
    """验证 HDF5 文件结构，打印关键信息。"""
    with h5py.File(filepath, "r") as f:
        data = f["entry/data/data"]
        frames = f["entry/data/frame_number"][()]
        print(f"\n=== Verification: {Path(filepath).name} ===")
        print(f"Frames: {data.shape[0]}, Shape per frame: {data.shape[1:]}")
        print(f"Frame numbers: {frames}")
        print(f"Compression: {data.compression}, chunks: {data.chunks}")
        print(f"File size: {Path(filepath).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    out_dir = Path("output_week1")
    out_dir.mkdir(exist_ok=True)

    print("=== h5py Basics Demo ===\n")

    # 演示1：基础读写
    basic_file = str(out_dir / "basic.h5")
    create_basic_hdf5(basic_file)
    read_hdf5_demo(basic_file)

    # 演示2：流式写入
    print("\n=== Streaming Write Demo ===")
    stream_file = str(out_dir / "streaming.h5")
    create_streaming_hdf5(stream_file, n_frames=5)
    verify_file(stream_file)

 
