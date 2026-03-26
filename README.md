# Optical Flow Test（ROS2 Bag 手动标注光流追踪）

该项目提供一个**独立脚本**，用于回放 ROS2 bag 图像并进行目标区域手动描边，再通过 CPU 光流进行持续追踪。

> 脚本文件：`optical_flow_test/optical_flow_test.py`

## 1. 功能概览

- 手动画多边形 Mask（模拟检测器输出）
- Shi-Tomasi 角点提取
- Lucas-Kanade（LK）光流跟踪特征点
- 通过仿射变换（`estimateAffinePartial2D`）更新 Mask
- 实时显示半透明红色追踪区域 + 绿色特征点

## 2. 环境要求

- Python 3.8+
- 依赖安装：

```bash
pip install opencv-python numpy rosbags
```

## 3. 使用前配置

编辑 `optical_flow_test/optical_flow_test.py` 中 `main()` 的这两个参数：

- `BAG_PATH`：ROS2 bag 目录（包含 `metadata.yaml`）
- `IMAGE_TOPIC`：图像 Topic 名称（例如 `/camera_dcw2/color/image_raw`）

## 4. 运行方法

```bash
python optical_flow_test/optical_flow_test.py
```

启动后终端会提示操作方式。

## 5. 交互操作

### 播放阶段

- `s`：暂停并进入手动画 Mask
- `q`：退出程序

### 画 Mask 窗口

- 鼠标左键：添加一个多边形顶点
- 鼠标右键：撤销上一个点
- `空格` / `Enter`：确认并开始追踪

### 追踪阶段

- `q`：退出程序
- `s`：中断当前追踪并重新标注 Mask

## 6. 数据与编码注意事项

当前脚本仅处理 `rgb8` 与 `bgr8` 编码图像；其他编码会被跳过。

## 7. 常见问题排查

- **无法读取 bag**：确认 `BAG_PATH` 是否正确、目录是否包含合法 ROS2 bag 数据。
- **画面不显示**：检查 `IMAGE_TOPIC` 是否和 bag 内图像话题一致。
- **无法开始追踪**：框选区域纹理过少（角点不足），请重新选择纹理更明显的区域。
- **追踪中断**：目标遮挡、变形过大或特征点丢失时会停止，按 `s` 重新标注。

## 8. 代码结构说明

- `CPUFastTracker`
  - `init_tracker(frame_bgr, yolo_mask_uint8)`：初始化追踪器并提取初始特征点
  - `update_tracker(frame_bgr)`：跟踪特征点并更新当前 Mask
- `get_manual_mask(frame)`：弹窗手动绘制多边形并生成二值 Mask
- `main()`：读取 ROS2 bag、状态机控制播放/标注/追踪流程
