#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 node:
1) 订阅 RGB 图像；
2) 订阅 YOLO 输出的 mono8 mask（mask 的 header.stamp 必须对应原始 RGB 帧的时间戳）；
3) 在 YOLO mask 到达时，从该旧时间戳对应的 RGB 帧重新初始化追踪器；
4) 使用 CPU LK 光流 + RANSAC 仿射估计，将 mask 从旧帧一路追到最新帧；
5) 当 YOLO 暂时算不出来时，继续在新 RGB 帧上做纯 CPU mask 传播；
6) 发布追赶后的 tracked mask 和可视化结果。

适合场景：
- YOLO 分割延迟较大；
- 目标刚体/准刚体运动为主（平移、旋转、轻微尺度变化）；
- 希望在两次 YOLO 输出之间用 CPU 追踪兜底。

注意：
- 这是“在线版”的第二段代码替代实现，不再依赖手工画 mask。
- 如果目标发生大形变、完全遮挡或出视野，任何光流方案都会失效；新到来的 YOLO mask 会重新校正漂移。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


@dataclass
class BufferedFrame:
    stamp_sec: float
    frame_bgr: np.ndarray
    header: object


class CatchupMaskTracker:
    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 5,
        block_size: int = 7,
        lk_win_size: int = 21,
        lk_max_level: int = 3,
        min_good_points: int = 8,
        re_detect_interval: int = 3,
        morph_kernel: int = 5,
    ) -> None:
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        self.lk_params = dict(
            winSize=(lk_win_size, lk_win_size),
            maxLevel=lk_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )

        self.min_good_points = int(min_good_points)
        self.re_detect_interval = int(re_detect_interval)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.last_stamp_sec: Optional[float] = None
        self.active = False
        self.track_step_count = 0

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_points = None
        self.current_mask = None
        self.last_stamp_sec = None
        self.active = False
        self.track_step_count = 0

    @staticmethod
    def _binarize_mask(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        out = np.where(mask > 0, 255, 0).astype(np.uint8)
        return out

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = self._binarize_mask(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _detect_points(self, gray: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        if mask is None or int(np.count_nonzero(mask)) == 0:
            return None
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        return pts

    def init_from_mask(self, frame_bgr: np.ndarray, mask_uint8: np.ndarray, stamp_sec: float) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = self._clean_mask(mask_uint8)
        pts = self._detect_points(gray, mask)

        if pts is None or len(pts) < self.min_good_points:
            self.reset()
            return False

        self.prev_gray = gray
        self.prev_points = pts
        self.current_mask = mask
        self.last_stamp_sec = float(stamp_sec)
        self.active = True
        self.track_step_count = 0
        return True

    def _try_redetect_and_flow(self, gray: np.ndarray):
        """当旧特征点太少时，在上一帧 mask 内重新检测点，再做一次 LK。"""
        if self.prev_gray is None or self.current_mask is None:
            return None, None

        redetected = self._detect_points(self.prev_gray, self.current_mask)
        if redetected is None or len(redetected) < self.min_good_points:
            return None, None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            redetected,
            None,
            **self.lk_params,
        )
        if next_points is None or status is None:
            return None, None

        good_new = next_points[status.reshape(-1) == 1]
        good_old = redetected[status.reshape(-1) == 1]
        if len(good_new) < self.min_good_points:
            return None, None

        return good_old, good_new

    def update_to_frame(self, frame_bgr: np.ndarray, stamp_sec: float) -> tuple[bool, Optional[np.ndarray]]:
        if not self.active or self.prev_gray is None or self.current_mask is None:
            return False, None

        if self.last_stamp_sec is not None and stamp_sec <= self.last_stamp_sec + 1e-9:
            return True, self.current_mask

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_points is None or len(self.prev_points) < self.min_good_points:
            good_old, good_new = self._try_redetect_and_flow(gray)
            if good_old is None:
                self.reset()
                return False, None
        else:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_points,
                None,
                **self.lk_params,
            )

            if next_points is None or status is None:
                self.reset()
                return False, None

            status = status.reshape(-1)
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]

            if len(good_new) < self.min_good_points:
                good_old, good_new = self._try_redetect_and_flow(gray)
                if good_old is None:
                    self.reset()
                    return False, None

        matrix, inliers = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )

        if matrix is None:
            self.reset()
            return False, None

        h, w = gray.shape
        warped_mask = cv2.warpAffine(
            self.current_mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = self._clean_mask(warped_mask)

        if int(np.count_nonzero(warped_mask)) == 0:
            self.reset()
            return False, None

        self.track_step_count += 1
        if self.track_step_count % self.re_detect_interval == 0:
            fresh_pts = self._detect_points(gray, warped_mask)
            if fresh_pts is not None and len(fresh_pts) >= self.min_good_points:
                self.prev_points = fresh_pts
            else:
                self.prev_points = good_new.reshape(-1, 1, 2)
        else:
            self.prev_points = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray
        self.current_mask = warped_mask
        self.last_stamp_sec = float(stamp_sec)
        return True, warped_mask


class DelayedYoloMaskCatchupNode(Node):
    def __init__(self) -> None:
        super().__init__("delayed_yolo_mask_catchup_node")
        self.bridge = CvBridge()

        # 话题参数
        self.declare_parameter("image_topic", "/camera_dcw2/sensor_color")
        self.declare_parameter("yolo_mask_topic", "/tracking/bottle_mask")
        self.declare_parameter("tracked_mask_topic", "/tracking/bottle_mask_tracked")
        self.declare_parameter("visualization_topic", "/tracking/bottle_mask_tracked_vis")

        # 时序 / 缓冲参数
        self.declare_parameter("buffer_max_frames", 180)        # 30fps 下大约 6 秒
        self.declare_parameter("lookup_tolerance_sec", 0.05)   # 找对应历史帧时允许的时间误差
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("log_every_n_frames", 30)

        # 追踪参数
        self.declare_parameter("feature_max_corners", 200)
        self.declare_parameter("feature_quality_level", 0.01)
        self.declare_parameter("feature_min_distance", 5)
        self.declare_parameter("feature_block_size", 7)
        self.declare_parameter("lk_win_size", 21)
        self.declare_parameter("lk_max_level", 3)
        self.declare_parameter("min_good_points", 8)
        self.declare_parameter("re_detect_interval", 3)
        self.declare_parameter("morph_kernel", 5)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.yolo_mask_topic = str(self.get_parameter("yolo_mask_topic").value)
        self.tracked_mask_topic = str(self.get_parameter("tracked_mask_topic").value)
        self.visualization_topic = str(self.get_parameter("visualization_topic").value)

        self.buffer_max_frames = int(self.get_parameter("buffer_max_frames").value)
        self.lookup_tolerance_sec = float(self.get_parameter("lookup_tolerance_sec").value)
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.log_every_n_frames = int(self.get_parameter("log_every_n_frames").value)

        self.tracker = CatchupMaskTracker(
            max_corners=int(self.get_parameter("feature_max_corners").value),
            quality_level=float(self.get_parameter("feature_quality_level").value),
            min_distance=int(self.get_parameter("feature_min_distance").value),
            block_size=int(self.get_parameter("feature_block_size").value),
            lk_win_size=int(self.get_parameter("lk_win_size").value),
            lk_max_level=int(self.get_parameter("lk_max_level").value),
            min_good_points=int(self.get_parameter("min_good_points").value),
            re_detect_interval=int(self.get_parameter("re_detect_interval").value),
            morph_kernel=int(self.get_parameter("morph_kernel").value),
        )

        qos = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.pub_mask = self.create_publisher(Image, self.tracked_mask_topic, qos)
        self.pub_vis = self.create_publisher(Image, self.visualization_topic, qos)

        self.sub_image = self.create_subscription(Image, self.image_topic, self._image_callback, qos)
        self.sub_mask = self.create_subscription(Image, self.yolo_mask_topic, self._mask_callback, qos)

        self.buffer_lock = threading.Lock()
        self.frame_buffer: deque[BufferedFrame] = deque(maxlen=self.buffer_max_frames)
        self.processed_frames = 0
        self.last_received_yolo_stamp: Optional[float] = None

        self.get_logger().info(
            "DelayedYoloMaskCatchupNode started. "
            f"image_topic={self.image_topic}, "
            f"yolo_mask_topic={self.yolo_mask_topic}, "
            f"tracked_mask_topic={self.tracked_mask_topic}"
        )

    @staticmethod
    def _to_sec(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _imgmsg_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.encoding == "bgr8":
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if msg.encoding == "rgb8":
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # 兜底
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert image to BGR: {exc}")
            return None

    def _maskmsg_to_mono8(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.encoding == "mono8":
                mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            else:
                mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if mask.ndim == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            return mask
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert mask to mono8: {exc}")
            return None

    def _find_buffer_index_by_stamp(self, stamp_sec: float) -> Optional[int]:
        if not self.frame_buffer:
            return None

        best_idx = None
        best_dt = float("inf")
        for i, item in enumerate(self.frame_buffer):
            dt = abs(item.stamp_sec - stamp_sec)
            if dt < best_dt:
                best_dt = dt
                best_idx = i

        if best_idx is None or best_dt > self.lookup_tolerance_sec:
            return None
        return best_idx

    def _build_visualization(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        vis = frame_bgr.copy()
        if mask is not None and int(np.count_nonzero(mask)) > 0:
            overlay = np.zeros_like(frame_bgr)
            overlay[:, :, 2] = mask
            vis = cv2.addWeighted(frame_bgr, 1.0, overlay, 0.45, 0.0)

            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self.tracker.prev_points is not None:
            for pt in self.tracker.prev_points.reshape(-1, 2):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
        return vis

    def _publish(self, mask: np.ndarray, frame_bgr: np.ndarray, header, source: str) -> None:
        mask_msg = self.bridge.cv2_to_imgmsg(mask.astype(np.uint8), encoding="mono8")
        mask_msg.header = header
        self.pub_mask.publish(mask_msg)

        if self.publish_overlay:
            vis = self._build_visualization(frame_bgr, mask)
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            vis_msg.header = header
            self.pub_vis.publish(vis_msg)

        self.processed_frames += 1
        if self.log_every_n_frames > 0 and self.processed_frames % self.log_every_n_frames == 0:
            nz = int(np.count_nonzero(mask))
            self.get_logger().info(
                f"published tracked mask | source={source} | nonzero={nz} | stamp={header.stamp.sec}.{header.stamp.nanosec:09d}"
            )

    def _image_callback(self, msg: Image) -> None:
        frame_bgr = self._imgmsg_to_bgr(msg)
        if frame_bgr is None:
            return

        stamp_sec = self._to_sec(msg)

        with self.buffer_lock:
            self.frame_buffer.append(
                BufferedFrame(
                    stamp_sec=stamp_sec,
                    frame_bgr=frame_bgr.copy(),
                    header=msg.header,
                )
            )

            # 没有 tracker 时，只缓存，不输出
            if not self.tracker.active:
                return

            # 已经追到这个时间戳或更后面时，不重复处理
            if self.tracker.last_stamp_sec is not None and stamp_sec <= self.tracker.last_stamp_sec + 1e-9:
                return

            ok, tracked_mask = self.tracker.update_to_frame(frame_bgr, stamp_sec)
            if not ok or tracked_mask is None:
                self.get_logger().warning("Tracker lost on incoming RGB frame; waiting for next YOLO mask.")
                return

            self._publish(tracked_mask, frame_bgr, msg.header, source="online_track")

    def _mask_callback(self, msg: Image) -> None:
        yolo_mask = self._maskmsg_to_mono8(msg)
        if yolo_mask is None:
            return

        if int(np.count_nonzero(yolo_mask)) == 0:
            # YOLO 这一帧没检出，保持当前 tracker 继续跑，不重置
            self.get_logger().info("Received empty YOLO mask; keep current tracker running.")
            return

        yolo_stamp_sec = self._to_sec(msg)

        with self.buffer_lock:
            idx = self._find_buffer_index_by_stamp(yolo_stamp_sec)
            if idx is None:
                self.get_logger().warning(
                    "No historical RGB frame matched this YOLO mask stamp. "
                    "Increase buffer_max_frames or lookup_tolerance_sec."
                )
                return

            base_item = self.frame_buffer[idx]
            ok = self.tracker.init_from_mask(base_item.frame_bgr, yolo_mask, base_item.stamp_sec)
            if not ok:
                self.get_logger().warning("YOLO mask received, but tracker init failed (not enough features).")
                return

            latest_mask = self.tracker.current_mask
            latest_frame = base_item.frame_bgr
            latest_header = base_item.header

            replay_count = 0
            for j in range(idx + 1, len(self.frame_buffer)):
                item = self.frame_buffer[j]
                ok, latest_mask = self.tracker.update_to_frame(item.frame_bgr, item.stamp_sec)
                if not ok or latest_mask is None:
                    self.get_logger().warning(
                        "Tracker failed during catch-up replay; waiting for next YOLO correction."
                    )
                    return
                latest_frame = item.frame_bgr
                latest_header = item.header
                replay_count += 1

            self.last_received_yolo_stamp = yolo_stamp_sec
            self.get_logger().info(
                "YOLO mask catch-up done. "
                f"yolo_stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, "
                f"replayed_frames={replay_count}, "
                f"latest_stamp={latest_header.stamp.sec}.{latest_header.stamp.nanosec:09d}"
            )
            self._publish(latest_mask, latest_frame, latest_header, source="yolo_catchup")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DelayedYoloMaskCatchupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
