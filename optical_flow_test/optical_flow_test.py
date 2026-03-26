# 这个文件是手动打mask并且利用CPU光流追踪的测试脚本，完全独立于ROS2和YOLO，适合在任何环境下测试光流追踪的稳定性和性能。

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

# ==========================================
# 模块 1: CPU 极速光流追踪器 (核心算法)
# ==========================================
class CPUFastTracker:
    def __init__(self):
        # 提取 Shi-Tomasi 特征点参数
        self.feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        # LK 光流法参数
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.prev_gray = None
        self.prev_points = None
        self.current_mask = None

    def init_tracker(self, frame_bgr, yolo_mask_uint8):
        """初始化：用第一帧原图和 Mask 提取特征点"""
        self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.current_mask = yolo_mask_uint8.copy()
        self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=self.current_mask, **self.feature_params)
        return self.prev_points is not None

    def update_tracker(self, frame_bgr):
        """更新：在后续帧中极速追踪特征点并平移 Mask"""
        if self.prev_points is None or self.current_mask is None:
            return False, None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)

        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        if len(good_new) < 5:
            return False, None # 点丢了太多，认为追踪失败

        # 计算仿射变换矩阵（包含平移和旋转）
        matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)

        if matrix is not None:
            h, w = gray.shape
            # 平移并旋转 Mask
            self.current_mask = cv2.warpAffine(self.current_mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            self.prev_gray = gray.copy()
            self.prev_points = good_new.reshape(-1, 1, 2)
            return True, self.current_mask
        else:
            return False, None


# ==========================================
# 模块 2: 交互式 Mask 绘制工具 (模拟 YOLO)
# ==========================================
polygon_points = []

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数：左键加点，右键撤销"""
    global polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(polygon_points) > 0:
            polygon_points.pop()

def get_manual_mask(frame):
    """弹出新窗口，让用户描边画多边形"""
    global polygon_points
    polygon_points = []
    clone = frame.copy()
    cv2.namedWindow("Draw Mask (Mock YOLO)")
    cv2.setMouseCallback("Draw Mask (Mock YOLO)", mouse_callback)
    
    print("\n[系统] 已暂停时间轴。")
    print("👉 请在弹出的图像上用【鼠标左键】沿着目标边缘描点。")
    print("👉 点错了用【鼠标右键】撤销��")
    print("👉 描完一圈后，按下【空格键】确认并恢复播放。")

    while True:
        temp = clone.copy()
        if len(polygon_points) > 0:
            for i in range(len(polygon_points)):
                cv2.circle(temp, polygon_points[i], 3, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(temp, polygon_points[i-1], polygon_points[i], (0, 255, 0), 2)
            if len(polygon_points) > 2:
                cv2.line(temp, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)

        cv2.imshow("Draw Mask (Mock YOLO)", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 32 or key == 13: # 按空格或回车完成
            break

    cv2.destroyWindow("Draw Mask (Mock YOLO)")
    
    # 生成二值化掩码
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if len(polygon_points) > 2:
        pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask


# ==========================================
# 模块 3: 主流程 (解析 Bag + 循环测试)
# ==========================================
def main():
    # ⚠️ TODO: 修改为你实际的 bag 文件夹路径
    BAG_PATH = './first_time' 
    # ⚠️ TODO: 修改为你实际的 RGB 图像 Topic
    IMAGE_TOPIC = '/camera_dcw2/color/image_raw' 

    tracker = CPUFastTracker()
    is_tracking = False

    print(f"正在打开 ROS2 Bag: {BAG_PATH}")
    print("操作说明：播放时按 's' 键暂停并打 Mask，按 'q' 键退出。\n")

    try:
        with Reader(BAG_PATH) as reader:
            for connection, timestamp, rawdata in reader.messages():
                # 仅筛选我们需要的图像话题
                if connection.topic == IMAGE_TOPIC:
                    # 1. 解码 ROS2 Image 消息为 OpenCV 矩阵
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
                    
                    # 处理颜色通道转换 (根据你相机的实际编码格式)
                    if msg.encoding in ['rgb8', 'bgr8']:
                        if msg.encoding == 'rgb8':
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        continue # 略过非标准格式

                    # 2. 状态机：如果还没在追踪，就正常播放画面等待触发
                    if not is_tracking:
                        cv2.imshow("Robot Vision Test", frame)
                        key = cv2.waitKey(30) & 0xFF
                        
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            # 触发手动打 Mask (此时 for 循环卡住，完美冻结时间)
                            mask = get_manual_mask(frame)
                            if tracker.init_tracker(frame, mask):
                                print("[系统] Mask 注入成功！恢复时间轴并开始纯 CPU 追踪！")
                                is_tracking = True
                            else:
                                print("[警告] 框选区域过于平滑无纹理，无法提取追踪特征，请重新框选！")

                    # 3. 状态机：如果正在追踪，调用光流法高频更新 Mask
                    else:
                        success, current_mask = tracker.update_tracker(frame)
                        if success:
                            # 渲染可视化：半透明红色覆盖层
                            overlay = frame.copy()
                            overlay[current_mask > 0] = (0, 0, 255)
                            output = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                            
                            # 渲染特征点位置
                            for pt in tracker.prev_points:
                                x, y = pt.ravel()
                                cv2.circle(output, (int(x), int(y)), 3, (0, 255, 0), -1)

                            cv2.imshow("Robot Vision Test", output)
                        else:
                            print("[警告] 目标丢失（可能被遮挡或严重变形），停止追踪。按 's' 重新打 Mask。")
                            is_tracking = False

                        # 追踪过程中依然可以控制
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'): # 发现漂移了，手动打断重画
                            is_tracking = False
                            
    except Exception as e:
        print(f"读取 Bag 时出错: {e}")
        print("请检查 BAG_PATH 和 IMAGE_TOPIC 是否正确！")
        
    cv2.destroyAllWindows()
    print("测试结束。")

if __name__ == "__main__":
    main()
