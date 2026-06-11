# Grasp_ROS_Flow 架构笔记

> 项目地址：https://github.com/lovelyyoshino/Grasp_ROS_Flow
>
> 核心技术栈：ROS2 Humble + YOLOv8 + MobileSAM + Contact-GraspNet + RealMan 双臂

---

## 一、系统概览

```

┌──────────┐    Topic     ┌──────────────────────┐    Topic    ┌──────────┐    Topic    ┌──────────┐
│ 相机驱动   │ ──────────→ │ grasp_hybrid_server   │ ──────────→ │  状态机   │ ──────────→ │ rm_driver│
│ (第三方)   │ color/depth │ (自研)                 │ grasp_poses │ (自研)    │ move/grip  │ (第三方)  │
└──────────┘             │   │  内部 JSON → inference_worker        └──────────┘            └──────────┘
                         │   │  (YOLOv8 + MobileSAM + CGN)                                          │
                         └───┴──────────────────────────────────────────────────────────────────────┘
                                                                                                    ↓
                                                                                              机械臂 + 夹爪
```

**四个 ROS2 Node**：相机驱动、grasp_hybrid_server、状态机、rm_driver。其中自研的是中间两个，两端是第三方现成驱动。

---

## 二、ROS2 基础

ROS2 不是编程语言，是一个**机器人通信框架**。核心概念：

| 概念 | 说明 | 类比 |
|------|------|------|
| **Node** | 独立运行的进程，干一件事 | 一个人 |
| **Topic** | 消息频道，发布者广播，订阅者接收 | 公告栏/大喇叭 |
| **Publish** | 往 Topic 发消息 | 往公告栏贴通知 |
| **Subscribe** | 从 Topic 收消息 | 蹲在公告栏旁边看 |
| **Service** | 一对一请求-响应 | 打电话 |

**发布/订阅是解耦的**：发布者不管有没有人接，订阅者不管谁发的，只认 Topic 名称。

---

## 三、四个 Node 详解

### Node 1：相机驱动（第三方，不写）

```
RealSense 官方 ROS2 包（realsense2_camera）

持续广播：
  /camera/color/image_raw   ← RGB 图像
  /camera/depth/image_raw   ← 深度图像
  /camera/depth/camera_info ← 相机内参（3×3矩阵）
```

### Node 2：grasp_hybrid_server（自研）

```
系统 Python 环境（ROS2 Humble 要求）

职责：
  1. 订阅三个相机话题，缓存最新一帧 RGB + Depth + CameraInfo
  2. 提供服务 /get_grasps（状态机触发时调用）
  3. 收到服务请求后，通过 subprocess + JSON 启动/调用推理子进程
  4. 推理完成后，把抓取位姿发布到 Topic /grasp_poses

关键设计：混合架构（见第五节）
```

### Node 3：状态机（自研）

```
系统 Python 环境

职责：
  1. while 循环 + 状态切换（OBSERVE → GRASP → TRANSPORT → ZONE_90 → DONE）
  2. 需要抓取时调用 Service /get_grasps
  3. 订阅 Topic /grasp_poses，获取抓取位姿
  4. 向 rm_driver 发送控制 Topic：
     - movej_cmd（关节运动）
     - movel_cmd（直线运动）
     - gripper_cmd（夹爪控制）
     - rotate_cmd（底盘旋转）

三个工位（0°/90°/180°）围绕旋转底盘：
  0° 工位：拣选上料（从料箱抓零件）
  90° 工位：装配摆盘（放下、检查、姿态调整）
  180° 工位：包装填入（放入飞机盒）
```

### Node 4：rm_driver（第三方，不写）

```
RealMan 官方 ROS2 包

职责：
  接收状态机的 move/gripper 命令
  → 逆运动学（IK）求解：末端位姿 → 6个关节角度
  → 轨迹规划：预抓取位 → 抓取位
  → 电机控制执行
  → 回报执行结果（move_result, joint_states）
```

---

## 四、完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ 一次抓取的完整数据流                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① 相机持续广播 RGB+Depth（Topic，30Hz）                          │
│       ↓                                                        │
│  ② grasp_hybrid_server 订阅并缓存最新帧                           │
│       ↓                                                        │
│  ③ 状态机调用 Service /get_grasps（"给我算个抓取"）                 │
│       ↓                                                        │
│  ④ grasp_hybrid_server 把 RGB+Depth 写成 JSON                   │
│     经 subprocess stdin 传给 inference_worker                    │
│       ↓                                                        │
│  ⑤ inference_worker（conda 环境）内部：                           │
│     YOLOv8 检测 → MobileSAM 分割 → 提取物体点云 → Contact-GraspNet │
│       ↓                                                        │
│  ⑥ 抓取位姿 JSON 经 stdout 返回 grasp_hybrid_server               │
│       ↓                                                        │
│  ⑦ grasp_hybrid_server 发布到 Topic /grasp_poses                │
│       ↓                                                        │
│  ⑧ 状态机订阅到抓取位姿，选最高分的                                │
│       ↓                                                        │
│  ⑨ 坐标变换（相机坐标系 → 机械臂基座坐标系）                         │
│       ↓                                                        │
│  ⑩ 状态机 → rm_driver（Topic：movel_cmd / gripper_cmd）         │
│       ↓                                                        │
│  ⑪ rm_driver IK + 轨迹规划 → 电机执行                             │
│       ↓                                                        │
│  ⑫ 夹爪闭合 → 抬起 → 旋转 → 下一工位                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、混合架构

### 为什么是混合架构

```
矛盾：ROS2 Humble 绑死系统 Python 3.10，而 PyTorch/YOLO 依赖 conda Python 3.9
      两者不能在同一进程共存（库冲突）。

解法：拆成两个进程，系统 Python 负责 ROS2 通信，conda Python 负责算法推理。
      两个进程之间用 subprocess + stdin/stdout + JSON 通信。
```

### 架构示意

```
系统 Python 进程                     conda Python 子进程
┌─────────────────────┐              ┌────────────────────┐
│ grasp_hybrid_server │  subprocess  │ inference_worker    │
│                     │ ──JSON(请求)─→│                     │
│ - 订阅相机话题       │              │ - YOLOv8 检测       │
│ - 提供/grasps 服务   │ ←─JSON(结果)─│ - MobileSAM 分割    │
│ - 发布 /grasp_poses  │              │ - Contact-GN 抓取   │
└─────────────────────┘              └────────────────────┘
```

### 推理流水线

```
RGB 图像 ──→ YOLOv8 ──→ 检测框 ──→ MobileSAM ──→ 精确掩码
                                                    │
                                          Depth 图像 + 掩码
                                                    │
                                                    ↓
                                              物体局部点云
                                                    │
                                                    ↓
                                          Contact-GraspNet
                                                    │
                                                    ↓
                                             6-DoF 抓取位姿
```

---

## 六、状态机与抓取执行

### 状态循环

```python
while True:
    if state == "OBSERVE":
        移动到观察位()
        state = "GRASP"

    elif state == "GRASP":
        调用 /get_grasps()
        选最高分抓取()
        移动到预抓取位()    # 目标正上方 z+8cm
        直线靠近()          # 精确到位
        夹爪力控闭合()
        抬起()
        state = "TRANSPORT"

    elif state == "TRANSPORT":
        移动到安全位()
        底盘旋转90°()
        state = "ZONE_90"

    elif state == "ZONE_90":
        # 装配/摆盘逻辑
        ...

    elif state == "DONE":
        底盘回0°()
        state = "OBSERVE"
```

### 从抓取位姿到机械臂运动

```
抓取位姿（相机坐标系）
      │
      ▼
手眼标定矩阵 → 机械臂基座坐标系
      │
      ▼
逆运动学（IK）→ 6个关节角度
      │
      ▼
运动规划 → 路径点（预抓取位 → 抓取位）
      │
      ▼
rm_driver → 电机执行
```

**我们写的**：选择抓取位姿 + 发 Topic 给 rm_driver
**rm_driver 搞定的**：IK、轨迹规划、电机控制

---

## 七、关键概念速查

| 概念 | 说明 |
|------|------|
| **ROS2** | 机器人通信框架，不是语言，是一套库 + 工具 |
| **Node** | 独立运行的进程，是 ROS2 的最小执行单位 |
| **Topic** | 广播式消息频道（发布→订阅），一对多 |
| **Service** | 请求-响应式调用，一对一 |
| **grasp_hybrid_server** | 自研 Node，ROS2 通信层 + 推理进程管理 |
| **inference_worker** | conda 环境的子进程，串行跑 YOLO+SAM+CGN |
| **状态机** | while 循环 + 状态切换，调度抓取→旋转→放置 |
| **rm_driver** | RealMan 官方 Node，内部做 IK + 运动规划 |
| **Contact-GraspNet** | 先预测接触点 → 反推抓取姿态（和 AnyGrasp 不同路线） |
| **子进程 JSON 通信** | 解决 ROS2 Python 和 conda Python 版本冲突的方案 |
| **预抓取位** | 抓取点正上方偏移位置，防止直接冲撞 |
| **逆运动学 (IK)** | 末端位姿 → 各关节角度 |
| **手眼标定** | 求相机坐标系到机械臂基座坐标系的变换矩阵 |

---

## 八、Contact-GraspNet vs 已读论文对比

| | AnyGrasp (2023) | 6-DOF GraspNet (2019) | Contact-GraspNet (2021) |
|---|---|---|---|
| 输入 | 整场景点云 | 单个物体点云 | 单个物体点云（需分割） |
| 核心思路 | 密集预测 | VAE 采样 + 梯度精炼 | 接触点预测 + 反推姿态 |
| 骨干网络 | PointNet++ | PointNet++ | 点云卷积 |
| 需要分割 | ❌ | ✅ | ✅ |
| 动态跟踪 | ✅ | ❌ | ❌ |
