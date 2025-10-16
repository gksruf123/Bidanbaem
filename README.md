.# 🦾 Bidanbaem — ROS2 자율주행 로봇 프로젝트

> Raspberry Pi 5 + ROS2 Humble + YOLOv5 + Depth Camera 기반의 완전 자율주행 로봇 🚘  

---

## 📘 프로젝트 개요

**Bidanbaem**은 **ROS2 기반 자율주행 로봇**으로,  
YOLOv5로 신호·표지판·횡단보도 및 도로 화살표를 인식하고  
**LaneDetector로 차선을 감지**, **PID를 활용**해 조향합니다.

### 🎯 주요 기능
- ✅ YOLOv5 객체 인식 (신호등, 횡단보도, 표지판 등)
- ✅ LAB 기반 차선 인식 + PID 조향 제향
- ✅ Depth Camera를 이용한 거리 계산 (mm 단위)
- ✅ 상태머신 기반 로직 (wait → start → turn → stop)
- ✅ RGB LED 및 GPIO 피드백 (현재 주행 상태 시각화)

---

## 📁 디렉토리 구조

```
Bidanbaem/
├── src/
│   └── example/example/self_driving/
│       ├── self_driving.launch.py              # 런치 스크립트(SelfDriving)
│       ├── self_driving.py                     # 메인 노드 (SelfDrivingNode)
│       └── lane_detect.py                      # 차선 검출 모듈
│   └── smooth_cmd_vel/smooth_cmd_vel/
│       └── smooth_cmd_vel.py                   # 가감속 제어 (SmoothCmdVel)
│   └── yolov5_ros2/yolov5_ros2/
│       └── yolo_detect.py                      # Yolov5 검출 노드 (YoloV5Ros2)
│   └── peripherals/
│       └── launch/   
│           └── include/    
│               └── ascamera.launch.py          # ascamera 런치 스크립트(외부 ascamera 노드)    
│           └── depth_camera_launch.py          # 뎁스 카메라 런치 스크립트    
│   └── driver/controller/
│       └── launch/ 
│           └── controller.launch.py            # controller 런치 스크립트
└── README.md
```

---

## ⚙️ 개발 환경

| 항목 | 내용 |
|------|------|
| OS | Ubuntu 22.04 (64-bit) |
| ROS2 | Humble |
| Python | 3.12 이상 |
| 주요 패키지 | `opencv-python`, `numpy`, `rclpy`, `cv_bridge`, `gpiozero`, `ros_robot_controller_msgs` |
| 하드웨어 | MentorPi M1 Standard (Raspberry Pi 5 + Angstrong RGB/Depth 카메라) |

---

## ▶️ 실행 방법

### ✅ 실제 센서 환경
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch example self_driving.launch.py
```

### ✅ rosbag 테스트
```bash
cd ~/ros2_ws/rosbag/rosbags
ros2 bag play <bagdir> --loop
```

### 필수 Topic
`/ascamera/camera_publisher/rgb0/image`  
`/ascamera/camera_publisher/depth0/image_raw`  
`/yolov5_ros2/object_detect`  

---

## 🧠 주요 노드 & 토픽

| 구분 | 이름 | 타입 | 설명 |
|------|------|------|------|
| Node | `SelfDrivingNode` |  | 메인 로직 노드 |
| Node | `YoloV5Ros2` |  | Yolov5 검출 노드 |
| Node | `SmoothCmdVel` |  | 가감속 노드 |
| Publisher | `/cmd_vel_input` | `Twist` | 속도속조향 명령 |
| Publisher | `/ros_robot_controller/set_rgb` | `RGBStates` | 차체 LED 제어 |
| Subscriber | `/ascamera/camera_publisher/rgb0/image` | `Image` | RGB 카메라 |
| Subscriber | `/ascamera/camera_publisher/depth0/image_raw` | `Image` | Depth(mm) |
| Subscriber | `/yolov5_ros2/object_detect` | `ObjectsInfo` | YOLO 객체 정보 |
| Service | `/yolov5/start`, `/yolov5/stop` | `Trigger` | YOLO on/off |

---

## 🚦 주행 로직 개요

1. **Lane Detection**
   - LAB 색영역 필터링 → ROI 이진화 → `LaneDetector`로 `left_lane_x` 계산  
2. **PID Steering**
   - `PID(0.4, 0.0, 0.05)`로 조향 보정 (`angular.z` 조절)
3. **Object Detection**
   - YOLO + Depth(mm) → `crosswalk`, `red`, `right`, `green`, `park`, `go`, `floor_right` 인식
4. **State Machine**
   | 상태 | 설명 | LED | 동작 | 
   |------|------|------|------|
   | `wait` | 대기 + YOLO 시작 | 흰색 점멸 | 버튼 입력 대기
   | `start` | 직진 주행 | 녹색 | 직진 및 차선 유지
   | `turn` | 우회전 | 노란색 점멸 |
   | `stop` | 정지 / 주차 | 빨강 | Yolov5 객체 인식
5. **LED / GPIO 표시**
   - `gpiozero` + `RGBStates` 메시지 병행 사용  

---

## ⚙️ 튜닝 가이드

| 파라미터 | 설명 | 기본값 | 단위
|-----------|------|---------|--------
| `go_linear_x` | 직진 속도 | 1.0 | m/s
| `line_angular_z` | 조향 한계 | 0.20 | rad/s
| `turn_angular_z` | 회전 속도 | -1.0 | rad/s
| `red_threshold_mm` | 신호등 정지 거리 | 1100 | mm
| `pid.kp, ki, kd` | PID 게인 | 0.4 / 0.0 / 0.05 |
| `SetPoint` | 차선 중심 좌표 | 75 |

🧩 **튜닝 팁**
- 직진 떨림 → `kp` ↓  
- 코너 늦음 → `kd` ↑  
- 직진 불안정 → ROI 하단 가중치 ↑  

---

## 🐞 트러블슈팅

| 문제 | 원인 / 해결 |
|------|--------------|
| YOLOv5가 시작 안됨 | `/yolov5/start` 서비스 미기동 → launch 순서 점검 |
| Depth None | 초기 프레임 inpaint 전이라 정상 |
| 차량 미동작 | `~/set_running` false 상태 → true로 전환 |
| 회전이 끝나지 않음 | 각도 오차 (`80~82°`) 조정 |
| 객체 미검출 | 클래스명 / 라벨 불일치, confidence 재조정 |

---

## 🧭 Git 워크플로우

```bash
git status
git add src/example/example/self_driving/
git commit -m "feat(self-driving): integrate depth-based distance logic"
git push origin main
```

---

## 🖼️ 실행 예시 (Demo)

<p align="center">
  <img src="https://github.com/gksruf123/Bidanbaem/assets/demo.gif" width="80%" alt="Demo Preview"/>
</p>

---

## 📜 라이선스

**MIT License © 2025 [gksruf123](https://github.com/gksruf123)**  
> 자유롭게 수정 / 배포 가능 (출처 명시 권장)

---

## 👨‍💻 비단뱀 팀

**Kim Minsung**[@Minssc](https://github.com/Minssc)  
**Kim Daeyong**[@Dae-Yong-Kim](https://github.com/Dae-Yong-Kim)  
**Hwang Hyeyun**[@Hwanghyeyun](https://github.com/Hwanghyeyun)  
**Yum Hankyul**[@gksruf123](https://github.com/gksruf123)  

---

<p align="center">
  <em>ROS2 + YOLOv5 + Depth Camera로 구현한 MentorPi 자율주행 로봇 🚘</em><br/>
</p>
