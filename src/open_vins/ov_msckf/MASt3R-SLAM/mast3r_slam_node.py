#!/usr/bin/env python3
import argparse
import time
import cv2
import torch
import torch.multiprocessing as mp
import zmq
import signal
import numpy as np
import lietorch
import traceback
import pathlib
import datetime
from scipy.spatial.transform import Rotation as R

# MASt3R 관련
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.visualization import run_visualization
from mast3r_slam.multiprocess_utils import new_queue
from mast3r_slam.mast3r_utils import load_retriever
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
import mast3r_slam.evaluate as eval

# --- Backend Process ---
def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)
    device = keyframes.device
    # factor graph 및 retrieval DB 초기화
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    while True:
        if states.get_mode() == Mode.TERMINATED:
            break
        
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0] 
        
        if idx == -1:
            time.sleep(0.01)
            continue

        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame, add_after_query=True,
            k=config["retrieval"]["k"], min_thresh=config["retrieval"]["min_thresh"]
        )
        
        kf_idx = [idx - 1] if idx > 0 else []
        kf_idx += retrieval_inds 
        kf_idx = list(set(kf_idx))
        if idx in kf_idx: kf_idx.remove(idx)

        frame_idx = [idx] * len(kf_idx)
        
        if kf_idx:
            # print(f"\033[96m[Backend] Optimize Frame {idx} with {kf_idx}\033[0m")
            factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                states.global_optimizer_tasks.pop(0)


class Mast3rSlamNode:
    def __init__(self):
        print("[MAST3R Node] Initializing...")
        load_config("./config/calib.yaml")
        
        # Viz용 ZMQ (포트 5560)
        self.ctx = zmq.Context()
        self.pub_raw = self.ctx.socket(zmq.PUB)
        self.pub_raw.bind("tcp://*:5560")
        self.pub_opt = self.ctx.socket(zmq.PUB)
        self.pub_opt.bind("tcp://*:5561")

        # 파라미터 튜닝 
        config['local_opt']['window_size'] = 200 
        config["local_opt"]["min_match_frac"] = 0.05
        config["reloc"]["min_match_frac"] = 0.1
        config["local_opt"]["sigma_dist"] = 3.0
        config["tracking"]["sigma_dist"] = 0.5
        config["local_opt"]["max_iters"] = 40 #20
        
        self.device = 'cuda:0'
        self.model = load_mast3r(device=self.device)
        self.model.share_memory()
        self.manager = mp.Manager()
        
        # 해상도 설정 (512 or 224로 맞춰야함)
        self.w_orig, self.h_orig = 720, 540
        self.inference_size = 512
        self.scale = self.inference_size / max(self.w_orig, self.h_orig)
        self.w_new = int(self.w_orig * self.scale)
        self.h_new = int(self.h_orig * self.scale)
        
        self.keyframes = SharedKeyframes(self.manager, self.h_new, self.w_new)
        self.states = SharedStates(self.manager, self.h_new, self.w_new)
        
        # Intrinsics
        fx, fy, cx, cy = 351.314, 351.491, 367.852, 253.840
        self.K_orig = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.D_orig = np.array([-0.0369, -0.0089, 0.0089, -0.0037])
        
        self.K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K_orig, self.D_orig, (self.w_orig, self.h_orig), np.eye(3), balance=0.5
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K_orig, self.D_orig, np.eye(3), self.K_new, (self.w_orig, self.h_orig), cv2.CV_16SC2
        )

        fx_new = self.K_new[0, 0] * self.scale
        fy_new = self.K_new[1, 1] * self.scale
        cx_new = self.K_new[0, 2] * self.scale
        cy_new = self.K_new[1, 2] * self.scale
        
        self.K_cpu = torch.tensor([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]], dtype=torch.float32)
        self.K = self.K_cpu.to(self.device)
        self.keyframes.set_intrinsics(self.K)

        # self.T_cam_imu_np = np.array([
        #     [0.0067, 0.9999, -0.0101, -0.0458],
        #     [0.0024, 0.0100, 0.9999, 0.0126],
        #     [0.9999, -0.0067, -0.0023, -0.0509],
        #     [0.0, 0.0, 0.0, 1.0]
        # ])
        self.T_cam_imu_np = np.array([
            [ 0.00670802, 0.00242564, 0.99997456, 0.05126355], 
            [ 0.99992642, 0.01009120, -0.00673218, 0.04539012], 
            [-0.01010727, 0.99994614, -0.00235777, -0.01321491], 
            [ 0.0, 0.0, 0.0, 1.0]
        ])
        
        self.T_IC_np = np.linalg.inv(self.T_cam_imu_np) 


        t_ic = self.T_IC_np[:3, 3]
        r_ic = R.from_matrix(self.T_IC_np[:3, :3])
        q_ic = r_ic.as_quat() 
        vec_ic = np.concatenate([t_ic, q_ic])
        self.T_IC = lietorch.SE3(torch.from_numpy(vec_ic).float().unsqueeze(0)).to(self.device)

        t_ci = self.T_cam_imu_np[:3, 3]
        r_ci = R.from_matrix(self.T_cam_imu_np[:3, :3])
        q_ci = r_ci.as_quat()
        vec_ci = np.concatenate([t_ci, q_ci])
        
        self.T_cam_imu_se3 = lietorch.SE3(
            torch.from_numpy(vec_ci).float().unsqueeze(0).to(self.device)
        )
        self.backend = mp.Process(target=run_backend, args=(config, self.model, self.states, self.keyframes, self.K))
        self.backend.start()
        
        self.main2viz = new_queue(self.manager, False)
        self.viz2main = new_queue(self.manager, False)
        self.viz = mp.Process(target=run_visualization, args=(config, self.states, self.keyframes, self.main2viz, self.viz2main))
        self.viz.start()

        self.timestamps = [] 
        self.last_processed_pose = None
        print(f"[MAST3R Node] Ready.")
        self.last_opt_pub_time = 0

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
        
    def process_frame(self, timestamp, pose_array, img_bgr):
        tx, ty, tz, qx, qy, qz, qw = pose_array[:7]
        if np.abs(tx) < 1000:
            raw_data = np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float64)
            self.pub_raw.send(raw_data.tobytes())
        self.last_processed_pose = pose_array

        self.timestamps.append(timestamp)

        # Pose Conversion
        T_WI_cpu = torch.from_numpy(pose_array).float().unsqueeze(0) 
        T_WI = lietorch.SE3(T_WI_cpu).to(self.device)
        T_WC_SE3 = T_WI * self.T_IC
        
        scale = torch.ones((1, 1), device=self.device, dtype=torch.float32)
        T_WC_data = torch.cat([T_WC_SE3.data, scale], dim=1)
        T_WC = lietorch.Sim3(T_WC_data)

        # Image Processing (CLAHE -> Undistort -> Mask -> RGB)
        if len(img_bgr.shape) == 3:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_bgr

        # low-light 보정
        #img_gamma = self.adjust_gamma(img_gray, gamma=2.5)

        # 1. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_gray = clahe.apply(img_gray)

        # 2. Undistort
        img_undistorted_gray = cv2.remap(img_gray, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        # 3. Masking
        h, w = img_undistorted_gray.shape
        mask_margin = int(min(h, w) * 0.1)
        img_undistorted_gray[:mask_margin, :] = 0 
        img_undistorted_gray[-mask_margin:, :] = 0
        img_undistorted_gray[:, :mask_margin] = 0
        img_undistorted_gray[:, -mask_margin:] = 0

        #cv2.imshow("Preprocessed Input", img_undistorted_gray)
        #cv2.waitKey(1)

        # 4. RGB & Tensor
        img_rgb = cv2.cvtColor(img_undistorted_gray, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0

        # Inference
        kf_idx = len(self.keyframes)
        frame = create_frame(kf_idx, img_tensor, T_WC, img_size=self.inference_size, device=self.device)
        X, C = mast3r_inference_mono(self.model, frame)
        frame.update_pointmap(X, C)

        self.keyframes.append(frame)
        self.states.queue_global_optimization(kf_idx)
        self.states.set_frame(frame)
        
        print(f"[Keyframe {kf_idx}] Added. TS: {timestamp:.6f}")

        if len(self.keyframes) % 5 == 0:
            self.publish_optimized_path()

    def publish_optimized_path(self):
        try:
            n_kf = len(self.keyframes)
            if n_kf < 1: return
            poses_sim3 = self.keyframes.T_WC.data[:n_kf].view(-1, 8)
            
            poses_se3_data = poses_sim3[:, :7].to(self.device) # (N, 7)
            T_WC_all = lietorch.SE3(poses_se3_data)
            
            T_WI_all = T_WC_all * self.T_cam_imu_se3 # Broadcasting
            
            flat_data = T_WI_all.data.cpu().numpy().flatten().astype(np.float32)
            self.pub_opt.send(flat_data.tobytes())
            
        except Exception as e:
            print(f"[Viz Error] {e}")
            traceback.print_exc() # 자세한 에러 확인용

    def save_results(self):
        print("\n\033[92m[Saving] Converting to IMU Frame & Saving...\033[0m")
        date_str = 'hilti21_1210_test3'
        save_dir = pathlib.Path(f"output/{date_str}")
        save_dir.mkdir(parents=True, exist_ok=True)

        traj_path = save_dir / "trajectory.txt"
        
        n_valid = len(self.timestamps)

        poses_sim3 = self.keyframes.T_WC.data[:n_valid].view(-1, 8)
        poses_se3_data = poses_sim3[:, :7].to(self.device)
        
        T_WC_all = lietorch.SE3(poses_se3_data) 
        T_WI_all = T_WC_all * self.T_cam_imu_se3

        poses_np = T_WI_all.data.cpu().numpy()
        
        with open(traj_path, "w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            
            for i in range(n_valid):
                ts = self.timestamps[i]
                p = poses_np[i] # (7,)
                
                tx, ty, tz = p[0], p[1], p[2]
                qx, qy, qz, qw = p[3], p[4], p[5], p[6]
                
                if np.abs(tx) > 10000: continue

                f.write(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        
        print(f" -> Saved: {traj_path}")
    def shutdown(self):
        print("Shutting down processes...")
        self.states.set_mode(Mode.TERMINATED)
        self.backend.join()
        self.viz.join()

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.set_grad_enabled(False)
    
    running = True
    def stop(*_):
        global running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    
    TARGET_IP = "127.0.0.1" 
    sock.connect(f"tcp://{TARGET_IP}:5555")
    sock.setsockopt(zmq.SUBSCRIBE, b"kf")

    sock.setsockopt(zmq.RCVTIMEO, -1) 
    
    node = Mast3rSlamNode()

    print(f"Connected to {TARGET_IP}.")
    print("\033[93m[Wait] Waiting for OpenVINS initialization... (Move the robot!)\033[0m")

    first_packet = True 

    try:
        while running:
            try:
                parts = sock.recv_multipart() 
                
                if first_packet:
                    print("\n\033[96m[Start] First Keyframe Received! Auto-save enabled.\033[0m")
                    sock.setsockopt(zmq.RCVTIMEO, 500000) 
                    first_packet = False

                if len(parts) != 4: continue

                topic, stamp_bytes, pose_bytes, jpg_bytes = parts
                
                stamp = np.frombuffer(stamp_bytes, dtype=np.float64)[0]
                pose = np.frombuffer(pose_bytes, dtype=np.float32)
                img_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
                
                node.process_frame(stamp, pose, img)

            except zmq.Again:
                print("\n\033[93m[Info] No data received for 500s. Assuming Rosbag finished.\033[0m")
                break
            except zmq.ZMQError:
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        pass
    finally:
        node.save_results()
        node.shutdown()
        sock.close()
        ctx.term()
        print("Done.")