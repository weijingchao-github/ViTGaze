import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import copy
import random

import cv2
import data.data_utils as data_utils
import deepface.modules.detection as deepface_detection
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from detectron2.config import LazyConfig, instantiate
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image
from torchvision import transforms

# record video
enable_record_video = True
if enable_record_video:
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    fps = 60
    output_video_path = os.path.join(
        os.path.dirname(__file__), "output_video_color.avi"
    )
    img_width = 640
    img_height = 480
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps, (img_width, img_height)
    )


class ViTGazeFollow:
    def __init__(self):
        # model init
        config_file_path = os.path.join(
            os.path.dirname(__file__), "configs/gazefollow_518.py"
        )
        model_weights_path = os.path.join(
            os.path.dirname(__file__), "weights/gazefollow.pth"
        )
        cfg = LazyConfig.load(config_file_path)
        self.gaze_follow_model = instantiate(cfg.model)
        self.gaze_follow_model.load_state_dict(torch.load(model_weights_path)["model"])
        self.gaze_follow_model.to(cfg.train.device)
        self.gaze_follow_model.train(False)
        # gaze follow train parameters
        self.input_size = 518  # 模型训练时用的这么大的
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        # face detector init
        self.face_detector_model = deepface_detection.deepface_face_detector_model_init(
            detector_backend="yolov8"
        )
        # others
        self.gaze_inout_threshold = 0.8
        self.enable_visualization = True
        # ROS init
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/camera/color/image_raw", Image, self._do_gaze_follow, queue_size=1
        )

    def _do_gaze_follow(self, image_color):
        image_color = self.bridge.imgmsg_to_cv2(image_color, desired_encoding="bgr8")
        image_color_plot = copy.deepcopy(image_color)
        image_height, image_width, _ = image_color.shape
        faces_detect_result_xywh = self._detect_face(image_color)
        # 先做一个人的gaze follow，后面再做图像场景中有多个人的gaze follow
        if len(faces_detect_result_xywh) != 0:
            for face_detect_result_xywh in faces_detect_result_xywh:
                plot_color = [random.randint(0, 255) for _ in range(3)]
                # xyxy compute
                x, y, w, h = face_detect_result_xywh
                x_min = max(0, int(x - w / 2))
                y_min = max(0, int(y - h / 2))
                x_max = min(image_width - 1, int(x + w / 2))
                y_max = min(image_height - 1, int(y + h / 2))
                xy1 = (int(x_min), int(y_min))
                xy2 = (int(x_max), int(y_max))
                # expand face bbox a bit
                k = 0.1
                x_min = max(x_min - k * abs(x_max - x_min), 0)
                y_min = max(y_min - k * abs(y_max - y_min), 0)
                x_max = min(x_max + k * abs(x_max - x_min), image_width - 1)
                y_max = min(y_max + k * abs(y_max - y_min), image_height - 1)
                # compute head_channel
                head_channel = data_utils.get_head_box_channel(
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    image_width,
                    image_height,
                    resolution=self.input_size,
                    coordconv=False,
                ).unsqueeze(0)
                # transform image
                image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
                image_pil = PIL_Image.fromarray(image_rgb)
                image_color_transformed = self._transform_image(image_pil)
                model_input_info = {
                    "images": image_color_transformed.unsqueeze(0),
                    "head_channels": head_channel.unsqueeze(0),
                }
                # gaze follow predicts
                with torch.no_grad():
                    gaze_heatmap_pred, gaze_inout_pred = self.gaze_follow_model(
                        model_input_info
                    )
                    gaze_heatmap_pred = (
                        gaze_heatmap_pred.squeeze(0, 1).cpu().detach().numpy()
                    )
                    gaze_inout_pred = gaze_inout_pred.squeeze(0).cpu().detach().numpy()
                if gaze_inout_pred >= self.gaze_inout_threshold:
                    gaze_point_x_ratio, gaze_point_y_ratio = self._dark_inference(
                        gaze_heatmap_pred
                    )
                    gaze_heatmap_height, gaze_heatmap_width = gaze_heatmap_pred.shape
                    gaze_point_x = int(
                        gaze_point_x_ratio / gaze_heatmap_width * image_width
                    )
                    gaze_point_y = int(
                        gaze_point_y_ratio / gaze_heatmap_height * image_height
                    )
                    if self.enable_visualization:
                        cv2.circle(
                            image_color_plot,
                            (gaze_point_x, gaze_point_y),
                            radius=5,
                            color=plot_color,
                            thickness=-1,
                        )
                        cv2.rectangle(
                            image_color_plot,
                            xy1,
                            xy2,
                            plot_color,
                            thickness=3,
                            lineType=cv2.LINE_AA,
                        )
                        # # get gaze heatmap
                        # gaze_heatmap = cv2.applyColorMap(
                        #     (gaze_heatmap_pred * 255).astype(np.uint8), cv2.COLORMAP_JET
                        # )
                        # gaze_heatmap = cv2.resize(
                        #     gaze_heatmap,
                        #     (image_width, image_height),
                        #     interpolation=cv2.INTER_LINEAR,
                        # )
                        # image_show = cv2.addWeighted(
                        #     image_color_plot, 1, gaze_heatmap, 0.5, 1
                        # )
                else:
                    if self.enable_visualization:
                        cv2.rectangle(
                            image_color_plot,
                            xy1,
                            xy2,
                            plot_color,
                            thickness=3,
                            lineType=cv2.LINE_AA,
                        )
        else:
            print("No face detected in the image.")
        if enable_record_video:
            video_writer.write(image_color_plot)
        if self.enable_visualization:
            cv2.imshow("gaze_follow", image_color_plot)
            cv2.waitKey(1)

    def _detect_face(self, image_detect):
        faces_detect_result = self.face_detector_model.detect_faces(image_detect)
        faces_detect_result_xywh = []
        for face_detect_result in faces_detect_result:
            faces_detect_result_xywh.append(face_detect_result["xywh"])
        return faces_detect_result_xywh

    def _transform_image(self, image):
        transform_pipline = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return transform_pipline(image)

    def _dark_inference(self, heatmap: np.ndarray, gaussian_kernel: int = 39):
        pred_x, pred_y = self._argmax_pts(heatmap)
        pred_x, pred_y = int(pred_x), int(pred_y)
        height, width = heatmap.shape[-2:]
        # Gaussian blur
        orig_max = heatmap.max()
        border = (gaussian_kernel - 1) // 2
        dr = np.zeros((height + 2 * border, width + 2 * border))
        dr[border:-border, border:-border] = heatmap.copy()
        dr = cv2.GaussianBlur(dr, (gaussian_kernel, gaussian_kernel), 0)
        heatmap = dr[border:-border, border:-border].copy()
        heatmap *= orig_max / np.max(heatmap)
        # Log-likelihood
        heatmap = np.maximum(heatmap, 1e-10)
        heatmap = np.log(heatmap)
        # DARK
        if 1 < pred_x < width - 2 and 1 < pred_y < height - 2:
            dx = 0.5 * (heatmap[pred_y][pred_x + 1] - heatmap[pred_y][pred_x - 1])
            dy = 0.5 * (heatmap[pred_y + 1][pred_x] - heatmap[pred_y - 1][pred_x])
            dxx = 0.25 * (
                heatmap[pred_y][pred_x + 2]
                - 2 * heatmap[pred_y][pred_x]
                + heatmap[pred_y][pred_x - 2]
            )
            dxy = 0.25 * (
                heatmap[pred_y + 1][pred_x + 1]
                - heatmap[pred_y - 1][pred_x + 1]
                - heatmap[pred_y + 1][pred_x - 1]
                + heatmap[pred_y - 1][pred_x - 1]
            )
            dyy = 0.25 * (
                heatmap[pred_y + 2][pred_x]
                - 2 * heatmap[pred_y][pred_x]
                + heatmap[pred_y - 2][pred_x]
            )
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset_x, offset_y = np.squeeze(np.array(offset.T), axis=0)
                pred_x += offset_x
                pred_y += offset_y
        return pred_x, pred_y

    def _argmax_pts(self, heatmap):
        idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        pred_y, pred_x = map(float, idx)
        return pred_x, pred_y


def main():
    rospy.init_node("gaze_follow")
    ViTGazeFollow()
    if enable_record_video:
        try:
            rospy.spin()
        finally:
            video_writer.release()
    else:
        rospy.spin()


if __name__ == "__main__":
    main()
