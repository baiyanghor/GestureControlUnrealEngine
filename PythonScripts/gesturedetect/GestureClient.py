#
# By Yang 'Ocean' He
#
import sys
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import category as category_module
from enum import Enum, StrEnum
import numpy as np
import cv2
import mediapipe as mp
import time
import asyncio
from typing import List, Dict
import pathlib

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MEASURE_THRESHOLD = 0.045
MEASURE_THRESHOLD_Z = 0.55
MEASURE_THRESHOLD_PALM_Z = 0.004
Z_AMPLIFIER = 1000000


class GESTURE(Enum):
    GESTURE_NONE = 0
    GESTURE_FIST_DOWN = 1
    GESTURE_INDEX_FINGER_RIGHT = 2
    GESTURE_INDEX_FINGER_LEFT = 3
    GESTURE_PALM_PUSH = 4


class MOVEMENT_DIR(Enum):
    MD_NONE = 0
    MD_LEFT = 1
    MD_RIGHT = 2
    MD_UP = 3
    MD_DOWN = 4
    MD_FORWARD = 5
    MD_BACKWARD =6


class Hand_Movement():
    x_direction: MOVEMENT_DIR
    y_direction: MOVEMENT_DIR
    z_direction: MOVEMENT_DIR


class GestureDetecting(object):
    UEIPAddress: str = ""
    UEPort: int = 0
    netWriter: asyncio.StreamWriter = None
    netReader: asyncio.StreamReader = None
    detect_result = None
    framesCache: int = 3
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmark = solutions.hands.HandLandmark

    def __init__(self, up_ip_address, ue_port):
        self.UEIPAddress = up_ip_address
        self.UEPort = ue_port

    def __str__(self) -> str:
        return "Gesture Detector!"

    async def initial_connection(self) -> bool:
        self.netReader, self.netWriter = await asyncio.open_connection(self.UEIPAddress, self.UEPort)
        if self.netReader and self.netWriter:
            return True
        else:
            return False

    async def single_thread_send(self, message):
        try:
            reader, writer = await asyncio.open_connection(self.UEIPAddress, self.UEPort)
            writer.write(message.encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except WindowsError:
            print(f"Can't access target Address!")
            sys.exit()

    def get_gesture(self, pose_list: Dict) -> GESTURE:
        #
        # Detect index finger and palm movement only in initial version
        #

        class PoseFilter(StrEnum):
            CLOSED_FIST = 'Closed_Fist'
            OPEN_PALM = 'Open_Palm'
            POINTING_UP = 'Pointing_Up'

        def pose_aligned(poses: List[str]):
            a_pose = poses[0]
            for idx in range(1, len(poses)):
                if a_pose == poses[idx]:
                    continue
                else:
                    return False

            return True

        def index_finger_movement_direction(x_list: List[float], y_list: List[float], z_list: List[float]) -> Hand_Movement:
            direction = Hand_Movement()
            direction.x_direction = MOVEMENT_DIR.MD_NONE
            if abs(z_list[0] - z_list[-1]) < MEASURE_THRESHOLD_Z:
                if x_list[-1] > x_list[0] + MEASURE_THRESHOLD:
                    direction.x_direction = MOVEMENT_DIR.MD_RIGHT
                elif x_list[-1] + MEASURE_THRESHOLD < x_list[0]:
                    direction.x_direction = MOVEMENT_DIR.MD_LEFT

                if y_list[0] > y_list[-1] + MEASURE_THRESHOLD:
                    direction.y_direction = MOVEMENT_DIR.MD_UP
                elif y_list[0] + MEASURE_THRESHOLD < y_list[-1]:
                    direction.y_direction = MOVEMENT_DIR.MD_DOWN

            return direction

        def palm_movement_direction(z_list: List[float]) -> Hand_Movement:
            direction = Hand_Movement()
            direction.z_direction = MOVEMENT_DIR.MD_NONE
            if z_list[-1] > z_list[0] + MEASURE_THRESHOLD_PALM_Z:
                direction.z_direction = MOVEMENT_DIR.MD_FORWARD
            if z_list[-1] + MEASURE_THRESHOLD_PALM_Z < z_list[0]:
                direction.z_direction = MOVEMENT_DIR.MD_BACKWARD

            return direction

        poses = []
        x_index_tip_list = []
        y_index_tip_list = []
        z_wrist_list = []
        gesture_result: GESTURE = GESTURE.GESTURE_NONE

        for key, gesture in pose_list.items():
            in_hand_landmarks:  List[landmark_module.NormalizedLandmark] = gesture[0]
            in_hand_gesture:  category_module.Category = gesture[1]

            poses.append(in_hand_gesture.category_name)

            index_finger_tip = in_hand_landmarks[self.HandLandmark.INDEX_FINGER_TIP]
            wrist = in_hand_landmarks[self.HandLandmark.WRIST]
            x_index_tip_list.append(index_finger_tip.x)
            y_index_tip_list.append(index_finger_tip.y)
            z_wrist_list.append(wrist.z * Z_AMPLIFIER)

        if len(poses) < self.framesCache - 1:
            return gesture_result

        if pose_aligned(poses):
            if poses[0] == PoseFilter.POINTING_UP.value:
                index_finger_direction = index_finger_movement_direction(x_index_tip_list, y_index_tip_list, z_wrist_list)
                if index_finger_direction.x_direction == MOVEMENT_DIR.MD_LEFT:
                    gesture_result = GESTURE.GESTURE_INDEX_FINGER_LEFT
                if index_finger_direction.x_direction == MOVEMENT_DIR.MD_RIGHT:
                    gesture_result = GESTURE.GESTURE_INDEX_FINGER_RIGHT

            if poses[0] == PoseFilter.OPEN_PALM.value:
                palm_direction = palm_movement_direction(z_wrist_list)
                if palm_direction.z_direction == MOVEMENT_DIR.MD_FORWARD:
                    gesture_result = GESTURE.GESTURE_PALM_PUSH

        return gesture_result

    def draw_landmarks_on_image(self, rgb_image, detection_result, gesture: GESTURE):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        gestures_list = detection_result.gestures
        annotated_image = np.copy(rgb_image)

        # Loop through hands
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            gestures = gestures_list[idx]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )

            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) + MARGIN
            wrist = hand_landmarks[self.HandLandmark.WRIST]
            cv2.putText(annotated_image,
                        f"{gestures[0].category_name}: {gesture}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE,
                        HANDEDNESS_TEXT_COLOR,
                        FONT_THICKNESS,
                        cv2.LINE_AA)

        return annotated_image

    def print_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.detect_result = result

    def execute(self):
        exe_path = pathlib.Path(__file__).parent.resolve()
        options = self.GestureRecognizerOptions(
            num_hands=1,
            base_options=self.BaseOptions(model_asset_path=f"{exe_path}\\models\\gesture_recognizer.task"),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result
        )

        with self.GestureRecognizer.create_from_options(options) as recognizer:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            cap.set(cv2.CAP_PROP_FPS, 30)

            gesture_list = dict()
            timestamp = 0
            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                if not ret:
                    print('Ignoring empty frame')
                    break

                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                recognizer.recognize_async(mp_image, timestamp)

                if self.detect_result:
                    cache_number = timestamp % self.framesCache
                    gesture = GESTURE.GESTURE_NONE
                    if timestamp > self.framesCache and len(self.detect_result.hand_landmarks) > 0:
                        if cache_number == 0:
                            gesture = self.get_gesture(gesture_list)
                            gesture_data = f"Gesture:{gesture.value}:{timestamp}"
                            asyncio.run(self.single_thread_send(gesture_data))
                            print(f"Gesture:{gesture}:{timestamp}")
                            gesture_list.clear()
                        else:
                            gesture_list[cache_number] = (self.detect_result.hand_landmarks[-1],
                                                          self.detect_result.gestures[0][-1])

                    annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), self.detect_result, gesture)

                    cv2.imshow('Gesture Detecting', annotated_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('Closing Stream')
                    break

            # asyncio.run(self.close_connection())
            cap.release()
            cv2.destroyAllWindows()
