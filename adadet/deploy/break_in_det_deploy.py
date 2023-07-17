# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
from adadet.utils.constants import StateCode
from adadet.utils.visualization import get_color

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks, ThirdParty
from modelscope.utils.logger import get_logger
from .deploy import DEPLOYS, Deploy

logger = get_logger()


@DEPLOYS.register_module('adadet_deploy', 'BreakInDet')
class BreakInDet(Deploy):

    def __init__(self, model_id, rules) -> None:
        super().__init__()
        self.rules = rules
        self.is_video = self.rules.get('is_video', False)
        self.frame_rate = self.rules.get('frame_rate', 1)
        self.det_thres = self.rules.get('det_thres', 0)
        self.region_polygon = self.rules.get('region_polygon', [[]])

        self.check_region_polygon()
        self.pipeline = pipeline(
            Tasks.domain_specific_object_detection,
            model=model_id,
            third_party=getattr(ThirdParty, 'ADADET', 'adadet'))
        self.result_dict = {}  # save detection result with frame idx as key

    def check_region_polygon(self):
        """check rule region config format.
        """
        if not isinstance(self.region_polygon,
                          list) or len(self.region_polygon) < 3:
            logger.error(
                'region_polygon should be at least 3 pairs of coords points, \
                    such as: [[x_1, y_1], [x_2, y_2], ..., [x_n, y_n]].')
            exit(StateCode.INVALID_REGION_POLYGON_FORMAT)

        # check if all point in a line
        xs = []
        ys = []
        for i in range(len(self.region_polygon)):
            xs.append(int(self.region_polygon[i][0]))
            ys.append(int(self.region_polygon[i][1]))

        if len(np.unique(np.array(xs))) == 1 or len(np.unique(
                np.array(ys))) == 1:
            logger.error('Points of region_polygon should not in a line.')
            exit(StateCode.INVALID_REGION_POLYGON_VALUE)

    def check_box_status(self, box) -> bool:
        """check if bbox center in region polygon.

        Args:
            box (list): [x1, y1, x2, y2]

        Returns:
            True: inside region polygon. False: not inside.
        """
        pts = np.array(self.region_polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cx = (box[2] + box[0]) / 2
        cy = (box[3] + box[1]) / 2
        center = (cx, cy)
        ret = cv2.pointPolygonTest(pts, center, False)
        if ret >= 0:
            return True
        return False

    def parse_frame_result(self, det_bboxes) -> dict:
        scores = det_bboxes.get(OutputKeys.SCORES)
        det_bboxes['alarms'] = []

        for i, score in enumerate(scores):
            if score >= self.det_thres:
                box = det_bboxes[OutputKeys.BOXES][i]
                if self.check_box_status(box):
                    det_bboxes['alarms'].append(True)
                else:
                    det_bboxes['alarms'].append(False)
        det_bboxes['alarms'] = np.array(det_bboxes['alarms'])
        return det_bboxes

    def __call__(self, input) -> dict:
        if self.is_video:
            res = self.process_video(input)
        else:
            res = self.process_image(input)
        return res

    def process_image(self, input, vis=False):
        res = self.pipeline(input)
        res = self.parse_frame_result(res)

        res = {
            0: res  # frame index with 0
        }
        return res

    def visualize(self, input_video, res, save_path) -> None:
        cap = cv2.VideoCapture(input_video)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            if 0 == frame_idx:
                size = (frame.shape[1], frame.shape[0])
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                video_writer = cv2.VideoWriter(save_path, fourcc,
                                               cap.get(cv2.CAP_PROP_FPS), size,
                                               True)
            if frame_idx in res.keys():
                labels = res[frame_idx][OutputKeys.LABELS]
                bboxes = res[frame_idx][OutputKeys.BOXES]
                scores = res[frame_idx][OutputKeys.SCORES]
                alarms = res[frame_idx]['alarms']

                # draw rule region mask
                pts = np.array(self.region_polygon, np.int32)
                img_h, img_w = frame.shape[:2]
                area_mask = np.zeros((img_h, img_w, 1), np.uint8)
                cv2.fillPoly(area_mask, [pts], 255)
                alpha = 0.3
                frame = np.array(frame).astype('float32')
                mask = area_mask[:, :, 0]
                color_mask = [0, 0, 255]
                idx = np.nonzero(mask)
                color_mask = np.array(color_mask)
                frame[idx[0], idx[1], :] *= 1.0 - alpha
                frame[idx[0], idx[1], :] += alpha * color_mask
                frame = np.array(frame).astype('uint8')

                # draw det results
                for i, box in enumerate(bboxes):
                    x1, y1, x2, y2 = box
                    color = get_color(i)
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), color, 2)
                    cv2.putText(frame,
                                f'{labels[i]}_{scores[i]:<.2f}_{alarms[i]}',
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            video_writer.write(frame)
            frame_idx += 1
        video_writer.release
        cap.release()
        return save_path

    def process_video(self, input):
        capture = cv2.VideoCapture(input)
        if not capture:
            logger.error(f'Open video {input} Failed!')
            exit(StateCode.INVALID_VIDEO_PATH)

        # Get Video infomation
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_number_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info('W:%d H: %d, Video fps: %d, Total frame number: %d' %
                    (width, height, fps, frame_number_total))
        frame_id = 0
        res = {}

        # process
        while (1):
            ret, frame = capture.read()
            if ret:
                if frame_id % self.frame_rate == 0:
                    res_single = self.pipeline(frame)
                    res_single = self.parse_frame_result(res_single)
                    res[frame_id] = res_single
                    if frame_id % 10 == 0:
                        logger.info('Processing frame idx: {}/{}'.format(
                            frame_id, frame_number_total))
                frame_id += 1
            else:
                break
        return res
