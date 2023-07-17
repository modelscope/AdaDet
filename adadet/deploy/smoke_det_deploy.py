# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import operator
import os
import os.path as osp

import cv2
import numpy as np
from adadet.utils.constants import StateCode

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks, ThirdParty
from modelscope.utils.logger import get_logger
from .deploy import DEPLOYS, Deploy

logger = get_logger()


@DEPLOYS.register_module('adadet_deploy', 'SmokeDetection')
class SmokeDetection(Deploy):

    def __init__(self, model_id, rules) -> None:
        super().__init__()
        human_det_model_id = model_id.get(
            'human_det', 'damo/cv_tinynas_human-detection_damoyolo')
        cigs_det_model_id = model_id.get(
            'cigarette_det',
            'damo/cv_tinynas_object-detection_damoyolo_cigarette')
        self.human_det_pipeline = pipeline(
            Tasks.domain_specific_object_detection,
            model=human_det_model_id,
            third_party=getattr(ThirdParty, 'ADADET', 'adadet'))
        self.cigs_det_pipeline = pipeline(
            Tasks.domain_specific_object_detection,
            model=cigs_det_model_id,
            third_party=getattr(ThirdParty, 'ADADET', 'adadet'))
        metric_type = rules.get('type', 'ioa')
        self.is_video = rules.get('is_video', False)
        self.frame_rate = rules.get('frame_rate', 1)
        self.metric_func = METRIC_FUNCS[metric_type]
        self.metric_thred = rules.get('threshold',
                                      100 if metric_type == 'dist' else 0.5)
        self.metric_com_func = operator.le if metric_type == 'dist' else operator.ge

    def _smoke_det(self, human_det_dict, cigs_det_dict, re_order=True):
        """smoke detection

        Args:
            human_det_dict (dict): modelscope detection style dict, including 3 keys: 'scores', 'labels', 'boxes'
            cigs_det_dict (dict): modelscope detection style dict, including 3 keys: 'scores', 'labels', 'boxes'

        Returns:
            dict: modelscope detection style dict, adding a 'smoke' key
                - smoke (list[dict]):
                    [
                        {'flag': 1, 'cigs': {'score': float, 'bbox': np.array([float,float,float,float])}},
                        {'flag': 0, 'cigs': None},
                        ...
                    ]
        """
        smoke_det_dict = copy.deepcopy(human_det_dict)
        smoke_det_dict['smoke'] = []

        human_num = len(human_det_dict.get('scores', []))
        cigs_num = len(cigs_det_dict.get('scores', []))
        # boundary cases (i.e., no human or no cigs)
        if human_num == 0 or cigs_num == 0:
            smoke_det_dict['smoke'] = [
                dict(flag=False, cigs=None) for _ in range(human_num)
            ]
            return smoke_det_dict

        # sort in descending order
        if re_order:
            human_order = human_det_dict['scores'].argsort()[::-1]
            cigs_order = cigs_det_dict['scores'].argsort()[::-1]
            human_det_dict['scores'] = human_det_dict['scores'][human_order]
            human_det_dict['labels'] = [
                human_det_dict['labels'][i] for i in human_order
            ]
            human_det_dict['boxes'] = human_det_dict['boxes'][human_order]
            cigs_det_dict['scores'] = cigs_det_dict['scores'][cigs_order]
            cigs_det_dict['labels'] = [
                cigs_det_dict['labels'][i] for i in cigs_order
            ]
            cigs_det_dict['boxes'] = cigs_det_dict['boxes'][cigs_order]

        metric_mat = self.metric_func(cigs_det_dict['boxes'],
                                      human_det_dict['boxes'])
        human_assign = [[False, None] for _ in range(human_num)]

        for i in range(cigs_num):
            max_id = np.argmax(metric_mat[i, :])
            flag = self.metric_com_func(metric_mat[i, max_id],
                                        self.metric_thred)
            if flag and (not human_assign[max_id][0]):
                human_assign[max_id][0] = True
                human_assign[max_id][1] = dict(
                    flag=True,
                    cigs=dict(
                        score=cigs_det_dict['scores'][i],
                        box=cigs_det_dict['boxes'][i]))

        for i in range(human_num):
            if human_assign[i][0]:
                smoke_det_dict['smoke'].append(human_assign[i][1])
            else:
                smoke_det_dict['smoke'].append(dict(flag=False, cigs=None))

        return smoke_det_dict

    def __call__(self, input) -> dict:
        if self.is_video:
            res = self.process_video(input)
        else:
            res = self.process_image(input)
        return res

    def process_image(self, input):
        human_det_res = self.human_det_pipeline(input)
        cigs_det_res = self.cigs_det_pipeline(input)
        smoke_det_res = self._smoke_det(human_det_res, cigs_det_res)
        res = {
            0:
            dict(
                human_res=human_det_res,
                cigs_res=cigs_det_res,
                smoke_res=smoke_det_res),  # frame index with 0
        }
        return res

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
        bar_interval = frame_number_total // 10
        logger.info(
            f'W: {width} H: {height}, Video fps: {fps}, Total frame number: {frame_number_total}'
        )
        frame_id = 0
        res = {}
        # process
        while (1):
            ret, frame = capture.read()
            if ret:
                if frame_id % self.frame_rate == 0:
                    human_det_res = self.human_det_pipeline(frame)
                    cigs_det_res = self.cigs_det_pipeline(frame)
                    smoke_det_res = self._smoke_det(human_det_res,
                                                    cigs_det_res)
                    res[frame_id] = dict(
                        human_res=human_det_res,
                        cigs_res=cigs_det_res,
                        smoke_res=smoke_det_res)
                frame_id += 1
                if frame_id % bar_interval == 0:
                    logger.info(
                        f'Processing frame idx: {frame_id}/{frame_number_total}'
                    )
            else:
                break
        return res

    def visualize(self, input_path, res, save_path) -> None:

        is_video = len(res) > 1
        os.makedirs(osp.dirname(save_path), exist_ok=True)

        if is_video and (not save_path.endswith('.avi')):
            logger.error('save_path must be end with .avi!')
            exit(StateCode.INVALID_SUFFIX)

        if not is_video:
            if isinstance(input_path, str):
                im = cv2.imread(input_path)
            elif isinstance(input_path, np.ndarray):
                im = input_path
            else:
                logger.error('input_path must be str or np.ndarray!')
                exit(StateCode.INVALID_IMAGE_PATH)

            for (score, label, bbox) in zip(res[0]['cigs_res']['scores'],
                                            res[0]['cigs_res']['labels'],
                                            res[0]['cigs_res']['boxes']):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 0, 0), 1)  # blue bbox for cigs
                im = cv2.putText(im, f'{label}:{score:.2f}',
                                 (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,
                                 1.2, (255, 0, 0), 1)

            for (score, label, bbox,
                 smoke) in zip(res[0]['smoke_res']['scores'],
                               res[0]['smoke_res']['labels'],
                               res[0]['smoke_res']['boxes'],
                               res[0]['smoke_res']['smoke']):
                x1, y1, x2, y2 = bbox
                if smoke['flag']:
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255),
                                  1)  # red bbox for smoking person
                    im = cv2.putText(im, f'{label}:{score:.2f}',
                                     (int(x1), int(y1)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                     (0, 0, 255), 1)
                else:
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0),
                                  1)  # green bbox for no smoking person
                    im = cv2.putText(im, f'{label}:{score:.2f}',
                                     (int(x1), int(y1)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                     (0, 255, 0), 1)
            total_person_num = len(res[0]['smoke_res']['smoke'])
            smoke_person_num = sum(x['flag']
                                   for x in res[0]['smoke_res']['smoke'])
            im = cv2.putText(
                im,
                f'Smoke number/Total number: {smoke_person_num}/{total_person_num}',
                (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imwrite(save_path, im)
        else:
            cap = cv2.VideoCapture(input_path)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if frame is None:
                    break
                if 0 == frame_idx:
                    size = (frame.shape[1], frame.shape[0])
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    video_writer = cv2.VideoWriter(save_path, fourcc,
                                                   cap.get(cv2.CAP_PROP_FPS),
                                                   size, True)
                if frame_idx in res.keys():
                    for (score, label,
                         bbox) in zip(res[frame_idx]['cigs_res']['scores'],
                                      res[frame_idx]['cigs_res']['labels'],
                                      res[frame_idx]['cigs_res']['boxes']):
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (255, 0, 0),
                                      1)  # blue bbox for cigs
                        frame = cv2.putText(frame, f'{label}:{score:.2f}',
                                            (int(x1), int(y1)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                            (255, 0, 0), 1)

                    for (score, label, bbox,
                         smoke) in zip(res[frame_idx]['smoke_res']['scores'],
                                       res[frame_idx]['smoke_res']['labels'],
                                       res[frame_idx]['smoke_res']['boxes'],
                                       res[frame_idx]['smoke_res']['smoke']):
                        x1, y1, x2, y2 = bbox
                        if smoke['flag']:
                            cv2.rectangle(frame, (int(x1), int(y1)),
                                          (int(x2), int(y2)), (0, 0, 255),
                                          1)  # red bbox for smoking person
                            frame = cv2.putText(frame, f'{label}:{score:.2f}',
                                                (int(x1), int(y1)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                                (0, 0, 255), 1)
                        else:
                            cv2.rectangle(
                                frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                (0, 255, 0),
                                1)  # green bbox for no smoking person
                            frame = cv2.putText(frame, f'{label}:{score:.2f}',
                                                (int(x1), int(y1)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                                (0, 255, 0), 1)
                    total_person_num = len(
                        res[frame_idx]['smoke_res']['smoke'])
                    smoke_person_num = sum(
                        x['flag']
                        for x in res[frame_idx]['smoke_res']['smoke'])
                    frame = cv2.putText(
                        frame,
                        f'Smoke number/Total number: {smoke_person_num}/{total_person_num}',
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                        1)
                video_writer.write(frame)
                frame_idx += 1
            video_writer.release
            cap.release()
        return save_path


def calculate_iou(array_a, array_b):
    """calculate the iou of two array

    Args:
        array_a (np.array): (N, 4), the 1st dim is number of bbox, the 2nd dim is the coordinate of (x1, y1, x2, y2)
        array_b (np.array): (M, 4), the 1st dim is number of bbox, the 2nd dim is the coordinate of (x1, y1, x2, y2)

    Returns:
        np.array: (N, M), the iou matrix of array_a and array_b
    """

    N = len(array_a)
    M = len(array_b)
    assert N > 0 and M > 0

    a_mat = array_a[:, np.newaxis, :]
    b_mat = array_b[np.newaxis, ...]
    # extend a and b to a matrix in the same shape
    a_mat = np.tile(a_mat, (1, M, 1))
    b_mat = np.tile(b_mat, (N, 1, 1))

    # calculate iou
    a_w = a_mat[..., 2] - a_mat[..., 0] + 1
    a_h = a_mat[..., 3] - a_mat[..., 1] + 1
    b_w = b_mat[..., 2] - b_mat[..., 0] + 1
    b_h = b_mat[..., 3] - b_mat[..., 1] + 1
    a_mat_areas = a_w * a_h
    b_mat_areas = b_w * b_h
    xx1 = np.maximum(a_mat[..., 0], b_mat[..., 0])
    yy1 = np.maximum(a_mat[..., 1], b_mat[..., 1])
    xx2 = np.minimum(a_mat[..., 2], b_mat[..., 2])
    yy2 = np.minimum(a_mat[..., 3], b_mat[..., 3])
    inter_w = np.maximum(0.0, xx2 - xx1 + 1)
    inter_h = np.maximum(0.0, yy2 - yy1 + 1)
    inter_areas = inter_w * inter_h

    iou = inter_areas / (a_mat_areas + b_mat_areas - inter_areas)

    return iou


def calculate_ioa(array_a, array_b):
    """calculate the ioa of two array

    Args:
        array_a (np.array): (N, 4), the 1st dim is number of bbox, the 2nd dim is the coordinate of (x1, y1, x2, y2)
        array_b (np.array): (M, 4), the 1st dim is number of bbox, the 2nd dim is the coordinate of (x1, y1, x2, y2)

    Returns:
        np.array: (N, M), the ioa matrix of array_a and array_b
    """

    N = len(array_a)
    M = len(array_b)
    assert N > 0 and M > 0

    a_mat = array_a[:, np.newaxis, :]
    b_mat = array_b[np.newaxis, ...]
    # extend a and b to a matrix in the same shape
    a_mat = np.tile(a_mat, (1, M, 1))
    b_mat = np.tile(b_mat, (N, 1, 1))

    # calculate ioa
    a_w = a_mat[..., 2] - a_mat[..., 0] + 1
    a_h = a_mat[..., 3] - a_mat[..., 1] + 1
    a_mat_areas = a_w * a_h
    # b_mat_areas = b_w * b_h
    xx1 = np.maximum(a_mat[..., 0], b_mat[..., 0])
    yy1 = np.maximum(a_mat[..., 1], b_mat[..., 1])
    xx2 = np.minimum(a_mat[..., 2], b_mat[..., 2])
    yy2 = np.minimum(a_mat[..., 3], b_mat[..., 3])
    inter_w = np.maximum(0.0, xx2 - xx1 + 1)
    inter_h = np.maximum(0.0, yy2 - yy1 + 1)
    inter_areas = inter_w * inter_h

    # ioa = inter_areas / (a_mat_areas + b_mat_areas - inter_areas)
    ioa = inter_areas / a_mat_areas

    return ioa


def calculate_dist(array_a, array_b):
    """calculate the dist of two array

    Args:
        array_a (np.array): (N, 4), the 1st dim is number of bbox, the 2nd dim is the coordinate of (x1, y1, x2, y2)
        array_b (np.array): (M, 4), the 1st dim is number of bbox, the 2nd dim is the coordinate of (x1, y1, x2, y2)

    Returns:
        np.array: (N, M), the dist matrix of array_a and array_b
    """

    N = len(array_a)
    M = len(array_b)
    assert N > 0 and M > 0

    a_mat = array_a[:, np.newaxis, :]
    b_mat = array_b[np.newaxis, ...]
    # extend a and b to a matrix in the same shape
    a_mat = np.tile(a_mat, (1, M, 1))
    b_mat = np.tile(b_mat, (N, 1, 1))
    a_c_mat = np.zeros((N, M, 2))
    b_c_mat = np.zeros((N, M, 2))

    # calculate dist
    # center point
    a_c_mat[..., 0] = (a_mat[..., 0] + a_mat[..., 2]) / 2
    a_c_mat[..., 1] = (a_mat[..., 1] + a_mat[..., 3]) / 2
    b_c_mat[..., 0] = (b_mat[..., 0] + b_mat[..., 2]) / 2
    b_c_mat[..., 1] = (b_mat[..., 1] + b_mat[..., 3]) / 2
    diff_c_mat = a_c_mat - b_c_mat

    dist = np.linalg.norm(diff_c_mat, axis=-1)

    return dist


METRIC_FUNCS = dict(
    dist=calculate_dist,
    ioa=calculate_ioa,
    iou=calculate_iou,
)
