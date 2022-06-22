from typing import Union, Literal

import cv2
import numpy as np
import numpy.typing as npt
from cv2 import VideoCapture

from ..arrays.array import SampledTimeView


class Video:
    def __init__(self, vid_name: str = None, cv2_vid=None):
        assert (vid_name is None) ^ (cv2_vid is None)
        self.__vid = cv2_vid or VideoCapture(vid_name)
        if not self.__vid.isOpened():
            if vid_name is None:
                raise RuntimeError(f'The provided cv2 video is not opened!')
            else:
                raise RuntimeError(f'Cannot open {vid_name}!')
        self.__current_cursor = 0

    def __enter__(self):
        return self

    def create_writer(self, path: str) -> 'Video':
        example_frame = self.read_frame(0)
        self.__set_cursor(0)
        return Video(cv2_vid=cv2.VideoWriter(path, int(self.__vid.get(cv2.CAP_PROP_FOURCC)), self.fps(),
                                             example_frame.shape[:2],
                                             bool(self.__vid.get(cv2.VIDEOWRITER_PROP_IS_COLOR))))

    @property
    def to_cv2(self):
        return self.__vid

    def __len__(self) -> int:
        return int(self.__vid.get(cv2.CAP_PROP_FRAME_COUNT))

    def fps(self) -> int:
        return int(self.__vid.get(cv2.CAP_PROP_FPS))

    def frame_inds(self) -> SampledTimeView:
        return SampledTimeView(0, self.fps())(np.arange(0, len(self), dtype=np.int32))

    def release(self) -> None:
        self.__vid.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __set_cursor(self, cursor_i: int) -> None:
        if cursor_i != self.__current_cursor:
            if cursor_i >= len(self):
                raise ValueError(f'Frame #{self.__current_cursor} does not exist.')
            self.__vid.set(cv2.CAP_PROP_POS_FRAMES, cursor_i)
            self.__current_cursor = cursor_i

    def read_frame(self, ind: int, error_on_overflow: bool = False) -> Union[npt.NDArray, Literal[-1]]:
        try:
            self.__set_cursor(ind)
        except ValueError:
            if error_on_overflow:
                raise
            else:
                return -1  # can't return None because None just means bad frame, not necessarily end of video
        ret, frame = self.__vid.read()
        if frame is not None and frame.ndim == 3 and frame.shape[-1] > 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if not ret:
        #     raise RuntimeError('Unexpected Error!')
        self.__current_cursor += 1
        return frame

    # returns None for corrupted frames or in case frame count is wrong (for the extra frames)
    def read(self, frame_inds: SampledTimeView = None, error_on_overflow: bool = False) -> npt.NDArray:
        if frame_inds is None:
            frame_inds = self.frame_inds()
        for ind in frame_inds.numpy:
            frame = self.read_frame(ind, error_on_overflow)
            if isinstance(frame, int) and frame == -1:
                return
            yield frame
