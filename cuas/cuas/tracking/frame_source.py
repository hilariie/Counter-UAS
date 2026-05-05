"""Pull-based narrow-FOV frame source backed by AirSim simGetImages."""
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import cosysairsim as airsim

from cuas.perception.frame_source import FrameMeta


class NarrowFovFrameSource:
    def __init__(self, client, vehicle_name: str = "Ownship",
                 camera_name: str = "narrow", fov_degrees: float = 12.0):
        self.client = client
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.fov_degrees = fov_degrees
        self._frame_id = -1
        self._req = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        self._logged_channels = False
        self.client.simSetCameraFov(camera_name, fov_degrees, vehicle_name=vehicle_name)

    def grab(self) -> Optional[Tuple[np.ndarray, FrameMeta]]:
        resps = self.client.simGetImages(self._req, vehicle_name=self.vehicle_name)
        if not resps:
            return None
        r = resps[0]
        if r.width == 0 or r.height == 0:
            return None
        buf = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        # Cosys-AirSim 3.3.0 returns RGB(A); convert to BGR for OpenCV / Ultralytics.
        if buf.size == r.height * r.width * 4:
            img = cv2.cvtColor(buf.reshape(r.height, r.width, 4), cv2.COLOR_RGBA2BGR)
            ch = 4
        elif buf.size == r.height * r.width * 3:
            img = cv2.cvtColor(buf.reshape(r.height, r.width, 3), cv2.COLOR_RGB2BGR)
            ch = 3
        else:
            return None
        if not self._logged_channels:
            print(f"[NarrowFovFrameSource] {self.camera_name}: {ch}-channel buffer, converted RGB→BGR")
            self._logged_channels = True
        self._frame_id += 1
        meta = FrameMeta(
            frame_id=self._frame_id,
            timestamp=time.time(),
            width=r.width,
            height=r.height,
        )
        return img, meta
