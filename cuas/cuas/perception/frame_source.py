"""Pull-based wide-FOV frame source backed by AirSim simGetImages."""
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import cosysairsim as airsim


@dataclass(frozen=True)
class FrameMeta:
    frame_id: int
    timestamp: float
    width: int
    height: int


class WideFovFrameSource:
    def __init__(self, client, vehicle_name: str = "Ownship",
                 camera_name: str = "wide", fov_degrees: float = 90.0):
        self.client = client
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.fov_degrees = fov_degrees
        self._frame_id = -1
        self._req = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        self._logged_channels = False
        # Force FOV via API — settings.json FOV_Degrees is unreliable.
        self.client.simSetCameraFov(camera_name, fov_degrees, vehicle_name=vehicle_name)

    def grab(self) -> Optional[Tuple[np.ndarray, FrameMeta]]:
        resps = self.client.simGetImages(self._req, vehicle_name=self.vehicle_name)
        if not resps:
            return None
        r = resps[0]
        if r.width == 0 or r.height == 0:
            return None
        buf = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        # Cosys-AirSim 3.3.0 returns RGB(A) (not BGR(A) as Microsoft AirSim docs claim).
        # Convert to BGR so OpenCV / Ultralytics receive what they expect.
        if buf.size == r.height * r.width * 4:
            img = cv2.cvtColor(buf.reshape(r.height, r.width, 4), cv2.COLOR_RGBA2BGR)
            ch = 4
        elif buf.size == r.height * r.width * 3:
            img = cv2.cvtColor(buf.reshape(r.height, r.width, 3), cv2.COLOR_RGB2BGR)
            ch = 3
        else:
            return None
        if not self._logged_channels:
            print(f"[WideFovFrameSource] {self.camera_name}: {ch}-channel buffer, converted RGB→BGR")
            self._logged_channels = True
        self._frame_id += 1
        meta = FrameMeta(
            frame_id=self._frame_id,
            timestamp=time.time(),
            width=r.width,
            height=r.height,
        )
        return img, meta
