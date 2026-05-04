import cv2
import numpy as np
import random
from pathlib import Path
import cv2


class CCTVEffect:
    def __init__(
        self,
        scale=0.9,
        fisheye_strength=0.0008,
        blur_kernel=3,
        dust_intensity=0.02,
        vignette_strength=0.6
    ):
        self.scale = scale
        self.fisheye_strength = fisheye_strength
        self.blur_kernel = blur_kernel
        self.dust_intensity = dust_intensity
        self.vignette_strength = vignette_strength

    # --------------------------
    # Core Effects
    # --------------------------

    def fake_zoom_out(self, frame):
        h, w = frame.shape[:2]
        new_w, new_h = int(w * self.scale), int(h * self.scale)

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros_like(frame)

        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

    def apply_fisheye(self, frame):
        h, w = frame.shape[:2]

        K = np.array([[w, 0, w/2],
                      [0, w, h/2],
                      [0, 0, 1]])

        D = np.array([self.fisheye_strength, self.fisheye_strength, 0, 0])

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        )

        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    def apply_blur(self, frame):
        return cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)

    def add_dust_haze(self, frame):
        h, w = frame.shape[:2]
        haze = np.zeros_like(frame, dtype=np.float32)

        num_particles = int(h * w * self.dust_intensity)

        for _ in range(num_particles):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            size = random.randint(1, 2)
            cv2.circle(haze, (x, y), size, (255, 255, 255), -1)

        haze = cv2.GaussianBlur(haze, (9, 9), 0)

        return cv2.addWeighted(frame.astype(np.float32), 0.9, haze, 0.1, 0).astype(np.uint8)

    def add_vignette(self, frame):
        h, w = frame.shape[:2]

        kernel_x = cv2.getGaussianKernel(w, w * self.vignette_strength)
        kernel_y = cv2.getGaussianKernel(h, h * self.vignette_strength)
        kernel = kernel_y @ kernel_x.T
        mask = kernel / kernel.max()

        vignette = frame.astype(np.float32)

        for i in range(3):
            vignette[:, :, i] *= mask

        return vignette.astype(np.uint8)

    # --------------------------
    # Full Pipeline
    # --------------------------

    def process(self, frame):
        frame = self.fake_zoom_out(frame)
        frame = self.apply_fisheye(frame)
        frame = self.apply_blur(frame)
        frame = self.add_dust_haze(frame)
        frame = self.add_vignette(frame)
        return frame

def main():
    INPUT_DIR = "/content/frames_ssim"
    OUTPUT_DIR = "/content/cctv_frames1"

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    effect = CCTVEffect(
        scale=0.8,
        fisheye_strength=0.0008,
        blur_kernel=3,
        dust_intensity=0.01,
        vignette_strength=0.5
    )


    for img_path in Path(INPUT_DIR).glob("*"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        processed = effect.process(frame)

        out_path = Path(OUTPUT_DIR) / img_path.name
        cv2.imwrite(str(out_path), processed)



if __name__ == "__main__":
    main()