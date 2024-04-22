import numpy as np
from PIL import Image, ImageDraw
import concurrent.futures


class UniformityLayer:
    def __init__(self, bin_data: list, circ_rad: int) -> None:
        self.bin_data = np.array(bin_data, dtype=np.uint8)
        self.circ_rad = circ_rad
        self.cropped_data = None

    def crop_to_circle(self) -> None:
        image = Image.fromarray(self.bin_data)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        transparent_image = Image.new('RGBA', image.size)
        transparent_image.paste(image, (0, 0))
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        center_x, center_y = image.width // 2, image.height // 2
        draw.ellipse((center_x - self.circ_rad, center_y - self.circ_rad,
                      center_x + self.circ_rad, center_y + self.circ_rad), fill=255)
        transparent_image.putalpha(mask)
        self.cropped_data = np.array(transparent_image)

    def _calculate_uniformity(self, slices):
        max_uniformity = 0
        for s in slices:
            if np.isnan(s).all():
                continue  # Skip entirely transparent slices
            valid_slice = s[~np.isnan(s)]
            if valid_slice.size >= 2:
                max_brightness = np.nanmax(valid_slice)
                min_brightness = np.nanmin(valid_slice)
                if max_brightness + min_brightness != 0:
                    uniformity = abs((max_brightness - min_brightness) / ((max_brightness + min_brightness) / 2) * 100)
                    max_uniformity = max(max_uniformity, uniformity)
        return max_uniformity

    def differential(self) -> float:
        if self.cropped_data is None:
            print("No data to process.")
            return 0

        alpha = self.cropped_data[:, :, 3] > 0
        grayscale = 0.299 * self.cropped_data[:, :, 0] + \
            0.587 * self.cropped_data[:, :, 1] + \
            0.114 * self.cropped_data[:, :, 2]
        grayscale[~alpha] = np.nan

        horizontal_slices = [grayscale[y, x:x + 5] for y in range(grayscale.shape[0]) for x in range(grayscale.shape[1] - 4)]
        vertical_slices = [grayscale[y:y + 5, x] for x in range(grayscale.shape[1]) for y in range(grayscale.shape[0] - 4)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._calculate_uniformity, horizontal_slices),
                executor.submit(self._calculate_uniformity, vertical_slices)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        return max(results)

    def integral(self) -> float:
        if self.cropped_data is None:
            print("No data to process.")
            return 0

        alpha = self.cropped_data[:, :, 3]
        grayscale = 0.299 * self.cropped_data[:, :, 0] + \
            0.587 * self.cropped_data[:, :, 1] + \
            0.114 * self.cropped_data[:, :, 2]
        grayscale[alpha == 0] = np.nan
        valid_grayscale = grayscale[~np.isnan(grayscale)]

        if valid_grayscale.size == 0:
            return 0

        max_brightness = np.nanmax(valid_grayscale)
        min_brightness = np.nanmin(valid_grayscale)

        integral_uniformity = abs((max_brightness - min_brightness) / ((max_brightness + min_brightness) / 2) * 100)
        return integral_uniformity
