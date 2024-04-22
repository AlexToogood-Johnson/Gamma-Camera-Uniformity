# dicom_functions.py
"""Functions relating to the manpulation of DICOM images"""

########## IMPORTS ##########

import numpy as np
import os
import json
from typing import NoReturn
from scipy.signal import convolve2d
import struct

########## FUNCTIONS ##########


def get_dicom_data(filepath: str) -> list:
    """Reads the binary of a DICOM file"""

    with open(filepath, "rb") as dicom_file:
        data = dicom_file.read()
    return data


def decode_value(vr, value):
    """Decodes the value of a DICOM element based on its VR"""

    try:
        if vr in "AE AS CS DA DS DT LO LT PN SH ST TM UC UI UR UT UV":
            return value.decode("ascii")
        match vr:
            case "AT": return value.hex()
            case "FL": return struct.unpack(">f", value)[0]
            case "FD": return struct.unpack(">d", value)[0]
            case "IS": return int(value.decode("ascii"))
            case "SL": return struct.unpack(">i", value)[0]
            case "SS": return struct.unpack(">h", value)[0]
            case "SV": return struct.unpack(">q", value)[0]
            case "UL": return int.from_bytes(value, byteorder="little")
            case "US": return int.from_bytes(value, byteorder="little")
            case _: return value

    except Exception:
        return decode_value(vr, value[:-2])


def rearrange_tag(tag: str) -> str:
    """When reading tags from binary, they are in the wrong order"""

    tag = tag[2:4] + tag[0:2] + tag[6:] + tag[4:6]
    return tag


def parse_binary(data) -> dict:
    """Parses the raw binary from a DICOM file, and returns the metadata and raw image data"""

    parsed_data = {}
    image_data = []
    data = data[128:]  # Remove preamble
    data = data[4:]  # Remove prefix

    while len(data) > 0:
        tag = rearrange_tag(data[:4].hex())
        data = data[4:]
        vr = data[:2].decode("ascii")
        data = data[2:]
        if vr in "AE AS AT CS DA DS DT FL FD IS LO LT PN SH SL ST SS TM UI UL US":
            length = int.from_bytes(data[:2], byteorder="little")
            data = data[2:]
        else:
            data = data[2:]
            length = int.from_bytes(data[:4], byteorder="little")
            data = data[4:]
        value = decode_value(vr, data[:length])
        data = data[length:]

        if tag == "7fe00010":
            image_data = value
            data = []
            break
        else:
            parsed_data[tag] = [vr, length, value]

    elements = json.load(open("dicom_elements.json"))

    for key, value in parsed_data.items():
        if key in elements:
            parsed_data[key].append(elements[key])
        else:
            parsed_data[key].append("UNKOWN")

    return parsed_data, image_data


def decode_image_data(parsed_data: dict, image_data: list) -> np.ndarray:
    """Converts the raw image data into a 3D Numpy array"""

    bits_allocated = parsed_data["00280100"][2] // 8

    image_data = [int.from_bytes(image_data[i:i + bits_allocated], byteorder="little") for i in range(0, len(image_data), bits_allocated)]
    rows = parsed_data["00280010"][2]
    columns = parsed_data["00280011"][2]
    planes = parsed_data["00280008"][2]
    image_data = np.array(image_data).reshape((planes, rows, columns))

    return image_data


def load_dicom_image(filepath: str) -> tuple:
    """Returns the metadata and image data of a DICOM file"""

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Error: {filepath} not found")
    if not filepath.endswith(".dcm"):
        raise ValueError(f"Error: {filepath} is not a DICOM file")

    parsed_data, image_data = parse_binary(get_dicom_data(filepath))
    image_data = decode_image_data(parsed_data, image_data)

    return parsed_data, image_data


def read_config_file(key: str) -> list | NoReturn:
    """Returns the contents of config.json"""

    if not os.path.isfile("config.json"):
        raise FileNotFoundError(f"Error: config.json not found, please run setup.py")

    with open("config.json", "r") as config_file:
        data = json.load(config_file)
    if key not in data:
        raise KeyError(f"Error: key {key} not found in config.json")
    return data[key]


def edit_config_file(key: str, value: list) -> NoReturn:
    """Edits the contents of config.json"""

    if not os.path.isfile("config.json"):
        raise FileNotFoundError(f"Error: config.json not found, please run setup.py")

    with open("config.json", "r") as config_file:
        data = json.load(config_file)
    if key not in data:
        raise KeyError(f"Error: key {key} not found in config.json")
    data[key] = value
    with open("config.json", "w") as config_file:
        json.dump(data, config_file, indent=4)


def crop(array: np.ndarray, crop_size: int) -> np.ndarray | NoReturn:
    """Returns the central n x n x n crop of the input array"""

    if not len(array.shape) == 3:
        raise ValueError(f"Expected 3D array. Got {len(array.shape)}D array instead")

    center = array.shape[0] // 2
    start = center - crop_size // 2
    end = start + crop_size
    return array[start:end, start:end, start:end]


def apply_convolution(array: np.ndarray, convolution: list) -> np.ndarray | NoReturn:
    """Applies a smoothing convolution to a 2D Numpy array"""

    if not len(array.shape) == 3:
        raise ValueError(f"Expected 3D array. Got {len(array.shape)}D array instead")
    new_array = np.zeros_like(array)
    for i in range(array.shape[0]):
        new_array[i] = convolve2d(array[i], convolution, mode='same', boundary='fill', fillvalue=0)
    return new_array


def remove_image_edges(array: np.ndarray) -> np.ndarray | NoReturn:
    """Removes the edges of a 3D Numpy array"""

    if not len(array.shape) == 3:
        raise ValueError(f"Expected 3D array. Got {len(array.shape)}D array instead")

    radius = array.shape[0] // 2
    for layer in range(len(array)):
        for row in range(len(array[layer])):
            for column in range(len(array[layer][row])):
                if (radius - column) ** 2 + (radius - row) ** 2 > radius ** 2:
                    array[layer][row][column] = 10000

    return array


def max_pixel(array: list) -> int | NoReturn:
    """Returns the maximum value in a list, ignoring None values"""

    return max([value for value in array if value != 10000])


def min_pixel(array: list) -> int | NoReturn:
    """Returns the minimum value in a list, ignoring None values"""

    return min([value for value in array if value != 10000])
