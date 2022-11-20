import argparse
import json
import os
import shutil

import cv2
import numpy as np

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

PATH_TO_DATA = config['paths']['path_to_data']
PATH_TO_SAVE = config['paths']['path_to_save']

NEXT_IMAGE_BUTTON = config['buttons']['next_image_button'].lower()
PREV_IMAGE_BUTTON = config['buttons']['prev_image_button'].lower()
REMOVE_IMAGE_BUTTON = config['buttons']['remove_image_button'].lower()
QUIT_BUTTON = config['buttons']['quit_button'].lower()
DEFAULT_SETTINGS_BUTTON = config['buttons']['default_settings_button'].lower()

MAX_SCALE_SIZE = config['settings']['max_scale_size']
DEFAULT_GAMMA = config['settings']['default_gamma']
MAX_GAMMA = config['settings']['max_gamma']
DEFAULT_CONTRAST = config['settings']['default_contrast']
DEFAULT_BRIGHTNESS = config['settings']['default_brightness']
BACKGROUND_SIZE = config['settings']['background_size']

CLASSES = config['classes']


class ButtonError(Exception):
    pass


def resize_image(image, scale_coefficient, background_size):
    width = int(image.shape[1] * scale_coefficient)
    width = min(width, background_size)
    height = int(image.shape[0] * scale_coefficient)
    height = min(height, background_size)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def brightness_and_contrast(image, brightness=0, contrast=0):
    new_image = image * (contrast / 127 + 1) - contrast + brightness
    new_image = np.clip(new_image, 0, 255)
    new_image = np.uint8(new_image)
    return new_image


def is_image(name):
    if name.endswith('.jpg') or \
            name.endswith('.jpeg') or \
            name.endswith('.bmp') or \
            name.endswith('.png') or \
            name.endswith('.gif'):
        return name
    else:
        return False


def get_classes_buttons(classes):
    values = classes.values()
    buttons = []
    for button in values:
        buttons.append(button['button'].lower())

    return buttons


def nothing(x):
    pass


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=False, default=1, type=int, help="Start index")
    args = parser.parse_args()
    return args


def check_buttons_duplicate(all_buttons):
    if len(set(all_buttons)) != len(all_buttons):
        error_message = 'Some keys are used more than once, check config.json\n'
        count_of_buttons = {button: all_buttons.count(button) for button in all_buttons}
        for button, count in count_of_buttons.items():
            if count > 1:
                error_message += f"Key '{button}' is used {count} times\n"

        raise ButtonError(error_message)


def centering_image(image, background_size):
    background = np.zeros((background_size, background_size, 3), np.uint8)
    left_offset_x = (background_size - image.shape[0]) // 2
    left_offset_y = (background_size - image.shape[1]) // 2
    right_offset_x = left_offset_x + image.shape[0]
    right_offset_y = left_offset_y + image.shape[1]
    background[left_offset_x:right_offset_x, left_offset_y:right_offset_y] = image
    return background


def set_default_settings():
    cv2.setTrackbarPos('Scale', 'Image', 1)
    cv2.setTrackbarPos('Gamma', 'Image', DEFAULT_GAMMA)
    cv2.setTrackbarPos('Brightness', 'Image', DEFAULT_BRIGHTNESS)
    cv2.setTrackbarPos('Contrast', 'Image', DEFAULT_CONTRAST)


def make_class_dirs(classes_path):
    for class_path in classes_path:
        os.makedirs(class_path, exist_ok=True)


def change_image(current_index, images_count, next_image):
    if next_image:
        if current_index + 1 == images_count:
            current_index = 0
        else:
            current_index += 1

    else:
        if current_index == 0:
            current_index = images_count - 1
        else:
            current_index -= 1

    return current_index


def window_initialization():
    cv2.namedWindow('Image')
    cv2.createTrackbar('Scale', 'Image', 1, MAX_SCALE_SIZE, nothing)
    cv2.createTrackbar('Gamma', 'Image', DEFAULT_GAMMA, MAX_GAMMA, nothing)

    cv2.createTrackbar('Brightness', 'Image', DEFAULT_BRIGHTNESS, 255, nothing)
    cv2.setTrackbarMin('Brightness', 'Image', -127)
    cv2.setTrackbarMax('Brightness', 'Image', 127)
    cv2.setTrackbarPos('Brightness', 'Image', DEFAULT_BRIGHTNESS)

    cv2.createTrackbar('Contrast', 'Image', DEFAULT_CONTRAST, 255, nothing)
    cv2.setTrackbarMin('Contrast', 'Image', -127)
    cv2.setTrackbarMax('Contrast', 'Image', 127)
    cv2.setTrackbarPos('Contrast', 'Image', DEFAULT_CONTRAST)


if __name__ == '__main__':
    args = load_args()

    CLASSES_NAMES = list(CLASSES.keys())
    CLASSES_BUTTONS = get_classes_buttons(CLASSES)

    base_buttons = [NEXT_IMAGE_BUTTON, PREV_IMAGE_BUTTON, REMOVE_IMAGE_BUTTON, QUIT_BUTTON, DEFAULT_SETTINGS_BUTTON]
    all_buttons = base_buttons + CLASSES_BUTTONS
    check_buttons_duplicate(all_buttons)

    CLASSES_PATHS = {class_name: os.path.join(PATH_TO_SAVE, class_name)
                     for class_name in CLASSES_NAMES}

    make_class_dirs(CLASSES_PATHS.values())

    images = [os.path.join(PATH_TO_DATA, name)
              for name in os.listdir(PATH_TO_DATA)]
    images = list(filter(is_image, images))

    window_initialization()

    index = args.i - 1
    while True:
        current_image = images[index]
        image_basename = os.path.basename(current_image)
        image = cv2.imread(current_image)

        brightness_coefficient = cv2.getTrackbarPos('Brightness', 'Image')
        contrast_coefficient = cv2.getTrackbarPos('Contrast', 'Image')

        image = brightness_and_contrast(image, brightness=brightness_coefficient,
                                        contrast=contrast_coefficient)
        scale_coefficient = cv2.getTrackbarPos('Scale', 'Image')

        if scale_coefficient > 1:
            image = resize_image(image, scale_coefficient, BACKGROUND_SIZE)

        if image.shape[0] < BACKGROUND_SIZE and image.shape[1] < BACKGROUND_SIZE:
            image = centering_image(image, BACKGROUND_SIZE)

        gamma_coefficient = cv2.getTrackbarPos('Gamma', 'Image')
        if gamma_coefficient > 1:
            image = adjust_gamma(image, gamma_coefficient)

        cv2.imshow("Image", image)

        try:
            button = chr(cv2.waitKey(1)).lower()
        except ValueError:
            continue

        if button == PREV_IMAGE_BUTTON:
            index = change_image(index, len(images), False)
            print(f'[PREV] Image {index + 1}/{len(images)} ({image_basename})')

        elif button == NEXT_IMAGE_BUTTON:
            index = change_image(index, len(images), True)
            print(f'[NEXT] Image {index + 1}/{len(images)} ({image_basename})')

        elif button == REMOVE_IMAGE_BUTTON:
            os.remove(current_image)
            images.pop(index)
            print(f'[REMOVE] Image {index + 1} has been removed ({image_basename})')

        elif button in CLASSES_BUTTONS:
            image_index = index + 1
            image_name = f'{image_index}.jpg'
            class_name = CLASSES_NAMES[CLASSES_BUTTONS.index(button)]
            class_path = CLASSES_PATHS[class_name]
            new_path = os.path.join(class_path, image_name)
            shutil.copy(current_image, new_path)
            index = change_image(index, len(images), True)
            print(f'[SAVE] Image {index + 1} saved as {image_name} (class: {class_name}) ({image_basename})')

        elif button == DEFAULT_SETTINGS_BUTTON:
            set_default_settings()

        elif button == QUIT_BUTTON:
            break

print('[EXIT] Script has been closed')
