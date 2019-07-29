import os
import cv2
import numpy


class data_manager:

    def __init__(self):
        pass

    def _load_image(self, path, image_type):
        image_list = []

        file_list = os.listdir(path)
        # 현재 위치의 이미지 데이터만 가져오기.
        images = [file for file in file_list if
                  file.endswith(".png") or file.endswith(".jpg")]

        images = sorted(images, key=lambda x: (len(x), x))

        for image in images:
            print("load image name : ", image)
            real_path = path + image
            one_image = cv2.imread(real_path, image_type)
            image_list.append(one_image)

        image_list = numpy.array(image_list)

        return image_list

    def get_all_images(self):
        images = self._load_image("./train_set/image/", cv2.IMREAD_COLOR)
        labels = self._load_image("./train_set/label/", cv2.IMREAD_GRAYSCALE)

        return images, labels
