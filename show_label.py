import os.path

import cv2

from utils.plots import Annotator, colors


def simple_xywh2xyxy(x, width, height):
    x1 = int((x[0] - x[2] / 2) * width)
    y1 = int((x[1] - x[3] / 2) * height)
    x2 = int((x[0] + x[2] / 2) * width)
    y2 = int((x[1] + x[3] / 2) * height)
    return x1, y1, x2, y2


def show_label(img, label, class_file, show, project, name, thickness=2):
    # read img
    image = cv2.imread(img)
    # read label
    with open(label, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # real class file
    with open(class_file, 'r', encoding='utf-8') as f:
        classes = f.readlines()

    annotator = Annotator(image,
                          line_width=thickness,
                         )

    # plot label
    for line in lines:
        # resolve labels
        label = line.strip().split()
        class_index = int(label[0])  # class index
        x, y, w, h = map(float, label[1:])  # bbox cord, float

        # bbox cord convert to pixel cord
        height, width, _ = image.shape
        xyxy = simple_xywh2xyxy((x, y, w, h), width=width, height=height)

        # x1 = int((x - w / 2) * width)
        # y1 = int((y - h / 2) * height)
        # x2 = int((x + w / 2) * width)
        # y2 = int((y + h / 2) * height)
        # xyxy = (x1, y1, x2, y2)
        annotator.box_label(xyxy, classes[class_index].strip('\n'), color=colors(class_index, True))

    image = annotator.result()

    # save img
    if not os.path.exists(project):
        os.makedirs(project)
    cv2.imwrite(f'{project}{os.sep}{name}', image)

    # show img
    if show:
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    show_label(img='demo/with_label/13.jpg',
               label='demo/with_label/13.txt',
               class_file='demo/with_label/classes.txt',
               show=True,
               project='demo/with_label/',
               name='13_label.jpg')
