import json
import os
import cv2


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def parseJson(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    rois_xywh_list = []
    for shape in data['shapes']:
        if len(shape['points']) == 2:
            roi_xywh = [(shape['points'][0][0] + shape['points'][1][0]) / 2 / 2592,  # x_center
                        (shape['points'][0][1] + shape['points'][1][1]) / 2 / 1944,  # y_center
                        abs(shape['points'][1][0] - shape['points'][0][0]) / 2592,  # w
                        abs(shape['points'][1][1] - shape['points'][0][1]) / 1944]  # h

            rois_xywh_list.append(roi_xywh)

    return rois_xywh_list


input_data_dir = '/home/ancy/Desktop/yolov3_data/transfer'
root_output_dir = '/home/ancy/Desktop/yolov3_data/output'

output_img_dir = os.path.join(root_output_dir, 'images')
output_labels_dir = os.path.join(root_output_dir, 'labels')

cv2.namedWindow('img', 0)
all_folders = os.listdir(input_data_dir)
for folder in all_folders:
    folder_dir = os.path.join(input_data_dir, folder)
    if os.path.isdir(folder_dir):
        all_files = os.listdir(folder_dir)
        for file in all_files:
            file_path = os.path.join(folder_dir, file)
            if file_path.endswith('json'):
                data = parseJson(file_path)
                img = cv2.imread(file_path.replace('json', 'png'))
                out_img_path = os.path.join(output_img_dir, folder + '_' + file.replace('json', 'jpg'))
                out_label_path = os.path.join(output_labels_dir, folder + '_' + file.replace('json', 'txt'))
                with open(out_label_path, 'w') as f:
                    for i in data:
                        a = str(i)[1:-1].replace(',', '')
                        f.write(str(0) + ' ' + a)
                        f.write('\n')

                # print(0, *data[0])
                cv2.imwrite(out_img_path, img)
                cv2.imshow('img', img)
                cv2.waitKey(33)
