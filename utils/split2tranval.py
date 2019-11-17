import os

input_data_dir = '/home/ancy/Desktop/yolov3_data/output/images'
root_output_dir = '/home/ancy/Desktop/yolov3_data/output/train.txt'

all_files = os.listdir(input_data_dir)
for file in all_files:
    with open(root_output_dir, 'a+') as f:
        f.write(('data/custom/images/' + file))
        f.write('\n')
