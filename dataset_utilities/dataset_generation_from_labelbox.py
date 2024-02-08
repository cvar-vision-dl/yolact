import argparse
from datetime import datetime
import os
import random
import numpy as np

from labelbox_exporter.exporter import export_dataset
from format_unifier.file_types.type_labelbox import TypeLabelBox
from format_unifier.file_types.type_coco import TypeCoco
from format_unifier.file_manager import FileManager

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=str,
                    help="Output path to store resulting split dataset")
    ap.add_argument("-k", "--label-box-key", type=str,
                    help="Path to labelbox key file")
    ap.add_argument("-s", "--suffix", type=str,
                    help="suffix that images should contain [wide: _W, thermal: _T]",
                    default='_W')
    ap.add_argument("-n", "--dataset-name", type=str,
                    help="Output dataset name",
                    default='generic-dataset')
    ap.add_argument("-f", "--format", type=str,
                    help="output format [coco]",
                    default='coco')
    ap.add_argument("-d", "--working-dir", type=str,
                    help="Folder where intermmediate datasets will be stored",
                    default='/tmp')
    ap.add_argument("-a", "--annotations",
                    help="Download only annotations",
                    action='store_true')
    ap.add_argument("-sp", "--only-split",
                    help="Assuming datasets are stored in working dir, only split datasets",
                    action='store_true')

    args = vars(ap.parse_args())

    # TODO: move this configuration to a file
    dataset_structure = {
        'Cala_Mocha_Copilot': {
            'trainvalratio': 0.9,
        },
        'Copilot_paper_javier': {
            'trainvalratio': 0.9,
        }
    }

    print(f"Labelbox datasets structure to split {dataset_structure}")

    if not args['only_split']:
        # Downlaod datasets from label box
        for dataset_name in dataset_structure:
            dataset_path = os.path.join(args['working_dir'], dataset_name)
            os.system(f"mkdir -p {dataset_path}")
            export_dataset(args['label_box_key'], dataset_name, dataset_path)

    # Read datasets and parse annotations
    file_manager = FileManager()
    for idx, dataset_name in enumerate(dataset_structure):
        # Read annotations
        dataset_path = os.path.join(args['working_dir'], dataset_name)
        annotations = file_manager.read_and_transform(filename=os.path.join(dataset_path, 'annotations.json'),
                                                      type=TypeLabelBox)
        trainvalratio = float(dataset_structure[dataset_name]['trainvalratio'])
        # Select indices based on train val ratio
        k = int(len(annotations['annotations']) * trainvalratio)
        idxs_train = random.sample(range(len(annotations['annotations'])), k)
        idxs_val = list(set(range(len(annotations['annotations']))).symmetric_difference(set(idxs_train)))
        # Split annotations
        annotations_train = annotations.copy()
        annotations_train['annotations'] = dict(np.array(list(annotations['annotations'].items()))[idxs_train])
        annotations_val = annotations.copy()
        annotations_val['annotations'] = dict(np.array(list(annotations['annotations'].items()))[idxs_val])
        # Write out annotations
        file_manager.write_and_transform(annotations_train, TypeCoco,
                                         os.path.join(dataset_path, 'annotations_coco_train'))
        file_manager.write_and_transform(annotations_val, TypeCoco, os.path.join(dataset_path, 'annotations_coco_val'))
        # Compute joint annotations
        if idx == 0:
            joint_annotations_train = annotations.copy()  # Assuming same dataset format for all
            joint_annotations_train['annotations'] = {}
            joint_annotations_val = annotations.copy()
            joint_annotations_val['annotations'] = {}
        joint_annotations_train['annotations'].update(annotations_train['annotations'])
        joint_annotations_val['annotations'].update(annotations_val['annotations'])
    print(f"INFO: Train dataset: {len(joint_annotations_train['annotations'])} images")
    print(f"INFO: Val dataset: {len(joint_annotations_val['annotations'])} images")
    file_manager.write_and_transform(joint_annotations_train, TypeCoco, os.path.join(args['working_dir'], 'joint_annotations_coco_train'))
    file_manager.write_and_transform(joint_annotations_val, TypeCoco, os.path.join(args['working_dir'], 'joint_annotations_coco_val'))

# Copy dataset split to desired location
output_path = os.path.join(args['output'], args['dataset_name'] + '-split-' + datetime.now().strftime('%Y%m%d%H%M'))
print(f'INFO: Copying dataset to {output_path}')
os.system(f"mkdir -p {output_path}")
os.system(f"mv {os.path.join(args['working_dir'], 'joint_annotations_coco_train.json')} {os.path.join(output_path, 'joint_annotations_coco_train.json')}")
os.system(f"mv {os.path.join(args['working_dir'], 'joint_annotations_coco_val.json')} {os.path.join(output_path, 'joint_annotations_coco_val.json')}")
for dataset_name in dataset_structure:
    os.system(f"mv {os.path.join(os.path.join(args['working_dir'], dataset_name), '*.jpg')} {output_path} 2> /dev/null")
    os.system(f"mv {os.path.join(os.path.join(args['working_dir'], dataset_name), '*.JPG')} {output_path} 2> /dev/null")
    os.system(f"mv {os.path.join(os.path.join(args['working_dir'], dataset_name), '*.png')} {output_path} 2> /dev/null")
    os.system(f"mv {os.path.join(os.path.join(args['working_dir'], dataset_name), '*.PNG')} {output_path} 2> /dev/null")

print(f'INFO: Dataset copied  to {output_path}')
