import argparse
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

    if not args['only_split']:
        # Downlaod datasets from label box
        for dataset_name in dataset_structure:
            dataset_path = os.path.join(args['working_dir'], dataset_name)
            os.system(f"mkdir -p {dataset_path}")
            export_dataset(args['label_box_key'], dataset_name, dataset_path)

    # Read datasets and parse annotations
    for dataset_name in dataset_structure:
        # Read annotations
        dataset_path = os.path.join(args['working_dir'], dataset_name)
        file_manager = FileManager()
        annotations = file_manager.read_and_transform(filename=os.path.join(dataset_path, 'annotations.json'), type=TypeLabelBox)
        trainvalratio = float(dataset_structure[dataset_name]['trainvalratio'])
        # Select indices based on train val ratio
        k = int(len(annotations['annotations']) * trainvalratio)
        idxs_train = random.sample(range(len(annotations['annotations'])), k)
        idxs_val = list(set(range(len(annotations['annotations']))).symmetric_difference(set(idxs_train)))
        # Split annotations
        # TODO: Solve index bug and get a subset of dictionary properly
        annotations_train = annotations
        annotations_train['annotations'] = dict(np.array(list(annotations['annotations'].items()))[idxs_train])
        annotations_val = annotations
        annotations_val['annotations'] = dict(np.array(list(annotations['annotations'].items()))[idxs_val])
        # Write out annotations
        file_manager.write_and_transform(annotations_train, TypeCoco, os.path.join(dataset_path, 'annotations_coco_train'))
        file_manager.write_and_transform(annotations_val, TypeCoco, os.path.join(dataset_path, 'annotations_coco_val'))

# TODO: copy dataset to desired outputpath or just remove working directory and use output path for everything
