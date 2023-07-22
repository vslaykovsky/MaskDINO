# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager
 
    
HUBMAP_CATEGORIES = [{
      "id": 1,
      "name": "blood_vessel"
    },
    # {
    #   "id": 2,
    #   "name": "glomerulus"
    # },
    # {
    #   "id": 3,
    #   "name": "unsure"
    # }
]


def get_predefined_splits(dir):
    return {
        "hubmap_instance_train": (
            f"{dir}/train/data",
            f"{dir}/train/labels.json",
        ),
        "hubmap_instance_val": (
            f"{dir}/val/data",
            f"{dir}/val/labels.json",
        ),
    }


def _get_hubmap_instances_meta():
    thing_ids = [k["id"] for k in HUBMAP_CATEGORIES]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in HUBMAP_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_hubmap_instance(root):
    for dir in os.listdir(root):
        print(dir)
        if os.path.isdir(os.path.join(root, dir)) and not dir.startswith('.'):
            for key, (image_root, json_file) in get_predefined_splits(dir).items():
                # Assume pre-defined datasets live in `./datasets`.
                register_coco_instances(
                    key,
                    _get_hubmap_instances_meta(),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hubmap_instance(_root)
