# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


def try_to_find(file, return_dir=False, search_path=['./DATASET', './OUTPUT', './data', './MODEL']):
    if not file:
        return file

    if file.startswith('catalog://'):
        return file

    DATASET_PATH = ['./']
    if 'DATASET' in os.environ:
        DATASET_PATH.append(os.environ['DATASET'])
    DATASET_PATH += search_path

    for path in DATASET_PATH:
        if os.path.exists(os.path.join(path, file)):
            if return_dir:
                return path
            else:
                return os.path.join(path, file)

    print('Cannot find {} in {}'.format(file, DATASET_PATH))
    exit(1)


class DatasetCatalog(object):
    DATASETS = {
        # pretrained grounding dataset
        # mixed vg and coco
        "mixed_train": {
            "coco_img_dir": "refcoco/train2014",
            "vg_img_dir": "gqa/images",
            "ann_file": "mdetr_annotations/final_mixed_train.json",
        },
        "mixed_train_no_coco": {
            "coco_img_dir": "refcoco/train2014",
            "vg_img_dir": "gqa/images",
            "ann_file": "mdetr_annotations/final_mixed_train_no_coco.json",
        },

        # flickr30k
        "flickr30k_train": {
            "img_folder": "flickr30k/flickr30k_images/train",
            "ann_file": "mdetr_annotations/final_flickr_separateGT_train.json",
            "is_train": True
        },
        "flickr30k_val": {
            "img_folder": "flickr30k/flickr30k_images/val",
            "ann_file": "mdetr_annotations/final_flickr_separateGT_val.json",
            "is_train": False
        },
        "flickr30k_test": {
            "img_folder": "flickr30k/flickr30k_images/test",
            "ann_file": "mdetr_annotations/final_flickr_separateGT_test.json",
            "is_train": False
        },

        # refcoco
        "refexp_all_val": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/final_refexp_val.json",
            "is_train": False
        },

        "refcoco_train": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco_train.json",
            "is_train": True
        },

        "refcoco_val": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco_val.json",
            "is_train": False
        },

        "refcoco_real_val": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco_val.json",
            "is_train": False
        },

        "refcoco_testA": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco_testA.json",
            "is_train": False
        },

        "refcoco_testB": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco_testB.json",
            "is_train": False
        },

        "refcoco+_train": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco+_train.json",
            "is_train": True
        },

        "refcoco+_val": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco+_val.json",
            "is_train": False
        },

        "refcoco+_testA": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco+_testA.json",
            "is_train": False
        },

        "refcoco+_testB": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcoco+_testB.json",
            "is_train": False
        },

        "refcocog_train": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcocog_train.json",
            "is_train": True
        },

        "refcocog_val": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcocog_val.json",
            "is_train": False
        },

        "refcocog_test": {
            "img_dir": "refcoco/train2014",
            "ann_file": "mdetr_annotations/finetune_refcocog_test_corrected.json",
            "is_train": False
        },

        # gqa
        "gqa_val": {
            "img_dir": "gqa/images",
            "ann_file": "mdetr_annotations/final_gqa_val.json",
            "is_train": False
        },

        # phrasecut
        "phrasecut_train": {
            "img_dir": "gqa/images",
            "ann_file": "mdetr_annotations/finetune_phrasecut_train.json",
            "is_train": True
        },

        # caption
        "bing_caption_train": {
            "yaml_path": "BingData/predict_yaml",
            "yaml_name": "dreamstime_com_dyhead_objvg_e39",
            "yaml_name_no_coco": "dreamstime_com_Detection_Pretrain_NoCOCO_Packed125",
            "is_train": True,
        },

        # od to grounding
        # coco tsv
        "coco_dt_train": {
            "dataset_file": "coco_dt",
            "yaml_path": "coco_tsv/coco_obj.yaml",
            "is_train": True,
        },
        "COCO_odinw_train_8copy_dt_train": {
            "dataset_file": "coco_odinw_dt",
            "yaml_path": "coco_tsv/COCO_odinw_train_8copy.yaml",
            "is_train": True,
        },
        "COCO_odinw_val_dt_train": {
            "dataset_file": "coco_odinw_dt",
            "yaml_path": "coco_tsv/COCO_odinw_val.yaml",
            "is_train": False,
        },
        # lvis tsv
        "lvisv1_dt_train": {
            "dataset_file": "lvisv1_dt",
            "yaml_path": "coco_tsv/LVIS_v1_train.yaml",
            "is_train": True,
        },
        "LVIS_odinw_train_8copy_dt_train": {
            "dataset_file": "coco_odinw_dt",
            "yaml_path": "coco_tsv/LVIS_odinw_train_8copy.yaml",
            "is_train": True,
        },
        # object365 tsv
        "object365_dt_train": {
            "dataset_file": "object365_dt",
            "yaml_path": "Objects365/objects365_train_vgoiv6.cas2000.yaml",
            "is_train": True,
        },
        "object365_odinw_2copy_dt_train": {
            "dataset_file": "object365_odinw_dt",
            "yaml_path": "Objects365/objects365_train_odinw.cas2000_2copy.yaml",
            "is_train": True,
        },
        "objects365_odtsv_train": {
            "dataset_file": "objects365_odtsv",
            "yaml_path": "Objects365/train.cas2000.yaml",
            "is_train": True,
        },
        "objects365_odtsv_val": {
            "dataset_file": "objects365_odtsv",
            "yaml_path": "Objects365/val.yaml",
            "is_train": False,
        },

        # ImagetNet OD
        "imagenetod_train_odinw_2copy_dt": {
            "dataset_file": "imagenetod_odinw_dt",
            "yaml_path": "imagenet_od/imagenetod_train_odinw_2copy.yaml",
            "is_train": True,
        },

        # OpenImage OD
        "oi_train_odinw_dt": {
            "dataset_file": "oi_odinw_dt",
            "yaml_path": "openimages_v5c/oi_train_odinw.cas.2000.yaml",
            "is_train": True,
        },

        # vg tsv
        "vg_dt_train": {
            "dataset_file": "vg_dt",
            "yaml_path": "visualgenome/train_vgoi6_clipped.yaml",
            "is_train": True,
        },

        "vg_odinw_clipped_8copy_dt_train": {
            "dataset_file": "vg_odinw_clipped_8copy_dt",
            "yaml_path": "visualgenome/train_odinw_clipped_8copy.yaml",
            "is_train": True,
        },
        "vg_vgoi6_clipped_8copy_dt_train": {
            "dataset_file": "vg_vgoi6_clipped_8copy_dt",
            "yaml_path": "visualgenome/train_vgoi6_clipped_8copy.yaml",
            "is_train": True,
        },

        # coco json
        "coco_grounding_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json",
            "is_train": True,
        },

        "lvis_grounding_train": {
            "img_dir": "coco",
            "ann_file": "coco/annotations/lvis_od_train.json"
        },

        "lvis_evaluation_val":{
            "img_dir": "lvis/coco2017",
            "ann_file": "lvis/lvis_v1_minival_inserted_image_name.json",
            "is_train": False,
        },

        # legacy detection dataset
        "hsd_v001": {
            "img_dir": "hsd/20170901_Detection_HeadShoulder.V001/RawImages",
            "ann_file": "hsd/HSD_V001.json"
        },
        "hsd_hddb": {
            "img_dir": "hddb/Images",
            "ann_file": "hddb/HDDB.json"
        },
        "opencoco_train": {
            "img_dir": "openimages/train",
            "ann_file": "openimages/opencoco_train.json"
        },
        "opencoco_val": {
            "img_dir": "openimages/val",
            "ann_file": "openimages/opencoco_val.json"
        },
        "opencoco_test": {
            "img_dir": "openimages/test",
            "ann_file": "openimages/opencoco_test.json"
        },
        "openhuman_train": {
            "img_dir": "openimages/train",
            "ann_file": "openimages/openhuman_train.json"
        },
        "openhuman_val": {
            "img_dir": "openimages/val",
            "ann_file": "openimages/openhuman_val.json"
        },
        "openhuman_test": {
            "img_dir": "openimages/test",
            "ann_file": "openimages/openhuman_test.json"
        },
        "opencrowd_train": {
            "img_dir": "openimages/train",
            "ann_file": "openimages/opencrowd_train.json"
        },
        "opencrowd_val": {
            "img_dir": "openimages/val",
            "ann_file": "openimages/opencrowd_val.json"
        },
        "opencrowd_test": {
            "img_dir": "openimages/test",
            "ann_file": "openimages/opencrowd_test.json"
        },
        "opencar_train": {
            "img_dir": "openimages/train",
            "ann_file": "openimages/opencar_train.json"
        },
        "opencar_val": {
            "img_dir": "openimages/val",
            "ann_file": "openimages/opencar_val.json"
        },
        "opencar_test": {
            "img_dir": "openimages/test",
            "ann_file": "openimages/opencar_test.json"
        },
        "openhumancar_train": {
            "img_dir": "openimages/train",
            "ann_file": "openimages/openhumancar_train.json"
        },
        "openhumancar_val": {
            "img_dir": "openimages/val",
            "ann_file": "openimages/openhumancar_val.json"
        },
        "openhuamncar_test": {
            "img_dir": "openimages/test",
            "ann_file": "openimages/openhumancar_test.json"
        },
        "open500_train": {
            "img_dir": "openimages/train",
            "ann_file": "openimages/openimages_challenge_2019_train_bbox.json",
        },
        "open500_val": {
            "img_dir": "openimages/val",
            "ann_file": "openimages/openimages_challenge_2019_val_bbox.json",
        },
        "openproposal_test": {
            "img_dir": "openimages/test2019",
            "ann_file": "openimages/proposals_test.json",
        },
        "object365_train": {
            "img_dir": "object365/train",
            "ann_file": "object365/objects365_train.json"
        },
        "object365_val": {
            "img_dir": "object365/val",
            "ann_file": "object365/objects365_val.json"
        },
        "lvis_train": {
            "img_dir": "coco",
            "ann_file": "coco/annotations/lvis_od_train.json"
        },
        "lvis_val": {
            "img_dir": "coco",
            "ann_file": "coco/annotations/lvis_od_val.json"
        },
        "image200_train": {
            "img_dir": "imagenet-od/Data/DET/train",
            "ann_file": "imagenet-od/im200_train.json"
        },
        "image200_val": {
            "img_dir": "imagenet-od/Data/DET/val",
            "ann_file": "imagenet-od/im200_val.json"
        },
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2017_test": {
            "img_dir": "coco/test2017",
            "ann_file": "coco/annotations/image_info_test-dev2017.json"
        },
        "coco10_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_minitrain2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "coco_2014_train_partial": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/partial0.2_train2014.json"
        },
        "coco_2014_valminusminival_partial": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/partial0.2_valminusminival2014.json"
        },
        "coco_2014_train_few100": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/few100_train2014.json"
        },
        "coco_2014_train_few300": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/few300_train2014.json"
        },
        "coco_human_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/humans_train2014.json"
        },
        "coco_human_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/humans_minival2014.json"
        },
        "coco_human_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/humans_valminusminival2014.json"
        },
        "coco_car_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/car_train2014.json"
        },
        "coco_car_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/car_minival2014.json"
        },
        "coco_car_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/car_valminusminival2014.json"
        },
        "coco_humancar_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/humancar_train2014.json"
        },
        "coco_humancar_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/humancar_minival2014.json"
        },
        "coco_humancar_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/humancar_valminusminival2014.json"
        },
        "coco_keypoint_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/person_keypoints_train2017.json"
        },
        "coco_keypoint_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/person_keypoints_val2017.json"
        },
        "coco_headshoulder_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/headshoulder_train2017.json"
        },
        "coco_headshoulder_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/headshoulder_val2017.json"
        },
        "coco_hskeypoint_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/person_hskeypoints_train2017.json"
        },
        "coco_hskeypoint_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/person_hskeypoints_val2017.json"
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        "crowdhuman_train": {
            "img_dir": "CrowdHuman/Images",
            "ann_file": "CrowdHuman/crowdhuman_train.json"
        },
        "crowdhuman_val": {
            "img_dir": "CrowdHuman/Images",
            "ann_file": "CrowdHuman/crowdhuman_val.json"
        },
        "crowdhead_train": {
            "img_dir": "CrowdHuman/Images",
            "ann_file": "CrowdHuman/crowdhead_train.json"
        },
        "crowdhead_val": {
            "img_dir": "CrowdHuman/Images",
            "ann_file": "CrowdHuman/crowdhead_val.json"
        },
        "crowdfull_train": {
            "img_dir": "CrowdHuman/Images",
            "ann_file": "CrowdHuman/crowdfull_train.json"
        },
        "crowdfull_val": {
            "img_dir": "CrowdHuman/Images",
            "ann_file": "CrowdHuman/crowdfull_val.json"
        },
        "ternium_train": {
            "img_dir": "ternium/images",
            "ann_file": "ternium/train_annotation.json"
        },
        "ternium_val": {
            "img_dir": "ternium/images",
            "ann_file": "ternium/val_annotation.json"
        },
        "ternium_test": {
            "img_dir": "ternium/images",
            "ann_file": "ternium/test_annotation.json"
        },
        "ternium_test_crop": {
            "img_dir": "ternium/test_motion_crop",
            "ann_file": "ternium/test_motion_crop.json"
        },
        "ternium_train_aug": {
            "img_dir": "ternium/train_crop_aug",
            "ann_file": "ternium/train_crop_aug.json"
        },
        "ternium_test_aug": {
            "img_dir": "ternium/test_crop_aug",
            "ann_file": "ternium/test_motion_crop_aug.json"
        },
        "ternium_vh_train": {
            "img_dir": "ternium-vehicle/train_dataset_coco/images",
            "ann_file": "ternium-vehicle/train_dataset_coco/coco_annotation.json"
        },
        "ternium_vh_val": {
            "img_dir": "ternium-vehicle/validation_dataset_coco/images",
            "ann_file": "ternium-vehicle/validation_dataset_coco/coco_annotation.json"
        },
        "msra_traffic": {
            "img_dir": "msra-traffic/Images",
            "ann_file": "msra-traffic/annotation.json"
        },
        "msra_traffic_car": {
            "img_dir": "msra-traffic/Images",
            "ann_file": "msra-traffic/car_annotation.json"
        },
        "msra_traffic_humancar": {
            "img_dir": "msra-traffic/Images",
            "ann_file": "msra-traffic/humancar_annotation.json"
        },
        "jigsaw_car_train": {
            "img_dir": "jigsaw",
            "ann_file": "jigsaw/train.json"
        },
        "jigsaw_car_val": {
            "img_dir": "jigsaw",
            "ann_file": "jigsaw/val.json"
        },
        "miotcd_train": {
            "img_dir": "MIO-TCD/MIO-TCD-Localization",
            "ann_file": "MIO-TCD/train.json"
        },
        "miotcd_val": {
            "img_dir": "MIO-TCD/MIO-TCD-Localization",
            "ann_file": "MIO-TCD/val.json"
        },
        "detrac_train": {
            "img_dir": "detrac/Insight-MVT_Annotation_Train",
            "ann_file": "detrac/train.json"
        },
        "detrac_val": {
            "img_dir": "detrac/Insight-MVT_Annotation_Train",
            "ann_file": "detrac/val.json"
        },
        "mrw": {
            "img_dir": "mrw/clips",
            "ann_file": "mrw/annotations.json"
        },
        "mrw_bg": {
            "img_dir": "mrw/bg",
            "ann_file": "mrw/bg_annotations.json"
        },
        "webmarket_bg": {
            "img_dir": "webmarket",
            "ann_file": "webmarket/bg_annotations.json"
        },
        "mot17_train": {
            "img_dir": "mot/MOT17Det",
            "ann_file": "mot/MOT17Det/train.json"
        },
        "egohands": {
            "img_dir": "egohands/images",
            "ann_file": "egohands/egohands.json"
        },
        "hof": {
            "img_dir": "hof/images_original_size",
            "ann_file": "hof/train.json"
        },
        "vlmhof": {
            "img_dir": "vlmhof/RGB",
            "ann_file": "vlmhof/train.json"
        },
        "vgghands_train": {
            "img_dir": "vgghands/training_dataset",
            "ann_file": "vgghands/training.json"
        },
        "vgghands_val": {
            "img_dir": "vgghands/validation_dataset",
            "ann_file": "vgghands/validation.json"
        },
        "vgghands_test": {
            "img_dir": "vgghands/test_dataset",
            "ann_file": "vgghands/test.json"
        },
        "od:coco_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/od_train2017.json"
        },
        "od:coco_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/od_val2017.json"
        },
        "od:lvis_train": {
            "img_dir": "coco",
            "ann_file": "coco/annotations/od_train-lvis.json"
        },
        "od:lvis_val": {
            "img_dir": "coco",
            "ann_file": "coco/annotations/od_val-lvis.json"
        },
        "od:o365_train": {
            "img_dir": "object365/train",
            "ann_file": "object365/od_train.json"
        },
        "od:o365_val": {
            "img_dir": "object365/val",
            "ann_file": "object365/od_val.json"
        },
        "od:oi500_train": {
            "img_dir": "openimages/train",
            "ann_file": "openimages/od_train2019.json",
            "paste_dir": "openimages/panoptic_train_challenge_2019",
            "paste_file": "openimages/panoptic_train2019.json",
        },
        "od:oi500_val": {
            "img_dir": "openimages/val",
            "ann_file": "openimages/od_val2019.json",
            "paste_dir": "openimages/panoptic_val_challenge_2019",
            "paste_file": "openimages/panoptic_val2019.json",
        },
        "od:im200_train": {
            "img_dir": "imagenet-od/Data/DET/train",
            "ann_file": "imagenet-od/train.json"
        },
        "od:im200_val": {
            "img_dir": "imagenet-od/Data/DET/val",
            "ann_file": "imagenet-od/val.json"
        },
        "cv:animal661_train": {
            "img_dir": "cvtasks/animal-661/images",
            "ann_file": "cvtasks/animal-661/train.json"
        },
        "cv:animal661_test": {
            "img_dir": "cvtasks/animal-661/images",
            "ann_file": "cvtasks/animal-661/test.json"
        },
        "cv:seeingai_train": {
            "img_dir": "cvtasks/SeeingAI/train.tsv",
            "ann_file": "cvtasks/SeeingAI/train.json"
        },
        "cv:seeingai_test": {
            "img_dir": "cvtasks/SeeingAI/test.tsv",
            "ann_file": "cvtasks/SeeingAI/test.json"
        },
        "cv:office_train": {
            "img_dir": "cvtasks/Ping-Office-Env/train.tsv",
            "ann_file": "cvtasks/Ping-Office-Env/train.json"
        },
        "cv:office_test": {
            "img_dir": "cvtasks/Ping-Office-Env/test.tsv",
            "ann_file": "cvtasks/Ping-Office-Env/test.json"
        },
        "cv:logo_train": {
            "img_dir": "cvtasks/Ping-Logo",
            "ann_file": "cvtasks/Ping-Logo/train.json"
        },
        "cv:logo_test": {
            "img_dir": "cvtasks/Ping-Logo",
            "ann_file": "cvtasks/Ping-Logo/test.json"
        },
        "cv:nba_train": {
            "img_dir": "cvtasks/Ping-NBA",
            "ann_file": "cvtasks/Ping-NBA/train.json"
        },
        "cv:nba_test": {
            "img_dir": "cvtasks/Ping-NBA",
            "ann_file": "cvtasks/Ping-NBA/test.json"
        },
        "cv:traffic_train": {
            "img_dir": "cvtasks/TrafficData/train.tsv",
            "ann_file": "cvtasks/TrafficData/train.json"
        },
        "cv:traffic_test": {
            "img_dir": "cvtasks/TrafficData/test.tsv",
            "ann_file": "cvtasks/TrafficData/test.json"
        },
        "cv:fashion5k_train": {
            "img_dir": "cvtasks/fashion5k",
            "ann_file": "cvtasks/fashion5k/train.json"
        },
        "cv:fashion5k_test": {
            "img_dir": "cvtasks/fashion5k",
            "ann_file": "cvtasks/fashion5k/test.json"
        },
        "cv:malaria_train": {
            "img_dir": "cvtasks/malaria",
            "ann_file": "cvtasks/malaria/train.json"
        },
        "cv:malaria_test": {
            "img_dir": "cvtasks/malaria",
            "ann_file": "cvtasks/malaria/test.json"
        },
        "cv:product_train": {
            "img_dir": "cvtasks/product_detection",
            "ann_file": "cvtasks/product_detection/train.json"
        },
        "cv:product_test": {
            "img_dir": "cvtasks/product_detection",
            "ann_file": "cvtasks/product_detection/test.json"
        },
        "vl:vg_train": {
            "yaml_file": "vlp/visualgenome/train_vgoi6_clipped.yaml"
        },
        "vl:vg_test": {
            "yaml_file": "vlp/visualgenome/test_vgoi6_clipped.yaml"
        },
        "imagenet_train": {
            "img_dir": "imagenet-tsv/train.tsv",
            "ann_file": None
        },
        "imagenet_val": {
            "img_dir": "imagenet-tsv/val.tsv",
            "ann_file": None
        }
    }

    @staticmethod
    def set(name, info):
        DatasetCatalog.DATASETS.update({name: info})

    @staticmethod
    def get(name):

        if name.endswith('_bg'):
            attrs = DatasetCatalog.DATASETS[name]
            data_dir = try_to_find(attrs["ann_file"], return_dir=True)
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="Background",
                args=args,
            )
        else:
            if "bing" in name.split("_"):
                attrs = DatasetCatalog.DATASETS["bing_caption_train"]
            else:
                attrs = DatasetCatalog.DATASETS[name]
            # if "yaml_file" in attrs:
            #     yaml_file = try_to_find(attrs["yaml_file"], return_dir=False)
            #     args = dict(yaml_file=yaml_file)
            #     return dict(
            #         factory="VGTSVDataset",
            #         args=args,
            #     )
            # elif attrs["img_dir"].endswith('tsv'):
            #     try:
            #         data_dir = try_to_find(attrs["img_dir"], return_dir=True)
            #         if attrs["ann_file"] is None:
            #             map_file = None
            #         elif attrs["ann_file"].startswith("./"):
            #             map_file = attrs["ann_file"]
            #         else:
            #             map_file = os.path.join(data_dir, attrs["ann_file"])
            #     except:
            #         return None
            #     args = dict(
            #         tsv_file=os.path.join(data_dir, attrs["img_dir"]),
            #         anno_file=map_file,
            #     )
            #     return dict(
            #         factory="TSVDataset",
            #         args=args,
            #     )
            if "voc" in name and 'split' in attrs:
                data_dir = try_to_find(attrs["data_dir"], return_dir=True)
                args = dict(
                    data_dir=os.path.join(data_dir, attrs["data_dir"]),
                    split=attrs["split"],
                )
                return dict(
                    factory="PascalVOCDataset",
                    args=args,
                )
            elif "mixed" in name:
                vg_img_dir = try_to_find(attrs["vg_img_dir"], return_dir=True)
                coco_img_dir = try_to_find(attrs["coco_img_dir"], return_dir=True)
                ann_file = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder_coco=os.path.join(coco_img_dir, attrs["coco_img_dir"]),
                    img_folder_vg=os.path.join(vg_img_dir, attrs["vg_img_dir"]),
                    ann_file=os.path.join(ann_file, attrs["ann_file"])
                )
                return dict(
                    factory="MixedDataset",
                    args=args,
                )
            elif "flickr" in name:
                img_dir = try_to_find(attrs["img_folder"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_folder"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                    is_train=attrs["is_train"]
                )
                return dict(
                    factory="FlickrDataset",
                    args=args,
                )
            elif "refexp" in name or "refcoco" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="RefExpDataset",
                    args=args,
                )
            elif "gqa" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="GQADataset",
                    args=args,
                )
            elif "phrasecut" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="PhrasecutDetection",
                    args=args,
                )
            elif "_caption" in name:
                yaml_path = try_to_find(attrs["yaml_path"], return_dir=True)
                if "no_coco" in name:
                    yaml_name = attrs["yaml_name_no_coco"]
                else:
                    yaml_name = attrs["yaml_name"]
                yaml_file_name = "{}.{}.yaml".format(yaml_name, name.split("_")[2])
                args = dict(
                    yaml_file=os.path.join(yaml_path, attrs["yaml_path"], yaml_file_name)
                )
                return dict(
                    factory="CaptionTSV",
                    args=args,
                )
            elif "inferencecap" in name:
                yaml_file_name = try_to_find(attrs["yaml_path"])
                args = dict(
                    yaml_file=yaml_file_name)
                return dict(
                    factory="CaptionTSV",
                    args=args,
                )
            elif "pseudo_data" in name:
                args = dict(
                    yaml_file=try_to_find(attrs["yaml_path"])
                )
                return dict(
                    factory="PseudoData",
                    args=args,
                )
            elif "_dt" in name:
                dataset_file = attrs["dataset_file"]
                yaml_path = try_to_find(attrs["yaml_path"], return_dir=True)
                args = dict(
                    name=dataset_file,
                    yaml_file=os.path.join(yaml_path, attrs["yaml_path"]),
                )
                return dict(
                    factory="CocoDetectionTSV",
                    args=args,
                )
            elif "_odtsv" in name:
                dataset_file = attrs["dataset_file"]
                yaml_path = try_to_find(attrs["yaml_path"], return_dir=True)
                args = dict(
                    name=dataset_file,
                    yaml_file=os.path.join(yaml_path, attrs["yaml_path"]),
                )
                return dict(
                    factory="ODTSVDataset",
                    args=args,
                )
            elif "_grounding" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="CocoGrounding",
                    args=args,
                )
            elif "lvis_evaluation" in name:
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                args = dict(
                    img_folder=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                return dict(
                    factory="LvisDetection",
                    args=args,
                )
            else:
                ann_dir = try_to_find(attrs["ann_file"], return_dir=True)
                img_dir = try_to_find(attrs["img_dir"], return_dir=True)
                args = dict(
                    root=os.path.join(img_dir, attrs["img_dir"]),
                    ann_file=os.path.join(ann_dir, attrs["ann_file"]),
                )
                for k, v in attrs.items():
                    args.update({k: os.path.join(ann_dir, v)})
                return dict(
                    factory="COCODataset",
                    args=args,
                )

        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/20171220/X-101-64x4d": "ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
