import skimage.io as io
from pycocotools.coco import COCO

trainFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
valFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"
train_caps = COCO(trainFile)
val_caps = COCO(valFile)


def load_captions(cid):
    annIds = train_caps.getAnnIds(imgIds=[cid])
    anns = train_caps.loadAnns(annIds)
    if anns == []:
        annIds = val_caps.getAnnIds(imgIds=[cid])
        anns = val_caps.loadAnns(annIds)

    if anns == []:
        print("no captions extracted for image: " + str(cid))

    captions = [d["caption"] for d in anns]
    return captions


def get_coco_image(id, coco_train, coco_val):
    try:
        img = coco_train.loadImgs([id])[0]
    except KeyError:
        img = coco_val.loadImgs([id])[0]
    I = io.imread(img["coco_url"])
    return I


def get_coco_anns(id, coco_train, coco_val):
    try:
        annIds = coco_train.getAnnIds([id])
        anns = coco_train.loadAnns(annIds)
    except KeyError:
        annIds = coco_val.getAnnIds([id])
        anns = coco_val.loadAnns(annIds)

    cats = [ann["category_id"] for ann in anns]
    return cats


def get_coco_caps(id, coco_train_caps, coco_val_caps):
    try:
        annIds = coco_train_caps.getAnnIds([id])
        anns = coco_train_caps.loadAnns(annIds)
    except KeyError:
        annIds = coco_val_caps.getAnnIds([id])
        anns = coco_val_caps.loadAnns(annIds)
    # import pdb; pdb.set_trace()
    caps = [ann["caption"] for ann in anns]
    return caps
