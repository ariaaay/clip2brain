import numpy as np

taskrepr_features = [
    "autoencoder",
    "denoise",
    "colorization",
    "curvature",
    "edge2d",
    "edge3d",
    "keypoint2d",
    "keypoint3d",
    "segment2d",
    "segment25d",
    "segmentsemantic",
    "rgb2depth",
    "rgb2mist",
    "reshade",
    "rgb2sfnorm",
    "room_layout",
    "vanishing_point",
    "class_1000",
    "class_places",
    # "jigsaw",
    "inpainting_whole",
]

convnet_structures = ["res50", "vgg16"]

model_features = dict()
model_features["taskrepr"] = taskrepr_features
model_features["convnet"] = convnet_structures

model_label = {
    "YFCC_simclr": "YFCC simCLR",
    "YFCC_clip": "YFCC CLIP",
    "YFCC_slip": "YFCC SLIP",
    "laion400m_clip": "LAION400m CLIP",
    "laion2b_clip": "LAION2b CLIP",
    "clip": "OpenAI CLIP",
}

task_label = {
    "class_1000": "Object Class",
    "segment25d": "2.5D Segm.",
    "room_layout": "Layout",
    "rgb2sfnorm": "Normals",
    "rgb2depth": "Depth",
    "rgb2mist": "Distance",
    "reshade": "Reshading",
    "keypoint3d": "3D Keypoint",
    "keypoint2d": "2D Keypoint",
    "autoencoder": "Autoencoding",
    "colorization": "Color",
    "edge3d": "Occlusion Edges",
    "edge2d": "2D Edges",
    "denoise": "Denoising",
    "curvature": "Curvature",
    "class_places": "Scene Class",
    "vanishing_point": "Vanishing Pts.",
    "segmentsemantic": "Semantic Segm.",
    "segment2d": "2D Segm.",
    # "jigsaw": "Jigsaw",
    "inpainting_whole": "Inpainting",
}

task_label_NSD_tmp = {
    "class_1000": "Object Class",
    "segment25d": "2.5D Segm.",
    "room_layout": "Layout",
    # "rgb2sfnorm": "Normals",
    # "rgb2depth": "Depth",
    # "rgb2mist": "Distance",
    # "reshade": "Reshading",
    "keypoint3d": "3D Keypoint",
    "keypoint2d": "2D Keypoint",
    # "auto[uencoder": "Autoencoding",
    # "colorization": "Color",
    "edge3d": "Occlusion Edges",
    "edge2d": "2D Edges",
    # "denoise": "Denoising",
    # "curvature": "Curvature",
    "class_places": "Scene Class",
    "vanishing_point": "Vanishing Pts.",
    "segmentsemantic": "Semantic Segm.",
    # "segment2d": "2D Segm.",
    # "jigsaw": "Jigsaw",
    "inpainting_whole": "Inpainting",
}


task_label_in_Taskonomy19_matrix_order = {
    "class_1000": "Object Class",
    "segment2d": "2D Segm.",
    "segmentsemantic": "Semantic Segm.",
    "class_places": "Scene Class",
    "denoise": "Denoising",
    "edge2d": "2D Edges",
    "edge3d": "Occlusion Edges",
    # "jigsaw": "Jigsaw",
    "autoencoder": "Autoencoding",
    "colorization": "Color",
    "keypoint3d": "3D Keypoint",
    "reshade": "Reshading",
    "rgb2mist": "Distance",
    "rgb2depth": "Depth",
    "rgb2sfnorm": "Normals",
    "room_layout": "Layout",
    "segment25d": "2.5D Segm.",
    "keypoint2d": "2D Keypoint",
    "inpainting_whole": "Inpainting",
    "curvature": "Curvature",
    "vanishing_point": "Vanishing Pts.",
}

conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
fc_layers = ["fc6", "fc7"]

ecc_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "0.5 deg",
    2: "1 deg",
    3: "2 deg",
    4: "4 deg",
    5: ">4 deg",
}

visual_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "V1v",
    2: "V1d",
    3: "V2v",
    4: "V2d",
    5: "V3v",
    6: "v3d",
    7: "h4v",
}

place_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "OPA",
    2: "PPA",
    3: "RSC",
}

face_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "OFA",
    2: "FFA-1",
    3: "FFA-2",
    4: "mTL-faces",
    5: "aTL-faces",
}

word_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "OWFA",
    2: "VWFA-1",
    3: "VWFA-2",
    4: "mfs-words",
}


body_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "EBA",
    2: "FBA-1",
    3: "FBA-2",
    4: "mTL-bodies",
}

word_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "OWFA",
    2: "VWFA-1",
    3: "VWFA-2",
    4: "mfs-words",
    5: "mTL-words",
}

language_roi_names = {
    0: "Unknown",
    1: "PTL",
    2: "ATL",
    3: "AG",
    4: "IFG",
    5: "MFG",
    6: "IFGorb",
}

kastner_roi_names = {
    0: "Unknown",
    1: "V1v",
    2: "V1d",
    3: "V2v",
    4: "V2d",
    5: "V3v",
    6: "V3d",
    7: "hV4",
    8: "VO1",
    9: "VO2",
    10: "PHC1",
    11: "PHC2",
    12: "TO2",
    13: "TO1",
    14: "LO2",
    15: "LO1",
    16: "V3B",
    17: "V3A",
    18: "IPS0",
    19: "IPS1",
    20: "IPS2",
    21: "IPS3",
    22: "IPS4",
    23: "IPS5",
    24: "SPL1",
    25: "FEF",
}

hcp_roi_names = {
    "0": "Unknown",
    "1": "V1",
    "2": "MST",
    "3": "V6",
    "4": "V2",
    "5": "V3",
    "6": "V4",
    "7": "V8",
    "8": "4",
    "9": "3b",
    "10": "FEF",
    "11": "PEF",
    "12": "55b",
    "13": "V3A",
    "14": "RSC",
    "15": "POS2",
    "16": "V7",
    "17": "IPS1",
    "18": "FFC",
    "19": "V3B",
    "20": "LO1",
    "21": "LO2",
    "22": "PIT",
    "23": "MT",
    "24": "A1",
    "25": "PSL",
    "26": "SFL",
    "27": "PCV",
    "28": "STV",
    "29": "7Pm",
    "30": "7m",
    "31": "POS1",
    "32": "23d",
    "33": "v23ab",
    "34": "d23ab",
    "35": "31pv",
    "36": "5m",
    "37": "5mv",
    "38": "23c",
    "39": "5L",
    "40": "24dd",
    "41": "24dv",
    "42": "7AL",
    "43": "SCEF",
    "44": "6ma",
    "45": "7Am",
    "46": "7PL",
    "47": "7PC",
    "48": "LIPv",
    "49": "VIP",
    "50": "MIP",
    "51": "1",
    "52": "2",
    "53": "3a",
    "54": "6d",
    "55": "6mp",
    "56": "6v",
    "57": "p24pr",
    "58": "33pr",
    "59": "a24pr",
    "60": "p32pr",
    "61": "a24",
    "62": "d32",
    "63": "8BM",
    "64": "p32",
    "65": "10r",
    "66": "47m",
    "67": "8Av",
    "68": "8Ad",
    "69": "9m",
    "70": "8BL",
    "71": "9p",
    "72": "10d",
    "73": "8C",
    "74": "44",
    "75": "45",
    "76": "47l",
    "77": "a47r",
    "78": "6r",
    "79": "IFJa",
    "80": "IFJp",
    "81": "IFSp",
    "82": "IFSa",
    "83": "p9-46v",
    "84": "46",
    "85": "a9-46v",
    "86": "9-46d",
    "87": "9a",
    "88": "10v",
    "89": "a10p",
    "90": "10pp",
    "91": "11l",
    "92": "13l",
    "93": "OFC",
    "94": "47s",
    "95": "LIPd",
    "96": "6a",
    "97": "i6-8",
    "98": "s6-8",
    "99": "43",
    "100": "OP4",
    "101": "OP1",
    "102": "OP2-3",
    "103": "52",
    "104": "RI",
    "105": "PFcm",
    "106": "PoI2",
    "107": "TA2",
    "108": "FOP4",
    "109": "MI",
    "110": "Pir",
    "111": "AVI",
    "112": "AAIC",
    "113": "FOP1",
    "114": "FOP3",
    "115": "FOP2",
    "116": "PFt",
    "117": "AIP",
    "118": "EC",
    "119": "PreS",
    "120": "H",
    "121": "ProS",
    "122": "PeEc",
    "123": "STGa",
    "124": "PBelt",
    "125": "A5",
    "126": "PHA1",
    "127": "PHA3",
    "128": "STSda",
    "129": "STSdp",
    "130": "STSvp",
    "131": "TGd",
    "132": "TE1a",
    "133": "TE1p",
    "134": "TE2a",
    "135": "TF",
    "136": "TE2p",
    "137": "PHT",
    "138": "PH",
    "139": "TPOJ1",
    "140": "TPOJ2",
    "141": "TPOJ3",
    "142": "DVT",
    "143": "PGp",
    "144": "IP2",
    "145": "IP1",
    "146": "IP0",
    "147": "PFop",
    "148": "PF",
    "149": "PFm",
    "150": "PGi",
    "151": "PGs",
    "152": "V6A",
    "153": "VMV1",
    "154": "VMV3",
    "155": "PHA2",
    "156": "V4t",
    "157": "FST",
    "158": "V3CD",
    "159": "LO3",
    "160": "VMV2",
    "161": "31pd",
    "162": "31a",
    "163": "VVC",
    "164": "25",
    "165": "s32",
    "166": "pOFC",
    "167": "PoI1",
    "168": "Ig",
    "169": "FOP5",
    "170": "p10p",
    "171": "p47r",
    "172": "TGv",
    "173": "MBelt",
    "174": "LBelt",
    "175": "A4",
    "176": "STSva",
    "177": "TE1m",
    "178": "PI",
    "179": "a32pr",
    "180": "p24",
}

roi_name_dict = {
    "floc-words": word_roi_names,
    "floc-faces": face_roi_names,
    "floc-places": place_roi_names,
    "floc-bodies": body_roi_names,
    "prf-visualrois": visual_roi_names,
    "prf-eccrois": ecc_roi_names,
    "Kastner2015": kastner_roi_names,
    "HCP_MMP1": hcp_roi_names,
    "language": language_roi_names,
}

COCO_cat = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_super_cat = [
    "person",
    "vehicle",
    "outdoor",
    "animal",
    "accessory",
    "sports",
    "kitchen",
    "food",
    "furtniture",
    "electronics",
    "appliance",
    "indoor",
]

COCO_cat = np.array(COCO_cat)
COCO_super_cat = np.array(COCO_super_cat)
