vocab = dict(
    object_name_to_idx={
        "!": 0,
        "(": 1,
        ")": 2,
        "+": 3,
        ",": 4,
        "-": 5,
        ".": 6,
        "/": 7,
        "0": 8,
        "1": 9,
        "2": 10,
        "3": 11,
        "4": 12,
        "5": 13,
        "6": 14,
        "7": 15,
        "8": 16,
        "9": 17,
        "=": 18,
        "A": 19,
        "B": 20,
        "C": 21,
        "E": 22,
        "F": 23,
        "G": 24,
        "H": 25,
        "I": 26,
        "L": 27,
        "M": 28,
        "N": 29,
        "P": 30,
        "R": 31,
        "S": 32,
        "T": 33,
        "V": 34,
        "X": 35,
        "Y": 36,
        "[": 37,
        "\\Delta": 38,
        "\\alpha": 39,
        "\\beta": 40,
        "\\cos": 41,
        "\\div": 42,
        "\\exists": 43,
        "\\forall": 44,
        "\\gamma": 45,
        "\\geq": 46,
        "\\gt": 47,
        "\\in": 48,
        "\\infty": 49,
        "\\int": 50,
        "\\lambda": 51,
        "\\ldots": 52,
        "\\leq": 53,
        "\\lim": 54,
        "\\log": 55,
        "\\lt": 56,
        "\\mu": 57,
        "\\neq": 58,
        "\\phi": 59,
        "\\pi": 60,
        "\\pm": 61,
        "\\prime": 62,
        "\\rightarrow": 63,
        "\\sigma": 64,
        "\\sin": 65,
        "\\sqrt": 66,
        "\\sum": 67,
        "\\tan": 68,
        "\\theta": 69,
        "\\times": 70,
        "\\{": 71,
        "\\}": 72,
        "]": 73,
        "a": 74,
        "b": 75,
        "c": 76,
        "d": 77,
        "e": 78,
        "f": 79,
        "g": 80,
        "h": 81,
        "i": 82,
        "j": 83,
        "k": 84,
        "l": 85,
        "m": 86,
        "n": 87,
        "o": 88,
        "p": 89,
        "q": 90,
        "r": 91,
        "s": 92,
        "t": 93,
        "u": 94,
        "v": 95,
        "w": 96,
        "x": 97,
        "y": 98,
        "z": 99,
        "|": 100,
    },
    object_idx_to_name=[
        "!",
        "(",
        ")",
        "+",
        ",",
        "-",
        ".",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "=",
        "A",
        "B",
        "C",
        "E",
        "F",
        "G",
        "H",
        "I",
        "L",
        "M",
        "N",
        "P",
        "R",
        "S",
        "T",
        "V",
        "X",
        "Y",
        "[",
        "\\Delta",
        "\\alpha",
        "\\beta",
        "\\cos",
        "\\div",
        "\\exists",
        "\\forall",
        "\\gamma",
        "\\geq",
        "\\gt",
        "\\in",
        "\\infty",
        "\\int",
        "\\lambda",
        "\\ldots",
        "\\leq",
        "\\lim",
        "\\log",
        "\\lt",
        "\\mu",
        "\\neq",
        "\\phi",
        "\\pi",
        "\\pm",
        "\\prime",
        "\\rightarrow",
        "\\sigma",
        "\\sin",
        "\\sqrt",
        "\\sum",
        "\\tan",
        "\\theta",
        "\\times",
        "\\{",
        "\\}",
        "]",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "|",
    ],
    pred_name_to_idx={
        "__None__": 0,
        "R": 1,
        "L": 2,
        "Sub": 3,
        "Sup": 4,
        "Above": 5,
        "Below": 6,
        "Inside": 7,
        "Outside": 8,
    },
    pred_idx_to_name=[
        "__None__",
        "R",
        "L",
        "Sub",
        "Sup",
        "Above",
        "Below",
        "Inside",
        "Outside",
    ],
)

model = dict(
    type="GAN",
    gen=dict(
        type="Sg2ImModel",
        vocab=vocab,
        image_size=(256, 256),
        embedding_dim=128,
        gconv_dim=128,
        gconv_hidden_dim=512,
        gconv_num_layers=5,
        mlp_normalization="none",
        refinement_dims=(1024, 512, 256, 128, 64),
        normalization="batch",
        activation="leakyrelu-0.2",
        mask_size=16,
        layout_noise_dim=32,
    ),
    img=dict(
        type="PatchDiscriminator",
        arch="C4-64-2,C4-128-2,C4-256-2",
        normalization="batch",
        activation="leakyrelu-0.2",
        padding="valid",
    ),
    obj=dict(
        type="AcCropDiscriminator",
        vocab=vocab,
        arch="C4-64-2,C4-128-2,C4-256-2",
        normalization="batch",
        activation="leakyrelu-0.2",
        padding="valid",
        object_size=32,
    ),
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type="CROHME",
        npy_path="datasets/crohme2019/link_npy",
        img_path="datasets/crohme2019/Train_imgs",
        pipeline=[
            dict(type="LoadImageFromFile", to_float32=True),
            dict(
                type="Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
            dict(type="Resize", scale=(256, 256), keep_ratio=True),
            dict(type="Pad", size=(256, 256), pad_val=1),
            dict(type="ImageToTensor", keys=["img"]),
        ],
        test_mode=False,
    ),
)

train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=1)

optim_wrapper = dict(optimizer=dict(type="Adam", lr=1e-4))

work_dir = "results"

# deprecate! use load_from instead
ckpt = "weight/layout_conv_sum2_new_with_model.pt"

#
#
## Switch the generator to eval mode after this many iterations
# parser.add_argument("--eval_mode_after", default=100000, type=int)
#
## Dataset options common to both VG and COCO
# parser.add_argument("--num_train_samples", default=None, type=int)
# parser.add_argument("--num_val_samples", default=1024, type=int)
# parser.add_argument("--shuffle_val", default=True, type=bool_flag)
# parser.add_argument("--loader_num_workers", default=4, type=int)
# parser.add_argument("--include_relationships", default=True, type=bool_flag)
#
## Generator options
# parser.add_argument("--use_boxes_pred_after", default=-1, type=int)
#
## Generator losses
# parser.add_argument("--mask_loss_weight", default=0, type=float)
# parser.add_argument("--l1_pixel_loss_weight", default=1.0, type=float)
# parser.add_argument("--bbox_pred_loss_weight", default=10, type=float)
# parser.add_argument("--predicate_pred_loss_weight", default=0, type=float)  # DEPRECATED
#
## Generic discriminator options
# parser.add_argument("--discriminator_loss_weight", default=0.01, type=float)
# parser.add_argument("--gan_loss_type", default="gan")
#
## Object discriminator
# parser.add_argument(
#    "--d_obj_weight", default=1.0, type=float
# )  # multiplied by d_loss_weight
# parser.add_argument("--ac_loss_weight", default=0.1, type=float)
#
## Image discriminator
# parser.add_argument(
#    "--d_img_weight", default=1.0, type=float
# )  # multiplied by d_loss_weight
#
## Output options
# parser.add_argument("--timing", default=False, type=bool_flag)
# parser.add_argument("--output_dir", default=os.getcwd())
# parser.add_argument("--checkpoint_name", default="layout_conv_sum2_new")
# parser.add_argument("--checkpoint_start_from", default=None)
# parser.add_argument("--restore_from_checkpoint", default=False, type=bool_flag)
#
#
# CROHME_DIR = os.path.expanduser("datasets/crohme2019")
