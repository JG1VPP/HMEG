train = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type="CROHME",
        npy_path="datasets/crohme2019/link_npy",
        img_path="datasets/crohme2019/Train_imgs",
        pipeline=[
            dict(type="LoadImageFromFile", to_float32=True),
            dict(
                type="Normalize", mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]
            ),
            dict(type="Resize", scale=(256, 256), keep_ratio=True),
            dict(type="Pad", size=(256, 256), pad_val=1),
            dict(type="ImageToTensor", keys=["img"]),
        ],
        test_mode=False,
    ),
)


# parser.add_argument("--dataset", default="crohme", choices=["vg", "coco", "crohme"])
#
## Optimization hyperparameters
# parser.add_argument("--batch_size", default=8, type=int)
# parser.add_argument("--num_epochs", default=1000, type=int)
# parser.add_argument("--learning_rate", default=1e-4, type=float)
#
## Switch the generator to eval mode after this many iterations
# parser.add_argument("--eval_mode_after", default=100000, type=int)
#
## Dataset options common to both VG and COCO
# parser.add_argument("--image_size", default="256,256", type=int_tuple)
# parser.add_argument("--num_train_samples", default=None, type=int)
# parser.add_argument("--num_val_samples", default=1024, type=int)
# parser.add_argument("--shuffle_val", default=True, type=bool_flag)
# parser.add_argument("--loader_num_workers", default=4, type=int)
# parser.add_argument("--include_relationships", default=True, type=bool_flag)
#
## Generator options
# parser.add_argument(
#    "--mask_size", default=16, type=int
# )  # Set this to 0 to use no masks
# parser.add_argument("--embedding_dim", default=128, type=int)
# parser.add_argument("--gconv_dim", default=128, type=int)
# parser.add_argument("--gconv_hidden_dim", default=512, type=int)
# parser.add_argument("--gconv_num_layers", default=5, type=int)
# parser.add_argument("--mlp_normalization", default="none", type=str)
# parser.add_argument(
#    "--refinement_network_dims", default="1024,512,256,128,64", type=int_tuple
# )
# parser.add_argument("--normalization", default="batch")
# parser.add_argument("--activation", default="leakyrelu-0.2")
# parser.add_argument("--layout_noise_dim", default=32, type=int)
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
# parser.add_argument("--d_clip", default=None, type=float)
# parser.add_argument("--d_normalization", default="batch")
# parser.add_argument("--d_padding", default="valid")
# parser.add_argument("--d_activation", default="leakyrelu-0.2")
#
## Object discriminator
# parser.add_argument("--d_obj_arch", default="C4-64-2,C4-128-2,C4-256-2")
# parser.add_argument("--crop_size", default=32, type=int)
# parser.add_argument(
#    "--d_obj_weight", default=1.0, type=float
# )  # multiplied by d_loss_weight
# parser.add_argument("--ac_loss_weight", default=0.1, type=float)
#
## Image discriminator
# parser.add_argument("--d_img_arch", default="C4-64-2,C4-128-2,C4-256-2")
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
