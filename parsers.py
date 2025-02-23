from pprint import pprint
from TIM.utils.constants import *
from argparse import ArgumentParser

class Options:
    parser = ArgumentParser()
    init_args = {}

    def ReadArgs():
        opt = {}

        #region nn TIM

        # ===============================================================
        #                     General options
        # ===============================================================
        Options.parser.add_argument('--data_dir', type=str, default='./data/h3.6m/dataset/', help='path to H36M dataset')
        Options.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        Options.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')

        # ===============================================================
        #                     Model options
        # ===============================================================
        Options.parser.add_argument('--max_norm', dest='max_norm', action='store_true',
                                 help='maxnorm constraint to weights')
        Options.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        Options.parser.add_argument('--num_stage', type=int, default=12, help='# layers in linear model')

        # ===============================================================
        #                     Running options
        # ===============================================================
        Options.parser.add_argument('--lr', type=float, default=5.0e-4)
        Options.parser.add_argument('--lr_autoencoder', type=float, default=5.0e-4)
        Options.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        Options.parser.add_argument('--lr_gamma', type=float, default=0.96)

        Options.parser.add_argument('--input_n', type=int, default=10, help='observed seq length')
        Options.parser.add_argument('--output_n', type=int, default=10, help='future seq length')
        
        Options.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        Options.parser.add_argument('--epochs', type=int, default=50)
        Options.parser.add_argument('--dropout', type=float, default=0.5,
                                 help='dropout probability, 1.0 to make no dropout')
        Options.parser.add_argument('--train_batch', type=int, default=16)
        Options.parser.add_argument('--test_batch', type=int, default=16)
        Options.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        Options.parser.add_argument('--is_load', dest='is_load', action='store_true', help='wether to load existing model')
        Options.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        Options.parser.add_argument('--filename_ext', type=str, default='')

        Options.parser.set_defaults(max_norm=True)
        Options.parser.set_defaults(is_load=False)

        #endregion

        #region nn MMPose

        Options.parser.add_argument('inputs', type=str, nargs='?', help='Input image/video path or folder path.')
        Options.parser.add_argument('--pose2d', type=str, default=None,
            help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
            'config file or the model name defined in metafile.')
        Options.parser.add_argument('--pose2d-weights', type=str, default=None,
            help='Path to the custom checkpoint file of the selected pose model. '
            'If it is not specified and "pose2d" is a model name of metafile, '
            'the weights will be loaded from metafile.')
        Options.parser.add_argument('--pose3d', type=str, default='MMPOSE/configs/body_3d_keypoint/image_pose_lift/h36m/video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m.py',
            help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
            'config file or the model name defined in metafile.')
        Options.parser.add_argument('--pose3d-weights', type=str, default='MMPOSE/configs/body_3d_keypoint/image_pose_lift/h36m/videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth',
            help='Path to the custom checkpoint file of the selected pose model. '
            'If it is not specified and "pose3d" is a model name of metafile, '
            'the weights will be loaded from metafile.')
        Options.parser.add_argument('--det-model', type=str, default=None, help='Config path or alias of detection model.')
        Options.parser.add_argument('--det-weights', type=str, default=None, help='Path to the checkpoints of detection model.')
        Options.parser.add_argument('--det-cat-ids', type=int, nargs='+', default=0, help='Category id for detection model.')
        Options.parser.add_argument('--scope', type=str, default='mmpose', help='Scope where modules are defined.')
        Options.parser.add_argument('--device', type=str, default=None,
            help='Device used for inference. '
            'If not specified, the available device will be automatically used.')
        Options.parser.add_argument('--show', action='store_true', help='Display the image/video in a popup window.')
        Options.parser.add_argument('--draw-bbox', action='store_true', help='Whether to draw the bounding boxes.')
        Options.parser.add_argument('--draw-heatmap', action='store_true', default=False, help='Whether to draw the predicted heatmaps.')
        Options.parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
        Options.parser.add_argument('--nms-thr', type=float, default=0.3, help='IoU threshold for bounding box NMS')
        Options.parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
        Options.parser.add_argument('--tracking-thr', type=float, default=0.3, help='Tracking threshold')
        Options.parser.add_argument('--use-oks-tracking', action='store_true', help='Whether to use OKS as similarity in tracking')
        Options.parser.add_argument('--disable-norm-pose-2d', action='store_true',
            help='Whether to scale the bbox (along with the 2D pose) to the '
            'average bbox scale of the dataset, and move the bbox (along with the '
            '2D pose) to the average bbox center of the dataset. This is useful '
            'when bbox is small, especially in multi-person scenarios.')
        Options.parser.add_argument('--disable-rebase-keypoint', action='store_true', default=False,
            help='Whether to disable rebasing the predicted 3D pose so its '
            'lowest keypoint has a height of 0 (landing on the ground). Rebase '
            'is useful for visualization when the model do not predict the '
            'global position of the 3D pose.')
        Options.parser.add_argument('--num-instances', type=int, default=1,
            help='The number of 3D poses to be visualized in every frame. If '
            'less than 0, it will be set to the number of pose results in the '
            'first frame.')
        Options.parser.add_argument('--radius', type=int, default=3,
            help='Keypoint radius for visualization.')
        Options.parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization.')
        Options.parser.add_argument('--skeleton-style', default='mmpose', type=str, choices=['mmpose', 'openpose'],
            help='Skeleton style selection')
        Options.parser.add_argument('--black-background', action='store_true', help='Plot predictions on a black image')
        Options.parser.add_argument('--vis-out-dir', type=str, default='', help='Directory for saving visualized results.')
        Options.parser.add_argument('--pred-out-dir', type=str, default='', help='Directory for saving inference results.')
        Options.parser.add_argument('--show-alias', action='store_true', help='Display all the available model aliases.')

        #endregion

        #region options added with HMP

        Options.parser.add_argument("--ckpt_dir", type=str, default="./TIM/checkpoint/test/", help="")
        Options.parser.add_argument("--ckpt_file", type=str, default="ckpt_main_3d_3D_in10_out10_best.pth.tar", help="")
        Options.parser.add_argument("--segments", type=int, default=1, help="")
        Options.parser.add_argument("--savetofile", type=bool, default=False, help="")
        Options.parser.add_argument("--savemode", type=str, default="json", help="")
        Options.parser.add_argument("--k0", type=int, default=2, help="")
        Options.parser.add_argument("--humancap", type=str, default="live", help="")
        Options.parser.add_argument("--humansrc", type=str, default="human_gts_moving.json", help="") # file used to emulate the human
        Options.parser.add_argument("--robotcap", type=str, default="live", help="")
        Options.parser.add_argument("--robotsrc", type=str, default="robot_gts_moving.json", help="") # file used to emulate the robot
        Options.parser.add_argument("--targetbatches", type=int, default=1, help="")
        Options.parser.add_argument("--horizon", type=int, default=2, help="")
        Options.parser.add_argument("--robottask", type=str, default="ping_pong", help="")
        Options.parser.add_argument("--ignorelegs", type=bool, default=False, help="")

        # loi = "Links Of Interest". Takes as input an integer array, where each number a label related to the single joint of the 3D pose.
        #  See "https://mmpose.readthedocs.io/en/latest/dataset_zoo/3d_body_keypoint.html" for more information.
        #  N.B: "nargs" is needed to specify multiple values for the same argument
        Options.parser.add_argument("--loi", nargs="*", type=int, default=None, help="") 

        #endregion

        # parsing args from command line
        opt = vars(Options.parser.parse_args())

        # "special" keys used by MMPose
        init_kws = [
            'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
            'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights'
        ]
    
        Options.init_args.clear()
 
        # create "special" keys dictionary
        for init_kw in init_kws:
            Options.init_args[init_kw] = opt.pop(init_kw)

        # printing args
        print("\n==================Options=================")
        pprint(opt, indent=4)
        print("==========================================\n")

        return opt