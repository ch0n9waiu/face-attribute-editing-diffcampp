from argparse import ArgumentParser

class GenDatasetOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--device', type=int, default=0)
        self.parser.add_argument('--diffcam_ckpt_path', type=str, default='./checkpoints/diff_cam_weight.pt')
        self.parser.add_argument('--diffcam_num_class', type=int, default=17)
        self.parser.add_argument('--diffcam_img_size', type=int, default=256)
        self.parser.add_argument('--direction_dir', type=str, default='./direction')
        self.parser.add_argument('--alpha', type=int, default=20, help="coefficient to be multiplied to directions")
        self.parser.add_argument('--src_image_dir', type=str, default='/home/chongyu/Documents/Editing-Out-of-Domain-master/datasets/celeba50')
        self.parser.add_argument('--dst_image_dir', type=str, default='./gen_diff_eval/fused')
        self.parser.add_argument('--blend_image_dir', type=str, default='./gen_diff_eval/blend')

        # psp parameters
        self.parser.add_argument('--psp_ckptpath', type=str, default='./checkpoints/psp_ffhq_encode.pt')
        self.parser.add_argument('--psp_encoder_type', type=str, default='GradualStyleEncoder')
        self.parser.add_argument('--psp_input_nc', type=int, default=3)
        self.parser.add_argument('--psp_output_size', type=int, default=1024)
        self.parser.add_argument('--psp_start_from_latent_avg', type=bool, default=True)
        self.parser.add_argument('--psp_learn_in_w', type=bool, default=False)

    def parse(self):
        opts = self.parser.parse_args()
        return opts