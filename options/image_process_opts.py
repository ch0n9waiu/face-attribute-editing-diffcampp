from argparse import ArgumentParser

class ImageProcessOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--device', type=int, default=0)
        self.parser.add_argument('--diffcam_ckpt_path', type=str, default='./checkpoints/diff_cam_weight.pt')
        self.parser.add_argument('--diffcam_img_size', type=int, default=256)
        self.parser.add_argument('--diffcam_num_class', type=int, default=17)
        self.parser.add_argument('--deghosting_ckpt_path', type=str, default='./checkpoints/deghosting.pt')
        self.parser.add_argument('--deghosting_ckpt_path2', type=str, default='./train_deg/0519/deghosting_checkpoint2/iter_200000')
        self.parser.add_argument('--folder', type=str, default='tu8')
        self.parser.add_argument('--direction_path', type=str, default='./directions2/Old.npy')
        self.parser.add_argument('--direction_path2', type=str, default='./directions2/Bushy_Eyebrows.npy')
        self.parser.add_argument('--alpha', type=int, default=20, help="coefficient to be multiplied to directions")
        self.parser.add_argument('--deghosting_in_size', type=int, default=128)
        self.parser.add_argument('--deghosting_out_size', type=int, default=1024)
        self.parser.add_argument('--needMouth', type=bool, default=False)
        self.parser.add_argument('--aug', type=bool, default=False)
        self.parser.add_argument('--eig', type=bool, default=False)

        self.parser.add_argument('--image_dir', type=str, default='/home/chongyu/Documents/Editing-Out-of-Domain-master/datasets/celeba50')
        self.parser.add_argument('--output_dir', type=str, default='pic4scoretest/')

        self.parser.add_argument('--image_path', type=str, default='/home/chongyu/Documents/Editing-Out-of-Domain-master/datasets/celeba1/000014.jpg')
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