import argparse

class inferenece_from_txt_opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video (KITTI and CityScapes)',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)


        self.parser.add_argument("--img-height", default=256, type=int, help="Image height")
        self.parser.add_argument("--img-width", default=832, type=int, help="Image width")
        self.parser.add_argument("--min-depth", default=1e-3)
        self.parser.add_argument("--max-depth", default=80)
        self.parser.add_argument("--dataset_path",
                                 default='/970evo/home/roit/datasets/kitti',
                                type=str,
                                 help="Dataset directory")
        self.parser.add_argument("--split",
                                 default='eigen',
                                 choices=[
                                     'eigen',
                                     'eigen_zhou'
                                 ]
                                 )
        self.parser.add_argument("--txt_file",
                                 default='dali.txt'
                                 )
        self.parser.add_argument("--dataset-list",
                                 default=None,
                                 type=str,
                                 help="Dataset list file")
        self.parser.add_argument("--models_path",
                                 default='/home/roit/models/SCBian')


        self.parser.add_argument("--output-dir",
                                 default='official_eigen',
                                 type=str)
        self.parser.add_argument('--model_name',
                                 default='res18_dispnet.tar')
        self.parser.add_argument("--wk_root",default="/home/roit/aws/aprojects/xdr940_Bian2019")


        self.parser.add_argument('--resnet-layers',
                                 default=18,
                                 choices=[18, 50],
                            help='depth network architecture.')


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options