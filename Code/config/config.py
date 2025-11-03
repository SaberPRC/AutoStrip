from easydict import EasyDict as edict

__C = edict()
cfg = __C

####### general parameters ######
__C.general = {}
__C.general_server = {}
####### config file in BME-cluster
__C.general_server = {}
__C.general_server.file_list = 'file_list.csv'
__C.general_server.root = '/public_bme2/bme-dgshen/JiamengLiu/BET'
__C.general_server.save_root = '/public_bme2/bme-dgshen/JiamengLiu/BET/Results'

###### training parameters
__C.train = {}
__C.train.num_epochs = 3000
__C.train.batch_size=4

__C.train.lr = 2e-4
__C.train.save_epoch = 1

###### loss function setting ######
__C.loss = {}

# parameters for focal loss
__C.loss.obj_weight = ['99', '1']
__C.loss.gamma = 2

# resume_epoch == -1 training from scratch
__C.general.resume_epoch = -1

# random seed
__C.general.seed = 42

####### dataset parameters #######
__C.dataset = {}
# number of classes
__C.dataset.num_classes = 1
# number of modalities
__C.dataset.num_modalities = 1
# image resolution
__C.dataset.spacing = [1,1,1]
# cropped image patch size
__C.dataset.crop_size = [128, 128, 128]
