'''
Author : Payal Mohapatra
Contact : PayalMohapatra2026@u.northwestern.edu
Project : Efficient Multimodal Disfluency Detection
'''
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from cffi.model import qualify
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.cuda import device
from torchsampler import ImbalancedDatasetSampler
import sys
# sys.path.append('/home/payal/multimodal_speech/main_codebase/')
import matplotlib.pyplot as plt
import IPython.display as ipd

import time
from thop import profile
import math
#from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import wandb
import random
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, balanced_accuracy_score


# Custom functions
from helper_functions import set_seed, __shuffle_pick_quarter_data__, save_checkpoint
from helper_functions import AverageMeter, ProgressMeter
from audio_helper_functions import _resample_if_necessary, _cut_if_necessary, _right_pad_if_necessary, _mix_down_if_necessary

#quantification net
from tool.quantification import quantification_net

class AVDataset(Dataset):
    def __init__(self, x_a, x_v, y, n_samples):
        # data loading
        self.x_a = x_a
        self.x_v = x_v
        self.y = y
        self.n_samples = n_samples

    def __getitem__(self, index):
        return self.x_a[index], self.x_v[index], self.y[index]

    def get_labels(self):
        return self.y

    def __len__(self):
        return self.n_samples

class PositionalEncoding(nn.Module):
    # def __init__(self, d_model, dropout, max_len):
    # def __init__(self, device, d_model, dropout=0.0):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = 90
        # max_len = 376 # FIXME :: UPdeate in the class definitions
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # (L, N, F)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    # def forward(self, x, device):
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # x = x + self.pe[:x.size(0)].to(device)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class encoder(nn.Module):
    # def __init__(self, d_model, device): #FIXME
    def __init__(self, d_model): #FIXME
        super(encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=32) ## README: d_model is the "f" in forward function of class network
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) ## num_layers is same as N in the transformer figure in the transformer paper
        # self.positional_encoding = PositionalEncoding(device,d_model)
        self.positional_encoding = PositionalEncoding(d_model)
    def forward(self, tgt):
        # tgt = self.positional_encoding(tgt, device) ##for positional encoding
        tgt = self.positional_encoding(tgt) ##for positional encoding
        out = self.transformer_encoder(tgt) ##when masking not required, just remove mask=tgt_mask
        return out


class RandomMasking(nn.Module,device):
    def __init__(self, p=0.1):
        super(RandomMasking, self).__init__()
        self.p = p
        # self.mask = torch.bernoulli(torch.ones(1) * self.p)
        self.mask = torch.bernoulli(torch.ones(1) * self.p).to(device)
        self.device = device

    def forward(self, x):
        if self.training and self.p > 0:
            # Draw a random number from a Bernoulli distribution
            # mask = torch.bernoulli(self.p).to(device)
            # mask = torch.bernoulli(torch.ones(1) * self.p).to(device)
            # mask = torch.bernoulli(torch.ones(1) * self.p)
            if (self.mask == 0).all():
                print('Dropping Video')
            # mask = torch.bernoulli()
            x = x.to(self.device)
            x = x * self.mask
            x = x.to(self.device)
        return x


############## UNIFIED FUSION ################
class AVStutterNet(nn.Module):
    def __init__(self):
        super(AVStutterNet, self).__init__()
        self.fc_dim = 768 // 2
        # self.fc_dim_clf = 96 * 4


        self.temporal_summarization_a = nn.Sequential(
            nn.Conv1d(in_channels = 149, out_channels = 90, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # self.ln0 = nn.LayerNorm(self.fc_dim)
        self.ln0_a = nn.LayerNorm(self.fc_dim * 2)
        self.ln0_v = nn.LayerNorm(self.fc_dim * 2)


        # Feature summarization -- audio
        self.layer1_a = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5)
        )

        self.layer1_bn_a = nn.BatchNorm2d(1)

        # Feature summarization -- video
        self.layer1_v = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer1_bn_v = nn.BatchNorm2d(1)

        self.tf_encoder = encoder(self.fc_dim)
        # self.tf_encoder = encoder(self.fc_dim, device)

        self.norm1D_a = nn.LayerNorm(self.fc_dim)
        self.norm1D_v = nn.LayerNorm(self.fc_dim)
        self.norm1D_f = nn.LayerNorm(self.fc_dim)

        self.clf_head = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.fc_dim // 2, self.fc_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(self.fc_dim // 4, 2),
            # nn.Softmax(dim=1)
        )

        # self.param_a = torch.nn.Parameter(torch.randn(768))
        # self.param_v = torch.nn.Parameter(torch.randn(768))

        # Create a learnable parameter with a constraint
        self.param_a = torch.nn.Parameter(torch.randn(self.fc_dim), requires_grad=True)
        # Set the constraint
        self.param_a.register_hook(lambda grad: grad.clamp(min=0, max=1))

        self.param_v = torch.nn.Parameter(torch.randn(self.fc_dim), requires_grad=True)
        # Set the constraint
        self.param_v.register_hook(lambda grad: grad.clamp(min=0, max=1))


    def forward(self, x_a, x_v):
        ## RandomMask augmentation
        if (len(x_a.shape) == 3):
            x_a = x_a.unsqueeze(1)
        if (len(x_v.shape) == 3):
            x_v = x_v.unsqueeze(1)

        ########## Project audio to same temporal dimension and fuse ##########
        if (len(x_a.shape) == 4):
            x_a = x_a.squeeze(1)
        if (len(x_v.shape) == 4):
            x_v = x_v.squeeze(1)
        x_a = self.temporal_summarization_a(x_a)

        x_a =self.ln0_a(x_a)
        x_v =self.ln0_v(x_v)

        ######### AUDIO feature summarization #########
        if (len(x_a.shape) == 3):
            x_a = x_a.unsqueeze(1)
        out_a = self.layer1_a(x_a)
        out_a = self.layer1_bn_a(out_a)
        # squeeze out_a
        out_a = out_a.squeeze(1)

        ########## AUDIO through common temporal encoder ##########
        ### Call the temporal learning module : TF encoder
        # Current input -- B,T,F
        # Expected input -- T,B,F
        out_a = out_a.permute(1, 0, 2)
        out_a = self.tf_encoder(out_a)
        ## Permute back to B,T,F
        out_a = out_a.permute(1, 0, 2)
        ## Normalize out_a
        out_a =  self.norm1D_a(out_a)
        out_a = out_a.mean(1, keepdim=True)  ## Mean pool along the temporal axis
        out_a = out_a.squeeze(1)

        ######### VIDEO feature summarization #########
        if (len(x_v.shape) == 3):
            x_v = x_v.unsqueeze(1)
        out_v = self.layer1_v(x_v)
        out_v = self.layer1_bn_v(out_v)
        # squeeze out_v
        out_v = out_v.squeeze(1)

        ## Pass only if all x_v is not 0
        if (x_v == 0.0).all().item():
            out_v = out_a
        else:
            ########## VIDEO through common temporal encoder ##########
            ### Call the temporal learning module : TF encoder
            # Current input -- B,T,F
            # Expected input -- T,B,F
            out_v = out_v.permute(1, 0, 2)
            out_v = self.tf_encoder(out_v)
            ## Permute back to B,T,F
            out_v = out_v.permute(1, 0, 2)
            ## Normalize out_v
            out_v =  self.norm1D_v(out_v)
            out_v = out_v.mean(1, keepdim=True)  ## Mean pool along the temporal axis
            out_v = out_v.squeeze(1)
        ########## FUSE both the feature sets by adding ##########
        out = self.param_a * out_a + self.param_v * out_v

        # Normalise the fused feature
        out = self.norm1D_f(out)
        # unsqueeze the feature dimension
        out = out.view(out.size(0), -1)
        # print('Input data shape after view:', x.shape)
        out = self.clf_head(out)


        return out

if __name__ == '__main__':

    ##################################################################################################
    # Global Variables
    ##################################################################################################
    target_sample_rate = 16000 ## We need all audio to be of 16kHz sampling rate
    num_samples = target_sample_rate * 3


    root_dir = 'G:/HKUST_DATA/5202/27216024/Multimodal_dataset/Multimodal_dataset/Fluencybank/FB_audio/'
    root_dir_video_feat = 'G:/HKUST_DATA/5202/27216024/Multimodal_dataset/Multimodal_dataset/Fluencybank/FB_video_feat/'

    meta_data_dir = './metadata/'
    ##################################################################################################
    # Add arguments you want to pass via commandline
    ##################################################################################################
    parser = argparse.ArgumentParser(description='EfficientMM')
    parser.add_argument('--chkpt_pth',
                        default='G:/HKUST_DATA/5202/27216024/Multimodal_dataset/multimodal_speech/main_codebase/saved_models/audio_video_only/',
                        type=str,
                        metavar='N',
                        help='which checkpoint do you wanna use to extract embeddings?')
    parser.add_argument('--model_name',
                        default='baseline_0053.pth',
                        type=str,
                        metavar='N',
                        help='which checkpoint do you wanna use to extract embeddings?')
    parser.add_argument('--num_epochs', default=1, type=int,
                        metavar='N',
                        )

    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N',
                        )

    parser.add_argument('--cuda_pick', default='cuda:0', type=str,
                        metavar='use the different cuda ids if you have multiple GPUs.',
                        )

    parser.add_argument('--stutter_type', default='Block/', type=str,
                        metavar='Acceptable values : Block/, Prolongation/, Interjection/, SoundRep/, WordRep'
                        )
    parser.add_argument('--seed_num', default=123, type=int,
                        metavar='seed_num',
                        )

    parser.add_argument('--p_mask', default=0.5, type=float,
                        metavar='Drop ratio in the augmentation (value between 0 and 1)',
                        )

    parser.add_argument('--lr', default=1e-5, type=float,
                        metavar='learning rate',
                        )
    parser.add_argument('--test_matric', default=True, type=bool, help="Set to Ture to get floap and fps if needed")
    parser.add_argument('--execute_quantification', default=True, type=bool, help="Set to Ture to quantify if needed")

    args = parser.parse_args()

    num_epochs = args.num_epochs
    model_chkpt_pth = args.chkpt_pth
    target_file = args.model_name
    cuda_pick = args.cuda_pick
    batch_size = args.batch_size
    seed_num = args.seed_num
    p_mask = args.p_mask
    lr = args.lr
    test_matric = args.test_matric
    trained_epoch = 0
    base_name = args.model_name.rsplit('_', 1)[0]


    ##################################################################################################
    set_seed(seed_num)
    device = torch.device(cuda_pick if torch.cuda.is_available() else "cpu")
    print(device)
    ## If the checkpoint path is not present create it
    if not os.path.exists(model_chkpt_pth):
        os.makedirs(model_chkpt_pth)

    wandb.login()

    for filename in os.listdir(model_chkpt_pth):
        if filename.endswith(target_file):
            checkpoint = torch.load(model_chkpt_pth + filename)
            trained_epoch = checkpoint['epoch']


    run = wandb.init(
        # Set the project where this run will be logged
        project="Multimodal-Speech-Disfluency",
        name= "Multimodal-Speech-Disfluency",
        settings=wandb.Settings(init_timeout=120),
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.01,
            "current_epochs":  trained_epoch,
            "num_epochs_to_train": args.num_epochs,
            "model_chkpt_pth": args.chkpt_pth,
            "cuda_pick": args.cuda_pick,
            "batch_size": args.batch_size,
            "seed_num": args.seed_num,
            "p_mask": args.p_mask,
            "lr": args.lr,
            "test_matric" : args.test_matric
        },
    )
    ##################################################################################################
    # Read the train-test split file and get samnples
    ##################################################################################################
    stutter_type = args.stutter_type
    fluent_type = 'NoStutteredWords/'

    if (stutter_type == 'Block/') :
        train_file_stutter = meta_data_dir + 'fb_train_block_123.csv'
        test_file_stutter =  meta_data_dir + 'fb_test_block_123.csv'
    elif (stutter_type == 'Prolongation/') :
        train_file_stutter = meta_data_dir + 'fb_train_pro_123.csv'
        test_file_stutter =  meta_data_dir + 'fb_test_pro_123.csv'
    elif (stutter_type == 'Interjection/') :
        train_file_stutter = meta_data_dir + 'fb_train_intrj_123.csv'
        test_file_stutter =  meta_data_dir + 'fb_test_intrj_123.csv'
    elif (stutter_type == 'SoundRep/') :
        train_file_stutter = meta_data_dir + 'fb_train_snd_123.csv'
        test_file_stutter =  meta_data_dir + 'fb_test_snd_123.csv'
    elif (stutter_type == 'WordRep/') :
        train_file_stutter = meta_data_dir + 'fb_train_wp_123.csv'
        test_file_stutter =  meta_data_dir + 'fb_test_wp_123.csv'


    train_file_fluent =  meta_data_dir + 'fb_train_fluent_123.csv'
    test_file_fluent =   meta_data_dir + 'fb_test_fluent_123.csv'

    train_df_stutter = pd.read_csv(train_file_stutter, header=None)
    test_df_stutter = pd.read_csv(test_file_stutter, header=None)

    train_df_fluent = pd.read_csv(train_file_fluent, header=None)
    test_df_fluent = pd.read_csv(test_file_fluent, header=None)


    ##################################################################################################
    # Prepare the audio data using wav2vec2 features
    ##################################################################################################
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model_wav2vec = bundle.get_model().to(device)
    def __extract_audio_feat__(signal, sr=16000) :
        signal = _resample_if_necessary(signal, sr)
        signal = _mix_down_if_necessary(signal)
        signal = _cut_if_necessary(signal)
        signal = _right_pad_if_necessary(signal)
        audio_feat, _ = model_wav2vec(signal.to(device))
        if audio_feat.shape[0] > 1 : # handle multi-channel audio
            audio_feat = torch.mean(audio_feat, dim=0, keepdim=True)
            audio_feat = audio_feat.unsqueeze(0)
        audio_feat = audio_feat.cpu().detach().numpy()
        return audio_feat

    ### NOTE : The audio and video functions are customized for the paired audio-video setting.
    def __audio_datasetprep__(train_df, root_dir, stutter_type,label, discard_v_list) :
        x_s = []
        y_s = []
        discarded_s = 0
        # check for discarded samples from video dataset
        if len(discard_v_list) > 0:
            # print('Length of training set before discarding:', len(train_df))
            for i in range(len(discard_v_list)):
                if discard_v_list[i] in train_df[0].values:
                    # print('Discarded sample:', discard_v_list[i])
                    train_df = train_df[train_df[0] != discard_v_list[i]]
            # print('Length of training set after discarding:', len(train_df))
            # print(train_df)
        # reindex train_df and convert to list for simplicity
        train_df_list = train_df.values.tolist()
        num_files = len(train_df_list)
        for i in tqdm(range(num_files)):
            # print('Processing sample:', root_dir + stutter_type + str(train_df_list[i][0]) + '.wav')
            signal, _ = torchaudio.load(root_dir + stutter_type + str(train_df_list[i][0]) + '.wav')
            signal_np = __extract_audio_feat__(signal)
            # print(np.shape(stutter_np))
            # breakpoint()
            # fluent_np --> (1, 149, 768)
            if ((np.shape(signal_np)[0] != 1) |(np.shape(signal_np)[1] != 149) | (np.shape(signal_np)[2] != 768)) :
                discarded_s += 1
                # print('Discarded sample shape:', signal_np.shape)
            else:
                x_s.append(signal_np)
                y_s.append(label)
        if discarded_s > 0:
            print("Number of discarded samples in audio: ", discarded_s)
        return x_s, y_s



    def __video_datasetprep__(train_df, stutter_type,label) :
        x_s = []
        y_s = []
        discard_list = []
        discarded_s = 0
        num_files = len(train_df)
        for i in tqdm(range(num_files)):
            signal_np= np.load(root_dir_video_feat + stutter_type + train_df[0][i] + '.npy')
            signal_np = np.reshape(signal_np, (1, signal_np.shape[0], signal_np.shape[1]))
            # breakpoint()
            # fluent_np --> (1, 149, 768)
            if ((np.shape(signal_np)[0] != 1) |(np.shape(signal_np)[1] != 90) | (np.shape(signal_np)[2] != 768)) :
                discarded_s += 1
                # print('Discarded sample shape:', signal_np.shape)
                discard_list.append(train_df[0][i])

            else:
                x_s.append(signal_np)
                y_s.append(label)
        if discarded_s > 0:
            print("Number of discarded samples in video: ", discarded_s)
        return x_s, y_s, discard_list


    # **************** Training Set Extraction ****************
    # Get the extracted stutter features
    x_v_s, y_v_s, discard_v_s = __video_datasetprep__(train_df_stutter, stutter_type, 1)
    x_a_s, y_a_s = __audio_datasetprep__(train_df_stutter, root_dir, stutter_type, 1, discard_v_s)


    # Get the extracted fluent features
    x_v_f, y_v_f, discard_v_f = __video_datasetprep__(train_df_fluent, fluent_type, 0)
    x_a_f, y_a_f = __audio_datasetprep__(train_df_fluent, root_dir, fluent_type, 0, discard_v_f)


    if (len(y_a_f) != len(y_v_f)) | (len(y_a_s) != len(y_v_s)):
        print('TRAIN DATA MISMATCH --- The number of samples in the two classes are not equal!')
        sys.exit()

    # x_train, y_train = __shuffle_pick_quarter_data__ (x_f, y_f, x_s, y_s)   # FIXME :: Verify for the training setting. Do we need to synthetically balance the majority class?

    random.shuffle(x_a_f)
    random.shuffle(x_a_s)
    random.shuffle(x_v_f)
    random.shuffle(x_v_s)

    x_train_audio = x_a_s + x_a_f
    x_train_video = x_v_s + x_v_f
    y_train = y_a_s + y_a_f



    # **************** Testing Set Extraction ****************
    # Get the extracted stutter features
    x_v_t_s, y_v_t_s, discard_v_t_s = __video_datasetprep__(test_df_stutter, stutter_type, 1)
    x_a_t_s, y_a_t_s = __audio_datasetprep__(test_df_stutter, root_dir, stutter_type, 1, discard_v_t_s)



    # Get the extracted fluent features
    x_v_t_f, y_v_t_f, discard_v_t_f = __video_datasetprep__(test_df_fluent, fluent_type, 0)
    x_a_t_f, y_a_t_f = __audio_datasetprep__(test_df_fluent, root_dir, fluent_type, 0, discard_v_t_f)

    if (len(y_a_t_f) != len(y_v_t_f)) | (len(y_a_t_s) != len(y_v_t_s)):
        print('TEST DATA MISMATCH --- The number of samples in the two classes are not equal!')
        sys.exit()
    random.shuffle(x_a_t_f)
    random.shuffle(x_a_t_s)
    random.shuffle(x_v_t_f)
    random.shuffle(x_v_t_s)

    x_test_audio = x_a_t_s + x_a_t_f
    x_test_video = x_v_t_s + x_v_t_f
    y_test = y_a_t_s + y_a_t_f

    ##################################################################################################
    # Dataloading
    ##################################################################################################


    n_samples_train = np.shape(x_train_audio)[0]
    n_samples_test = np.shape(x_test_audio)[0]

    train_dataset_main = AVDataset(x_train_audio, x_train_video,y_train,n_samples_train)
    test_dataset = AVDataset(x_test_audio, x_test_video, y_test,n_samples_test)

    # Split the dataset into train and valid
    train_dataset, valid_dataset = train_test_split(train_dataset_main, test_size=0.3, random_state=123, shuffle=True, stratify = y_train)

    print('Number of samples to train = ', len(train_dataset))
    print('Number of samples to validate = ', len(valid_dataset))
    print('Number of samples to test = ', len(test_dataset))




    train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                            #   sampler=ImbalancedDatasetSampler(train_dataset_main), # FIXME :: getlabels error if I use train_dataloader
                              shuffle=True,
                              num_workers=1)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                              batch_size=len(valid_dataset),
                              shuffle=True,
                              num_workers=1)
    test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=len(test_dataset),
                              shuffle=True,
                              num_workers=1)


    ##################################################################################################
    # Model Definition
    ##################################################################################################
    feat_channel = 3

    # ############### Optimiser and Loss Function ################
    model = AVStutterNet().to(device)
    # print(model)
    print('Number of Trainable parameters', sum(p.numel()for p in model.parameters()))
    class_loss_criterion =nn.CrossEntropyLoss()
    sim_loss_criterion   = nn.CosineEmbeddingLoss(margin=0.5)

    optimizer=torch.optim.Adam(model.parameters(), lr=lr)

    def l1_regularization(model, lambda_=1e-5):
        l1_reg = 0
        for name, param in model.named_parameters():
            if 'bias' not in name:  # Optionally exclude bias terms from regularization
                l1_reg += torch.sum(torch.abs(param))
        return lambda_ * l1_reg

    ##################################################################################################
    # Training and Evaluation
    ##################################################################################################
    from helper_functions import AverageMeter, ProgressMeter
    def train_one_epoch(train_loader, model, class_loss_criterion, optimizer, epoch,device):
        sim_loss_list = np.zeros(len(train_loader))
        ce_loss_list = np.zeros(len(train_loader))
        class_loss_list = np.zeros(len(train_loader))
        overall_loss_list = np.zeros(len(train_loader))
        class_acc_list = np.zeros(len(train_loader))

        loss_class = AverageMeter('Class Loss', ':.4f')
        loss_overall = AverageMeter('Overall Loss', ':.4f')
        loss_sim = AverageMeter('Overall Loss', ':.4f')
        loss_class = AverageMeter('Overall Loss', ':.4f')

        model.train()
        model.zero_grad()


        for i,(feat_a, feat_v, y) in enumerate(train_loader) :
            correct = 0
            # print(y)
            y = y.to(device)
            feat_a = feat_a.to(device).float()
            feat_v = feat_v.to(device).float()
            # print(feat_a.shape)
            # print(feat_v.shape)
            feat_v = RandomMasking(p=0.5)(feat_v)
            class_output = model(feat_a, feat_v)

            # loss list of a batch
            loss_class_iter = class_loss_criterion(class_output, y)
            # loss_sim_iter = sim_loss_criterion(out_a, out_v,).mean()

            loss = loss_class_iter

            ## Compute the accuracy per batch
            _, predicted_labels = torch.max(class_output.data, 1)
            correct += predicted_labels.eq(y).sum().item()

            # Average loss of a batch
            curr_loss = loss_overall.update(loss.item(), feat_a.size(0))
            curr_class_loss = loss_class.update(loss_class_iter.item(), feat_a.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Maintain the list of all losses per iteration (= total data/batch size)
            class_loss_list[i] = curr_class_loss
            overall_loss_list[i] = curr_loss


            # class_acc_list[i] = curr_class_acc
            # print('------------------------------')
            # print('Predicted labels:', predicted_labels)
            # print('True labels:', y)
            # print('------------------------------')
            class_acc_list[i] = balanced_accuracy_score(y.cpu().numpy(), predicted_labels.cpu().numpy())
            acc_class_meter = 'Class Acc: {:.4f}'.format(class_acc_list[i])

            progress = ProgressMeter(
            len(train_loader), # total_data/batch_size
            [loss_overall, loss_class, acc_class_meter],
            prefix="Epoch: [{}]".format(epoch))


            if (i % 50 == 0) | (i == len(train_loader)-1):
                progress.display(i)
            if (i == len(train_loader)-1):
                print('End of Epoch', epoch, 'Overall loss is','%.4f' % np.mean(overall_loss_list), '    Training accuracy is ', '%.4f' % np.mean(class_acc_list))

            # print(overall_loss_list)
            # print(sim_loss_list)
        return overall_loss_list, class_loss_list, class_acc_list


    def evaluate_one_epoch(valid_loader, model, class_loss_criterion, optimizer, epoch,device):
        ## Assume there is no mini-batch in validation
        ## Batch Size is same as length of all samples
        with torch.no_grad():
            correct_v = 0
            class_loss_list = np.zeros(len(valid_loader))

            #Gives output per example since batch_size = 1
            for i,(feat_a, feat_v, y) in enumerate(valid_loader) :
                model.eval()
                feat_v = RandomMasking(p=p_mask)(feat_v)
                y = y.to(device)
                feat_a = feat_a.to(device).float()
                feat_v = feat_v.to(device).float()

                class_output= model(feat_a, feat_v)

                # loss list of a batch
                class_loss_list[i] = class_loss_criterion(class_output, y)

                ## Compute class accuracy
                _, predicted_labels_v = torch.max(class_output.data, 1)


            class_loss_valid = sum(class_loss_list)/len(class_loss_list)
            # class_acc_valid = correct_v/len(class_loss_list)
            class_acc_valid = balanced_accuracy_score(y.cpu().numpy(), predicted_labels_v.cpu().numpy())
            f1_score_valid = f1_score(y.cpu().numpy(), predicted_labels_v.cpu().numpy())
            # Get F1 score
            # f1_score_v = f1_score(y, predicted_labels_v)
            # print('------------------------------')
            # print('Predicted labels:', predicted_labels_v)
            # print('True labels:', y)
            # print('------------------------------')


        return class_loss_valid, class_acc_valid, f1_score_valid

    def matric_flop(test_loader, model,device):
        for i, (feat_a, feat_v, y) in enumerate(test_loader):
            model.eval()
            feat_v = RandomMasking(p=p_mask)(feat_v)
            feat_a = feat_a.to(device).float()
            feat_v = feat_v.to(device).float()
            flops, params = profile(model, (feat_a, feat_v,))
            break
        print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
        wandb.log({'flops': flops, 'params': params})


    def matric_FPS(valid_loader, model,device):
        for i, (feat_a, feat_v, y) in enumerate(valid_loader):
            model.eval()
            feat_v = RandomMasking(p=p_mask)(feat_v)
            feat_a = feat_a.to(device).float()
            feat_v = feat_v.to(device).float()
            torch.cuda.synchronize()
            start = time.time()
            class_output = model(feat_a, feat_v)
            torch.cuda.synchronize()
            end = time.time()
            break
        print('infer_time:', end - start)
        wandb.log({'infer_time': end - start})

    test_acc_list = np.zeros(num_epochs)
    test_f1_list = np.zeros(num_epochs)

    trained_epoch = 0
    for filename in os.listdir(model_chkpt_pth):
        if filename.endswith(target_file):
            checkpoint = torch.load(model_chkpt_pth + filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trained_epoch = checkpoint['epoch']


    for epoch  in range(trained_epoch, num_epochs + trained_epoch):
        print('Inside Epoch : ', epoch )

        # train for one epoch
        overall_loss_list, class_loss_list, class_acc_train = train_one_epoch(train_dataloader, model, class_loss_criterion, optimizer, epoch,device)

        # average loss through all iterations --> Avg loss of an epoch
        overall_loss_epoch = sum(overall_loss_list)/len(overall_loss_list)
        class_loss_epoch = sum(class_loss_list)/len(class_loss_list)
        class_acc_epoch = sum(class_acc_train)/len(class_acc_train)
        wandb.log({"Overall Loss/train": overall_loss_epoch, "epoch": epoch})
        wandb.log({"Class Loss/train": class_loss_epoch, "epoch": epoch})
        wandb.log({"Accuracy/train": class_acc_epoch, "epoch": epoch})



        ## Evaluate every epoch for in-domain data in validation # FIXME :: Currently the valid and test sets are the same.
        class_loss_valid, class_acc_valid, _ = evaluate_one_epoch(valid_dataloader, model, class_loss_criterion, optimizer, epoch,device)
        wandb.log({"Accuracy/valid": class_acc_valid, "epoch": epoch})
        wandb.log({"Class Loss/valid": class_loss_valid, "epoch": epoch})


        ## Evaluate every epoch for in-domain data in validation # FIXME :: Currently the valid and test sets are the same.
        class_loss_test, class_acc_test, f1_score_test = evaluate_one_epoch(test_dataloader, model, class_loss_criterion, optimizer, epoch,device)
        # print('End of Epoch', epoch, 'Test loss is','%.4f' % class_loss_test, '    Test accuracy is ', '%.4f' % class_acc_test, '    F1 score is ', '%.4f' % f1_score)
        wandb.log({"Accuracy/test": class_acc_test, "epoch": epoch})
        wandb.log({"Class Loss/test": class_loss_test, "epoch": epoch})
        wandb.log({"F1/test": f1_score_test, "epoch": epoch})

        test_acc_list[epoch - trained_epoch] = class_acc_test
        test_f1_list[epoch - trained_epoch] = f1_score_test


    save_checkpoint({
        'epoch': num_epochs + trained_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
    }, filename=model_chkpt_pth +f'{base_name}{num_epochs + trained_epoch:04d}.pth')

    if test_matric:
        matric_FPS(test_dataloader, model,device)
        matric_flop(test_dataloader, model,device)


wandb.finish()


