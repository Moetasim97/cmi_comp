import os
import pandas as pd
import polars as pl
import os
import pandas as pd
import json
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from joblib import load
from torch.nn import init
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional


tof_data_mean = torch.Tensor([39.7010])[0]
tof_data_std = torch.Tensor([63.7176])[0]
stride=2
ks=3    
ADDED_PADDING=4
TARGET = "gesture"
ID_COL = "sequence_id"

preprocess_tof_tens = lambda tens:torch.sigmoid((tens-tof_data_mean)/tof_data_std).unsqueeze(1).float()
reshape_tof_row = lambda tof_row: torch.reshape(functional.pad(tof_row,(0,ADDED_PADDING)),(18,18))

FEATURES = ['acc_x',
 'acc_y',
 'acc_z',
 'rot_w',
 'rot_x',
 'rot_y',
 'rot_z',
 'thm_1',
 'thm_2',
 'thm_3',
 'thm_4',
 'thm_5',
 'tof_feature_0',
 'tof_feature_1',
 'tof_feature_2',
 'tof_feature_3',
 'tof_feature_4',
 'tof_feature_5',
 'tof_feature_6',
 'tof_feature_7',
 'tof_feature_8',
 'tof_feature_9',
 'tof_feature_10',
 'tof_feature_11',
 'tof_feature_12',
 'tof_feature_13',
 'tof_feature_14',
 'tof_feature_15']

def reverse_conv(ni,nf,stride=1,ks=3,sample_scale=2):
    layer = [nn.UpsamplingNearest2d(scale_factor=sample_scale)]
    layer.append(nn.Conv2d(ni,nf,stride=stride,kernel_size=ks,padding=ks//2))
    return nn.Sequential(*layer)

def collate_fn(entire_tensor):
    x_tensors= y_tensors = torch.stack(entire_tensor)
    standardized_x, standardized_y =torch.sigmoid((x_tensors-tof_data_mean)/tof_data_std),torch.sigmoid((y_tensors-tof_data_mean)/tof_data_std)
    return standardized_x.unsqueeze(1).float(), standardized_y.unsqueeze(1).float()


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x): 
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x -= self.sub
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x



def init_weights(m, leaky=0.):
    if isinstance(m, (nn.Conv1d,nn.Conv2d,nn.Conv3d)): init.kaiming_normal_(m.weight, a=leaky)

model = nn.Sequential(nn.Conv2d(1,8, stride=stride, kernel_size=ks, padding=ks//2),GeneralRelu(leak=0.4,sub=0.1),nn.BatchNorm2d(num_features=8), #9x9
                      nn.Conv2d(8,16, stride=stride, kernel_size=ks, padding=ks//2),GeneralRelu(leak=0.4,sub=0.1),nn.BatchNorm2d(num_features=16),
                      reverse_conv(16,8),GeneralRelu(leak=0.4,sub=0.1),nn.BatchNorm2d(num_features=8),
                      reverse_conv(8,1),nn.ZeroPad2d(-1))
model.apply(init_weights)
encoder = model[:6]
encoder_model = nn.Sequential(encoder,nn.AdaptiveAvgPool2d((1,1)))


def preprocess_tof_data(data,tof_cols):

    data_tof = torch.from_numpy(data[tof_cols].values)
    reshaped_data_tof = [reshape_tof_row(tof_row) for tof_row in data_tof]
    reshaped_tof_rows = torch.stack(reshaped_data_tof)
    stacked_tensor = torch.nan_to_num(reshaped_tof_rows.float(),nan=0.0)

    return stacked_tensor

def transform_tof_data(encoder_model,tof_tensor):
    encoder_model.to("cuda")
    with torch.no_grad():

        output_features = encoder_model(preprocess_tof_tens(tof_tensor).to("cuda"))

    flattened = output_features.view(output_features.size(0), -1).to("cpu")

    return flattened

def perform_grouped_analysis(df, groupby_column, sum_columns, mean_std_columns,target_colname):



        df[target_colname] = df[sum_columns].sum(axis=1)
        mean_std_columns.append(target_colname)
        # 2. Group by the specified column and calculate the sum, mean, and standard deviation
        grouped = df.groupby(groupby_column).agg(
    
            **{f'{col}_mean': (col, 'mean') for col in mean_std_columns},
            **{f'{col}_std': (col, 'std') for col in mean_std_columns}
        )
    
        return grouped

def drop_cols(arbitrary_df,cols_to_drop,inplace=True):
    try:
        arbitrary_df.drop(columns=cols_to_drop,inplace=inplace)
    except:pass
    return arbitrary_df

def reloading_pickles(pckl_file):
    with open(pckl_file,"rb")as file:
        loaded_obj  = pickle.load(file)
    return loaded_obj
    

    
def convert_df(pl_df):return pl_df.to_pandas()

def vanilla_categorical_procs(df,future_cats=None):
    if future_cats is not None:
        try:
            for col in future_cats:
                df[col] = df[col].astype('category')
               
        except:pass #will handle that later on
            
    categorical_cols = [col for col in df.columns if df[col].dtype == 'category']  # Columns with string values
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].cat.codes
    return df

def vanilla_quant_procs(df):
    numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']] # Columns with numerical values
    for col in numerical_cols:
        df[col] =  df[col].fillna(df[col].mean())
    return df



class CMIData(Dataset):
    def __init__(self, data, max_length=None):
        super().__init__()
        self.d_data = dict(list(data.groupby(ID_COL)))
        self.features = FEATURES
        self.target = TARGET
        self.max_length = max_length
        self.keys = list(self.d_data)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        
        df = self.d_data[self.keys[index]]
        if self.max_length is not None:
            df = df.iloc[-self.max_length :]
        return (
            torch.tensor(df[self.features].values.astype(np.float32)),
            torch.tensor(df[self.target].values[-1].astype(np.int64)),
        )