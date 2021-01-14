
import os
#from scipy import io
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from os.path import join as pjoin

import tensorflow as tf

config = tf.ConfigProto()

config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=25)
parser.add_argument('--n_his', type=int, default=6)
parser.add_argument('--n_pred', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=2)
parser.add_argument('--kt', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='sep')

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 8, 16]]
# Load wighted adjacency matrix W
if args.graph == 'default':
    #W = weight_matrix(pjoin('./dataset', 'PeMSD7_W_228{n}.csv'))
    W = weight_matrix(pjoin('./datasets', 'sstw2.csv'))

else:
    # load customized graph weight matrix
    #W = weight_matrix(pjoin('./dataset', args.graph))
    W = weight_matrix(pjoin('./datasets', args.graph))

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).


#L=first_approx(W,n)
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
data_file = 'sstmonth.csv'
n_train, n_val, n_test = 36, 1, 1
n_train, n_val, n_test = 36, 1, 1
PeMS = data_gen(pjoin('./datasets', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(PeMS)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
