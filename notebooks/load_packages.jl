using DataFrames
using CSV
using Random
using SpinGEMPIC

import SpinGEMPIC: set_common_weight
import SpinGEMPIC: get_s1, get_s2, get_s3
import SpinGEMPIC: set_s1, set_s2, set_s3
import SpinGEMPIC: set_weights, get_weights
import SpinGEMPIC: set_x, set_v

import SpinGEMPIC: operatorHE
import SpinGEMPIC: operatorHp
import SpinGEMPIC: operatorHA
import SpinGEMPIC: operatorHs

import GEMPIC: OneDGrid, Maxwell1DFEM
import GEMPIC: l2projection!


