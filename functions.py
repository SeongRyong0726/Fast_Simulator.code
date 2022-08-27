import numpy as np
import math
from Systolic_array import *

############################################################################################################################
####    FP MATRIX
####    Input = matrix                  dimension, (B = 1, num_channels, ifmap_height, ifmap_width)
####    Output = matrix(with padding)   dimension, (B = 1, num_channels, ifmap_height + 2f, ifmap_width + 2f) 
####    (  f = padding_size = (filtersize -1)/2  )
def padding_matrix(filter_size, matrix): ##4차원
    ####    right_padding_size > left_padding_size (if filter_size is ever)
    padding_width_right = math.ceil((filter_size -1)/2)
    padding_width_left = math.floor((filter_size -1)/2)

    padding_matrix = np.pad(matrix, ((0,0),(0,0),(padding_width_left,padding_width_right),(padding_width_left,padding_width_right)), 'constant', constant_values = 0)
    return padding_matrix

############################################################################################################################

def fp_to_BFP_Matrix(matrix):
    return matrix

def Make_WS_1st_Input(BFP_matrix, fMAC_group):
    d = 1
def Make_WS_2nd_Input(BFP_WS_F_Weight, fMAC_group):
    a= 1

if __name__ == "__main__":
    # matrix = np.full((1,3,5,5),1)
    # print(matrix)
    # result = padding_matrix(4,matrix)
    # print(result)
    ifmap_height=7
    ifmap_width=7
    filt_height=3
    filt_width=3 
    num_filt=7
    num_channels=3
    strides=1

    dim_col = 4
    dim_row = 4

    # 
    matrix_A = np.full((filt_height * filt_width * num_channels, ifmap_height * ifmap_width, 5),1)
    #
    matrix_W = np.full((filt_height * filt_width * num_channels,num_filt,   5 ),1)
    #print(matrix_A)
    #print(matrix_W)
    r2c = filt_height * filt_width * num_channels


    Sample = Systolic_array(dim_col, dim_row, fMAC, 4, "Forward_WS", matrix_A, matrix_W, None, None, r2c, num_filt, r2c, r2c)

    ####_______________________________________________________________________________________________________________________________________  
    ####    Functions.py TEST1
    ####    Check fold, use value.!

    # print(r2c)          ### col
    # print(num_filt)     ### row
    # for i in range(Sample.col_fold_end*Sample.row_fold_end):
    #     print(Sample.col_fold_current, Sample.row_fold_current)
    #     print(Sample.col_use, Sample.row_use)
    #     print()
    #     Sample.update_fold_variable_F_WS()
    ####_______________________________________________________________________________________________________________________________________


    ####_______________________________________________________________________________________________________________________________________
    ####    Functions.py TEST2
    ####    Check separate exp, mantissa (sram --> fmac)

        ####    [1 2 3 4 5] --> 1  [2 3 4 5]
    # one_group = np.array([1,2,3,4,5])
    # exp, mantissa = Sample.separate_exp_man(one_group)
    # print(one_group)
    # print(exp, mantissa)
    ####_______________________________________________________________________________________________________________________________________

    ####_______________________________________________________________________________________________________________________________________
    ####    Functions.py TEST3
    ####    Check Data_flow

        ####    weight값이 잘 load 되는지 확인
    array_W = np.full((Sample.dim_row, Sample.dim_col),0)
    Sample.Systolic_preload_Weight()
    for row in range(Sample.dim_row):
            for col in range(Sample.dim_col):
                array_W[row][col] = Sample.Unit_array[row][col].weight_exponant


        ####    값 flow가 잘 되는지 확인
    array_Flow = np.full((Sample.dim_row, Sample.dim_col),0)
    print(array_Flow)
    for i in range((r2c + Sample.dim_col + Sample.row_use)*20):
        #### error!! 끝이 안난다. numpy로 값을 저장하고 흘리는데 
        #### 그 뒤에 neg_inf = -1 * math.pow(2,32) 를 (padding으로) 넣어서 signal로(1cycle함수에 투입) 존재하게 해야할듯
        Sample.Systolic_Array_1cycle_calculation()
        
        for row in range(Sample.dim_row):
            for col in range(Sample.dim_col):
                array_Flow[row][col] = Sample.Unit_array[row][col].exponant_from_down
        print(i , array_Flow)
    ####_______________________________________________________________________________________________________________________________________



    ####_______________________________________________________________________________________________________________________________________
    ####    Functions.py TEST4
    ####    Check Forward WS

    ####_______________________________________________________________________________________________________________________________________
    




    ####_______________________________________________________________________________________________________________________________________
    ####    Functions.py TEST5
    ####    Check BackWard WS

    ####_______________________________________________________________________________________________________________________________________
    




    ####_______________________________________________________________________________________________________________________________________
    ####    Functions.py TEST6
    ####    Check Backward OS

    ####_______________________________________________________________________________________________________________________________________
    