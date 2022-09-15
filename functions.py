import numpy as np
import math
from Systolic_array import *
from tqdm import tqdm

neg_inf = -1 * math.pow(2,32)
############################################################################################################################
####    Convolution: first input matrix(=Input) * second input matrix(=filter) = Output(=output)
############################################################################################################################
####    This file is about pre-processing of Two Matrix which will be Dot-Producted.
####    According to the types of calculation (Forward WS, Backward WS, Backward OS), We define the pre-processing type with two.
####    1. WS_preprocessing
####    2. OS_preprocessing
####    two function get two matrix, but return matrix is different (due to its calculation way)
####   
####    Commom process is Here
####    a. padding to first matrix(to make output's size same with first input matrix)          (function: padding_matrix(filter_size, matrix))
####    b. through Im2col 4 Dimension --> 2 Dimention                                           (function: Im2col_WS(    ), Im2col_OS)
####    c. decide procision of the "matrix" & change from fp_Im2col_matrix to bfp_Im2col_matrix
####            consider groupsize (g values on one cell)                                       (function: TEST_BFP_precision(matrix), Im2col_fp_to_bfp)
####    d. skew to make perfect_fit input for Systolic array
############################################################################################################################

# TEST A
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

# TEST B (WS), TEST C (OS)
############################################################################################################################
####    Im2col
####    Input: pad_matrix_1, matrix_2, amtrix_1_shape  #### type: numpy_array(4 Dimension), numpy_array(4 Dimension), tuple(int,int,int,int)
####    Output: Im2col_matrix_1 (num_data_1*num_h_1*num_w_1, num_ch*num_h_2*num_w_2)
####            Im2col_matrix_2 (num_data_2, num_ch*num_h_2*num_w_2) or (num_ch*num_h_2*num_w_2, num_data_2)
def Im2col(pad_matrix_1, matrix_2, matrix_1_shape, mode, OS_result_h, OS_result_w): 
    num_data_2 = matrix_2.shape[0]
    num_ch = matrix_1_shape[1]
    num_h_2 = matrix_2.shape[2]
    num_w_2 = matrix_2.shape[3]

    num_data_1 = matrix_1_shape[0]
    num_h_1 = matrix_1_shape[2]
    num_w_1 = matrix_1_shape[3]

    ## Im2col_matrix_1
    if(mode == "Forward_WS" or mode =="Backward_WS"):
        Im2col_matrix_1 = np.full((num_data_1*num_h_1*num_w_1, num_ch*num_h_2*num_w_2), 0)   
        for b in range(num_data_1):
            for h in range(num_h_1):
                for w in range(num_w_1):
                    one_px = pad_matrix_1[b, 0:num_ch, h:h+num_h_2, w:w+num_w_2]
                    #print(one_px)
                    px_num = num_h_1*num_w_1*b + num_w_1*h + w
                    # put 4D matrix values in to Im2col matrix in one row
                    for i in range(num_ch):
                        for j in range(num_h_2):
                            for k in range(num_w_2):
                                Im2col_matrix_1[px_num][num_w_2*num_h_2*i + num_w_2*j + k]= one_px[i][j][k]
    elif(mode == "Backward_OS"):
        Im2col_matrix_1 = np.full((num_ch*num_h_1*num_w_1, num_data_1* OS_result_h*OS_result_w), 0) 
        for b in range(num_data_1):
            for h in range(OS_result_h):
                for w in range(OS_result_w):
                    one_px = pad_matrix_1[b, 0:num_ch, h:h+num_h_2, w:w+num_w_2]
                    #print(one_px)
                    px_num = OS_result_h*OS_result_w*b + OS_result_w*h + w
                    # put 4D matrix values in to Im2col matrix in one row
                    for i in range(num_ch):
                        for j in range(num_h_2):
                            for k in range(num_w_2):
                                Im2col_matrix_1[num_w_2*num_h_2*i + num_w_2*j + k][px_num]= one_px[i][j][k]



    #Im2col_matrix_1[num_data_1*num_h_1*num_w_1][:] = np.full((1,num_ch*num_h_2*num_w_2), neg_inf) ## end point
    if(mode == "Forward_WS"):
        Im2col_matrix_2 = np.full((num_data_2, num_ch*num_h_2*num_w_2),1)
        for n in range(num_data_2):
            for c in range(num_ch):
                for h in range(num_h_2):
                    for w in range(num_w_2):
                        Im2col_matrix_2[n][num_h_2*num_w_2*c + num_w_2*h + w] = matrix_2[n][c][h][w]
    elif(mode=="Backward_WS"):
        Im2col_matrix_2 = np.full((num_ch*num_h_2*num_w_2, num_data_2), 1)
        for n in range(num_data_2):
            for c in range(num_ch):
                for h in range(num_h_2):
                    for w in range(num_w_2):
                        Im2col_matrix_2[num_h_2*num_w_2*c + num_w_2*h + w][n] = matrix_2[n][c][h][w]
    elif(mode=="Backward_OS"):
        Im2col_matrix_2 = np.full((num_ch*num_h_2*num_w_2, num_data_2), 1)
    else:
        print("error: in Im2col function mode is not found")

    return Im2col_matrix_1, Im2col_matrix_2


############################################################################################################################

# TEST D(precision)
# TEST E(fp-->bfp_group matrix)
# TEST F(bfp_group matrix --> skewed_bfp_group matrix)
############################################################################################################################
####    fp_to_BFP_Matrix
####    BFP_decision ()
def TEST_BFP_precision(matrix): #return precision <2 or 4>
    a=1
def Im2col_fp_to_bfp(Im2col_matrix, bfp_precision_me, bfp_precision_other, group_size): # from (fp) -> (bfp + groupsize)
    a=1
def SKEW_matrix(bfp_Im2col_matrix_1, dim_row, dim_col, direction): ## skew to be input ## direction is "go through direction of row or col?"
    a=1
############################################################################################################################



















############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
####    WS preprocessing (Weight BFP skew matrix, other Input BFP skew matrix) 만들기
def WS_preprocessing(matrix_1, matrix_2, dim_col, dim_row, strides, group_size, mode):
    #num_data1 = matrix_1.shape[0] num_h1 = matrix_1.shape[2] num_w1 = matrix_1.shape[3] num_data2 = matrix_2.shape[0] num_w2= matrix_2.shape[3]  num_ch12 = matrix_1.shape[1]
    num_h2= matrix_2.shape[2]
    if (matrix_1.shape[1] != matrix_2.shape[1]):
        print("error : not same num_channel of two matrix")
    ####    padding on matrix1
    pad_matrix_1 = padding_matrix(num_h2, matrix_1)
    ####    Im2col
    Im2col_matrix_1, Im2col_matrix_2 = Im2col(pad_matrix_1, matrix_2, matrix_1.shape, mode) #.shape method사용하면 ( , , )나옴 [1]int 형으로 값쓸수 있다.
    ####    BFP precision (2 or 4)
    bfp_precision_1 = TEST_BFP_precision(matrix_1)
    bfp_precision_2 = TEST_BFP_precision(matrix_2)
    ####    change Im2col(fp) --> Im2col(bfp)
    bfp_Im2col_matrix_1 = Im2col_fp_to_bfp(Im2col_matrix_1, bfp_precision_1, bfp_precision_1, group_size)
    bfp_Im2col_matrix_2 = Im2col_fp_to_bfp(Im2col_matrix_2, bfp_precision_2, bfp_precision_2, group_size)
    if(mode == "Forward_WS"):
        direction = "col_direction"     #### Input data가 fold되는 기준이 col (col 0 ~ col n으로 들어감)
    elif(mode =="Backward_WS"):
        direction = "row_direction"     #### Input data가 fold되는 기준이 row (row 0 ~ row n으로 들어감)
    else:
        print("error: WS_preprocessing _ mode error")
    Input_1 = SKEW_matrix(bfp_Im2col_matrix_1, dim_row, dim_col, direction)
    Input_2 = bfp_Im2col_matrix_2

    return Input_1, Input_2

############################################################################################################################
############################################################################################################################
####    OS preprocessing (front Input BFP skew matrix, back Input BFP skew matrix) 만들기
def OS_preprocessing(matrix_1, matrix_2, dim_col, dim_row, strides, group_size):
    # num_data1 = matrix_1.shape[0] num_h1 = matrix_1.shape[2] num_w1 = matrix_1.shape[3] num_data2 = matrix_2.shape[0] num_w2= matrix_2.shape[3] num_ch12 = matrix_1.shape[1]
    num_h2= matrix_2.shape[2]
    if (matrix_1.shape[1] != matrix_2.shape[1]):
        print("error nor same num_channel")
    ####    padding on matrix1
    pad_matrix_1 = padding_matrix(num_h2, matrix_1)
    ####    Im2col
    Im2col_matrix_1, Im2col_matrix_2 = Im2col(pad_matrix_1, matrix_2, matrix_1.shape, "Backward_OS")
    ####    BFP precision (2 or 4)
    bfp_precision_1 = TEST_BFP_precision(matrix_1)
    bfp_precision_2 = TEST_BFP_precision(matrix_2)
    ####    change Im2col(fp) --> Im2col(bfp)
    bfp_Im2col_matrix_1 = Im2col_fp_to_bfp(Im2col_matrix_1, bfp_precision_1, bfp_precision_1, group_size)
    bfp_Im2col_matrix_2 = Im2col_fp_to_bfp(Im2col_matrix_2, bfp_precision_2, bfp_precision_2, group_size)
    Input_1 = SKEW_matrix(bfp_Im2col_matrix_1, dim_row, dim_col, "row_direction")  #==>  ㅅ
    Input_2 = SKEW_matrix(bfp_Im2col_matrix_2, dim_row, dim_col, "col_direction")      # ||
    
    return Input_1, Input_2

############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
def Im2col_to_4Dmatrix(input, outputshape):
    ch = outputshape[1]
    da = outputshape[0]
    he = outputshape[2]
    wi = outputshape[3]
    output = np.full(outputshape,0)
    for c in range(ch):
        for d in range(da):
            for h in range(he):
                for w in range(wi):
                    output[d][c][h][w] = input[d*he*wi + h*wi + w][c]
    return output









if __name__ == "__main__":
    mode = input()#
    if(mode != "Backward_OS" and mode != "Forward_WS" and mode != "Backward_WS"):
        print("input should be 'Backward_OS' or 'Forward_WS' or 'Backward_WS'")
        quit()
    print("FMAC simulator start with mode" , mode, ".")
    dim_col = 4
    dim_row = 4
    strides = 1
    group_size = 16
    num_of_exp_plus_man = 1+ group_size
    print("FMAC Systolic Array spec: dim_col=", dim_col, "dim_row=", dim_row, "group_size=",group_size)
    print("assume: stride is 1\n")
    if(mode != "Backward_OS"):
        num_data1 = 2
        num_h1 = 7
        num_w1 = 7
        num_data2 = 1
        num_h2=3
        num_w2=3 
        num_ch12 = 1 #common
        print("1st operand=>","(",num_data1," * ",num_ch12," * ", num_h1," * ",num_h2,")" )
        print("2nd operand=>","(",num_data2," * ",num_ch12," * ", num_h2," * ",num_h2,")\n\n" )
    elif(mode == "Backward_OS"):
        num_data1 = 2
        num_h1 = 7
        num_w1 = 7
        num_data2 = 1
        num_h2=7
        num_w2=7 
        num_ch12 = 1 #common
        W_height = 3
        W_width = 3
        print("1st operand=>","(",num_data1," * ",num_ch12," * ", num_h1," * ",num_h2,")" )
        print("2nd operand=>","(",num_data2," * ",num_ch12," * ", num_h2," * ",num_h2,")" )
        print("output G_weight is size of (",num_data1," * ",num_data2," * ", W_height," * ",W_width,")\n\n")
##########################   start of preprocessing   ###############################
    ####    Test for Forward_WS (matrix1 * matrix2)     
####    TEST START
    print("STEP: preprocessing")

    matrix_1 = np.full((num_data1,num_ch12,num_h1,num_w1),1)
    # list_1 = [[[[1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21],[22,23,24,25,26,27,28],[29,30,31,32,33,34,35],[36,37,38,39,40,41,42],[43,44,45,46,47,48,49]]],[[[1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21],[22,23,24,25,26,27,28],[29,30,31,32,33,34,35],[36,37,38,39,40,41,42],[43,44,45,46,47,48,49]]]]
    # matrix_1 = np.array(list_1)

    if(mode == "Backward_OS"):
        matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),1)
    elif(mode == "Forward_WS" or mode == "Backward_WS"):
        matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),1)
        # list_2 = [[[[1,2,3],[4,5,6],[7,8,9]]]]
        # matrix_2 = np.array(list_2)

####    TEST A :Padding matrix
    if(mode == "Forward_WS" or mode == "Backward_WS"):
        pad_matrix_1 = padding_matrix(num_h2, matrix_1)
    elif(mode == "Backward_OS"):
        pad_matrix_1 = padding_matrix(W_height, matrix_1)
    print("TEST A : padding matrix of matrix_1")
    print("\t matrix_1 size change from ", matrix_1.shape, "to ", pad_matrix_1.shape,"with padding")
    print("\t matrix_2 size is ", matrix_2.shape,"\n")
    
####    TEST B :Im2col _WS, Im2col _OS
    print("TEST B : Im2col of pad_matrix_1, matrix_2")
    if(mode == "Forward_WS" or mode == "Backward_WS"):
        Im2col_matrix_1_WS, Im2col_matrix_2_WS = Im2col(pad_matrix_1, matrix_2, matrix_1.shape, mode, None, None)
        print("\t size of Im2col_matrix_1: ", Im2col_matrix_1_WS.shape," size of Im2col_matrix_2: " ,Im2col_matrix_2_WS.shape)
    elif(mode == "Backward_OS"):
        Im2col_matrix_1_OS, Im2col_matrix_2_OS = Im2col(pad_matrix_1, matrix_2, matrix_1.shape, "Backward_OS",W_height, W_width )
        print("\t size of Im2col_matrix_1: ", Im2col_matrix_1_OS.shape," size of Im2col_matrix_2: " ,Im2col_matrix_2_OS.shape)

    ####    TEST D(precision)
    ####    TEST E(fp-->bfp_group matrix)
    ####    TEST F(bfp_group matrix --> skewed_bfp_group matrix)
####    TEST END
     
##########################   end of preprocessing   ###############################


################################   START OF SYSTOLIC ARRAY TEST #########################################
    # FOR TEST We assume num_of_exp_plus_man == 1, instead of 'group_size(16)+1' == 17
    #fp형식->bfp모양에 맞게 넣는다 (시뮬레이션 돌리기 위해) (BFP변환은 하지 않은 상태) fp값을 단지 exp위치에 넣어줌 mantissa는 다 1
    #################################################
    print("\n\n\n")
    print("data go through Systolic Array")
    if(mode == "Forward_WS" or mode == "Backward_WS"):
        r2c = num_h2 * num_w2 * num_ch12
        matrix_1_h = num_data1 * num_h1 * num_w1
        matrix_1_w = r2c
        matrix_1 = np.full((matrix_1_h, matrix_1_w, num_of_exp_plus_man),1)
        for i in range (matrix_1_h):
            for j in range (matrix_1_w):
                matrix_1[i][j][0]=Im2col_matrix_1_WS[i][j]
        np.savetxt('./Im2col_1', Im2col_matrix_1_WS, fmt='%i', delimiter=',\t')
    elif(mode == "Backward_OS"):
        matrix_1_h = num_ch12 * num_h1 * num_w1
        matrix_1_w = num_data1 * W_height * W_width
        matrix_1 = np.full((matrix_1_h, matrix_1_w, num_of_exp_plus_man),1)
        for i in range (matrix_1_h):
            for j in range (matrix_1_w):
                matrix_1[i][j][0]=Im2col_matrix_1_OS[i][j]
        np.savetxt('./Im2col_1', Im2col_matrix_1_OS, fmt='%i', delimiter=',\t')
    #################################################################
    if(mode == "Backward_OS"):
        matrix_2_h = num_ch12 * num_h2 * num_w2
        matrix_2_w = num_data2 
        matrix_2 = np.full((matrix_2_h,matrix_2_w, num_of_exp_plus_man),1)
        for i in range (matrix_2_h):    
            for j in range (matrix_2_w):      
                matrix_2[i][j][0]=Im2col_matrix_2_OS[i][j]
        np.savetxt('./Im2col_2', Im2col_matrix_2_OS, fmt='%i', delimiter=',\t')
    else:
        if(mode == "Forward_WS"):
            matrix_2_h = num_data2
            matrix_2_w = r2c
        elif(mode == "Backward_WS"):
            matrix_2_h = r2c
            matrix_2_w = num_data2
        matrix_2 = np.full((matrix_2_h,matrix_2_w, num_of_exp_plus_man),1)
        for i in range (matrix_2_h):    
            for j in range (matrix_2_w):      
                matrix_2[i][j][0]=Im2col_matrix_2_WS[i][j]
        np.savetxt('./Im2col_2', Im2col_matrix_2_WS, fmt='%i', delimiter=',\t')
    
    
    ##CASE : OS
    # r2c = W_height * W_weight * num_ch12
    # matrix_1 = np.full((num_data1 * num_h1 * num_w1, r2c, num_of_exp_plus_man),1)
    # for i in range (num_data1 * num_h1 * num_w1):
    #     for j in range (r2c):
    #         matrix_1[i][j][0]=Im2col_matrix_1_WS[i][j]
    # np.savetxt('./Im2col_1', Im2col_matrix_1_WS, fmt='%i', delimiter=',\t')


    #Sample = Systolic_array(dim_col, dim_row, fMAC, group_size, "Forward_WS", matrix_1, matrix_2, None, None, r2c, num_ch12, num_data1 * num_h1 * num_w1 + dim_row + dim_col)
    #Sample = Systolic_array(dim_col, dim_row, fMAC, group_size, "Forward_WS", matrix_1, matrix_2, None, None, r2c, num_ch12, num_data1 * num_h1 * num_w1)

    ####_______________________________________________________________________________________________________________________________________  
    ####    Functions.py TEST1 (update_fold_variable_F_WS)
    ####    Check fold, use value.!

    # for i in range(Sample.col_fold_end*Sample.row_fold_end):
    #     print(Sample.col_fold_current, Sample.row_fold_current)
    #     print(Sample.col_use, Sample.row_use)
    #     print()
    #     Sample.update_fold_variable_B_WS()
    ####_______________________________________________________________________________________________________________________________________


    ####_______________________________________________________________________________________________________________________________________
    ####    Functions.py TEST2 (function: Sample.separate_exp_man(one_group))
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
    # array_W = np.full((Sample.dim_row, Sample.dim_col),0)
    # Sample.Systolic_preload_Weight()    #1234
    # Sample.update_fold_variable_F_WS()
    # Sample.Systolic_preload_Weight()    #5678
    # Sample.update_fold_variable_F_WS()
    # Sample.Systolic_preload_Weight()    #9000
    # for row in range(Sample.dim_row):
    #         for col in range(Sample.dim_col):
    #             array_W[row][col] = Sample.Unit_array[row][col].weight_exponant
    # print(array_W)



####    case 1 : Forward_WS
    if(mode == "Forward_WS"):
        Sample = Systolic_array(dim_col, dim_row, fMAC, group_size, "Forward_WS", matrix_1, matrix_2, None, None, r2c, num_data2, num_data1 * num_h1 * num_w1)
        array_W = np.full((Sample.dim_row, Sample.dim_col),0)
        Sample.update_colrow_use_variable()

        for i in tqdm(range(Sample.col_fold_end*Sample.row_fold_end)):
            print("fold_count",i,"= col_fold",Sample.col_fold_current, "row_fold" , Sample.row_fold_current)
            Sample.Systolic_preload_Weight()
            for row in range(Sample.dim_row):
                for col in range(Sample.dim_col):
                    array_W[row][col] = Sample.Unit_array[row][col].weight_exponant
            Sample.Systolic_Forward_WS_one_folds()
            Sample.make_square_Forward_WS_result()
            Sample.store_Forward_WS_result()
            Sample.update_fold_variable_F_WS()

####    case 2 : Backward_WS    
    elif(mode == "Backward_WS"):                                                                                   #col fold  #row fold 
        Sample = Systolic_array(dim_col, dim_row, fMAC, group_size, "Backward_WS",  None, matrix_2,matrix_1, None, num_ch12, num_data2 * num_h2 * num_w2, num_data1 * num_h1 * num_w1)
        array = np.full((Sample.dim_row, Sample.dim_col),0)           
        Sample.update_colrow_use_variable()
        
        for i in tqdm(range(Sample.col_fold_end*Sample.row_fold_end)):
            print("fold_count",i,"= col_fold",Sample.col_fold_current, "row_fold" , Sample.row_fold_current)
            Sample.Systolic_preload_Weight()
            for row in range(Sample.dim_row):
                for col in range(Sample.dim_col):
                    array[row][col] = Sample.Unit_array[row][col].weight_exponant
            Sample.Systolic_Backward_WS_one_folds()
            Sample.make_square_Backward_WS_result()
            Sample.store_Backward_WS_result()
            Sample.update_fold_variable_B_WS()

####    case 3 : Backward_OS 
    elif(mode == "Backward_OS"):
        Sample = Systolic_array(dim_col, dim_row, fMAC, group_size, "Backward_OS",  matrix_1, None, matrix_2, (W_height,W_width), num_data1*W_height*W_width, num_data2 , None)
        array = np.full((Sample.dim_row, Sample.dim_col),0) 
        Sample.update_colrow_use_variable()
        
        for i in tqdm(range(Sample.col_fold_end*Sample.row_fold_end)):
            print("fold_count",i,"= col_fold",Sample.col_fold_current, "row_fold" , Sample.row_fold_current)
            Sample.Systolic_Backward_OS_one_folds()

            array = np.full((Sample.dim_row, Sample.dim_col),0) 
            for row in range(Sample.dim_row):
                for col in range(Sample.dim_col):
                    array[row][col] = Sample.Unit_array[row][col].accumulator
            #print(array)
            Sample.store_OS_result()
            Sample.update_fold_variable_F_WS()
    else:
        print("error")


    print("return_result")
    print(Sample.return_result)
    print(Sample.return_result.shape)
    print("\n")
    print("Total Cycle: " , Sample.cycle,"cycles \n\n")
    if(mode =="Backward_OS"):
        outputshape = (num_data1,num_data2,W_height,W_width)
    else:
        outputshape = (num_data1,num_data2,num_h1,num_w1)

    result = Im2col_to_4Dmatrix(Sample.return_result, outputshape)
    print("result of caculation") 
    print(result)









