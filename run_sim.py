import numpy as np
from PE_Array import *
from tqdm import tqdm
from preprocess import *

if __name__ == "__main__":
    mode = input()
    if(mode != "Backward_OS" and mode != "Forward_WS" and mode != "Backward_WS"):
        print("input should be 'Backward_OS' or 'Forward_WS' or 'Backward_WS'")
        quit()
    print("FMAC simulator start with mode" , mode, ".")
######## Start of hard_setting #########################################################################################
    dim_col = 4
    dim_row = 4
    strides = 1
    group_size = 16
    num_of_exp_plus_man = 1+ group_size
    mantissa_bit = 8 # "11.01"
    round1 = int(mantissa_bit/2)
    round2 = int(mantissa_bit/2)
    print("FMAC Systolic Array spec: dim_col=", dim_col, "dim_row=", dim_row, "group_size=",group_size)
    print("assume: stride is 1\n")
    ############  data_size_table  #############
    aC = 2
    aB = 2
    aN = 2
    d_H = 14
    d_W = 14
    w_H = 7
    w_W = 7
    ############################################
    if(mode != "Backward_OS"):
        num_data1 = aB
        num_h1 = d_H
        num_w1 = d_W
        num_data2 = aN
        num_h2 = w_H
        num_w2 = w_W 
        num_ch12 = aC #common
        result_size = np.zeros(2)
        print("1st operand=>","(",num_data1," * ",num_ch12," * ", num_h1," * ",num_h2,")" )
        print("2nd operand=>","(",num_data2," * ",num_ch12," * ", num_h2," * ",num_h2,")\n\n" )
    elif(mode == "Backward_OS"):
        num_data1 = aC
        num_h1 = d_H
        num_w1 = d_W
        num_data2 = aN
        num_h2 = d_H
        num_w2 = d_W 
        num_ch12 = aB #common
        W_height = w_H
        W_width = w_W
        result_size = np.array([W_width, W_height])
        print("1st operand=>","(",num_data1," * ",num_ch12," * ", num_h1," * ",num_h2,")" )
        print("2nd operand=>","(",num_data2," * ",num_ch12," * ", num_h2," * ",num_h2,")" )
        print("output G_weight is size of (",num_data1," * ",num_data2," * ", W_height," * ",W_width,")\n\n")
######## End of hard_setting #########################################################################################




##########################   start of preprocessing   ###############################
    print("STEP: preprocessing")
    preprocessor = preprocess()
    # MAKE matrix_1, matrix_2
    matrix_1 = np.full((num_data1,num_ch12,num_h1,num_w1),1)
    if(mode == "Backward_OS"):
        matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),1)
    elif(mode == "Forward_WS" or mode == "Backward_WS"):
        matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),1)
    
    # preprocessing
    if(mode == "Forward_WS" or mode == "Backward_WS"):
        Im2col_matrix_1_WS, Im2col_matrix_2_WS = preprocessor.preprocessing(matrix_1, matrix_2, mode, result_size)
    elif(mode == "Backward_OS"):
        Im2col_matrix_1_OS, Im2col_matrix_2_OS = preprocessor.preprocessing(matrix_1, matrix_2, mode, result_size)
##########################   end of preprocessing   ###############################






##########################   start of data_generator   ###############################
    ####    TEST D(precision)
    # precision은 고정값으로 한다. (group_size = 16 ,bfp_M_Bit = 8)

    ####    TEST E(fp-->bfp_group matrix)
    
    
    # set_reproducibility(1)
    # BfpConfig.bfp_M_Bit = 8
    # fp_input = torch.randn(("R", "C"), dtype=torch.float32)
    # wrapper = BfpWrapper(fp_tensor=fp_input, should_sort=False)
    # wrapper.run_convert_fp_to_bfp()
    # =>  wrapper.np_bfp_S, wrapper.np_bfp_E, wrapper.np_bfp_M  ["R" * "C"]에서 값 추출.
    
    ### cpy_output = wrapper.get_fp_from_current_bfp_info()

    ####    TEST F(bfp_group matrix --> skewed_bfp_group matrix)
    #"BPF_import" 2
    # 아래 코드는 그냥 실험을 위해서 size만 정하고 다1로 채움)
    # 아래 코드 기반으로 각 element에다가 값을 집어 넣는다.

     

    # FOR TEST We assume num_of_exp_plus_man == 1, instead of 'group_size(16)+1' == 17
    #fp형식->bfp모양에 맞게 넣는다 (시뮬레이션 돌리기 위해) (BFP변환은 하지 않은 상태) fp값을 단지 exp위치에 넣어줌 mantissa는 다 1
   
# data_generator 구현전에 bfp contaner로 hard coding한것 
# MAKE matrix_1, matrix_2
    #### Input 1. ####
    print("\n\n\n")
    print("data go through Systolic Array")
    if(mode == "Backward_OS"):
        #"BPF_import" 3 width는 group_size떄문에 작아지므로 조정해주어야 하고 4경우 모두
        matrix_1_h = num_ch12 * num_h1 * num_w1
        matrix_1_w = num_data1 * W_height * W_width
        matrix_1 = np.full((round1, matrix_1_h, matrix_1_w, num_of_exp_plus_man),0)
        #"BPF_import" mantissa넣는 칸도 채워줘야 한다. 4경우 모두
        for r in range(round1):
            for i in range (matrix_1_h):
                for j in range (matrix_1_w):
                    matrix_1[r][i][j][0]=Im2col_matrix_1_OS[i][j]
        print("size of input1 matrix = (", matrix_1.shape, ")")
    elif(mode == "Forward_WS" or mode == "Backward_WS"):
        r2c = num_h2 * num_w2 * num_ch12
        matrix_1_h = num_data1 * num_h1 * num_w1
        matrix_1_w = r2c
        matrix_1 = np.full((round1, matrix_1_h, matrix_1_w, num_of_exp_plus_man),0)
        for r in range(round1):
            for i in range (matrix_1_h):
                for j in range (matrix_1_w):
                    matrix_1[r][i][j][0]=Im2col_matrix_1_WS[i][j]
        print("size of input1 matrix = (", matrix_1.shape, ")")
    #### input2. ####
    if(mode == "Backward_OS"):
        matrix_2_h = num_ch12 * num_h2 * num_w2
        matrix_2_w = num_data2 
        matrix_2 = np.full((round2, matrix_2_h,matrix_2_w, num_of_exp_plus_man),0)
        for r in range(round2):
            for i in range (matrix_2_h):    
                for j in range (matrix_2_w):      
                    matrix_2[r][i][j][0]=Im2col_matrix_2_OS[i][j]
        print("size of input2 matrix = (", matrix_2.shape, ")")
    else:
        if(mode == "Forward_WS"):
            matrix_2_h = num_data2
            matrix_2_w = r2c
        elif(mode == "Backward_WS"):
            matrix_2_h = r2c
            matrix_2_w = num_data2
        matrix_2 = np.full((round2, matrix_2_h,matrix_2_w, num_of_exp_plus_man),0)
        for r in range(round2):
            for i in range (matrix_2_h):    
                for j in range (matrix_2_w):      
                    matrix_2[r][i][j][0]=Im2col_matrix_2_WS[i][j]
        print("size of input2 matrix = (", matrix_2.shape, ")")
##########################   end of data_generator   ###############################







################################   START OF SYSTOLIC ARRAY #########################################
# input : matrix_1, matrix_2
####    case 1 : Forward_WS
    if(mode == "Forward_WS"):
        Sample = PE_Array(dim_col, dim_row, PE_Unit, group_size, "Forward_WS", matrix_1, matrix_2, None, None, r2c, num_data2, num_data1 * num_h1 * num_w1, round1, round2)
        Sample.Forward_WS_Calculation()
####    case 2 : Backward_WS    
    elif(mode == "Backward_WS"):                                                                                   #col fold  #row fold 
        Sample = PE_Array(dim_col, dim_row, PE_Unit, group_size, "Backward_WS",  None, matrix_2,matrix_1, None, num_ch12, num_data2 * num_h2 * num_w2, num_data1 * num_h1 * num_w1, round1, round2)          
        Sample.Backward_WS_Calculation()
####    case 3 : Backward_OS 
    elif(mode == "Backward_OS"):
        Sample = PE_Array(dim_col, dim_row, PE_Unit, group_size, "Backward_OS",  matrix_1, None, matrix_2, (W_height,W_width), num_data1*W_height*W_width, num_data2 , None, round1, round2)
        Sample.Backward_OS_Calculation()
    else:
        print("error")
################################   End OF SYSTOLIC ARRAY  #########################################



################################   START OF Getting Result   #########################################

    #% return_result(2D) ==> result(4D)
    # print("return_result")
    # print(Sample.return_result)
    # print(Sample.return_result.shape)
    # print("\n")

    print("Total Cycle: " , Sample.cycle,"cycles \n\n")

    if(mode =="Backward_OS"):
        outputshape = (num_data1,num_data2,W_height,W_width)
    else:
        outputshape = (num_data1,num_data2,num_h1,num_w1)

    result = preprocessor.Im2col_to_4Dmatrix(Sample.return_result, outputshape)
    Sample.clean_return_result()
    print("result of caculation") 
    print(result)
################################   End OF Getting Result   #########################################
