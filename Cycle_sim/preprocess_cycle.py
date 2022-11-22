import numpy as np
import math
from PE_Array_cycle import *
from tqdm import tqdm

neg_inf = -1 * math.pow(2,32)
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#  fp 4D -> padding -> fp 2D
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
class preprocess():
    def __init__(self):
        a=1


    """+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    ####    FP MATRIX
    ####    Input = matrix                  dimension, (B = 1, num_channels, ifmap_height, ifmap_width)
    ####    Output = matrix(with padding)   dimension, (B = 1, num_channels, ifmap_height + 2f, ifmap_width + 2f) 
    ####    (  f = padding_size = (filtersize -1)/2  )
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-"""
    def padding_matrix(self, filter_size, matrix): ##4차원
        ####    right_padding_size > left_padding_size (if filter_size is ever)
        self.filter_size = filter_size
        self.pad = math.ceil((filter_size -1)/2)
        padding_width_right = math.ceil((filter_size -1)/2)
        padding_width_left = math.floor((filter_size -1)/2)

        padding_matrix = np.pad(matrix, ((0,0),(0,0),(padding_width_right,padding_width_right),(padding_width_right,padding_width_right)), 'constant', constant_values = 0.)
        return padding_matrix
    
    """+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    ####    Im2col
    ####    Input: pad_matrix_1, matrix_2, amtrix_1_shape  #### type: numpy_array(4 Dimension), numpy_array(4 Dimension), tuple(int,int,int,int)
    ####    Output: Im2col_matrix_1 (num_data_1*num_h_1*num_w_1, num_ch*num_h_2*num_w_2)
    ####            Im2col_matrix_2 (num_data_2, num_ch*num_h_2*num_w_2) or (num_ch*num_h_2*num_w_2, num_data_2)
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-"""
    def Im2col(self, pad_matrix_1, matrix_2, matrix_1_shape, mode, OS_result_h, OS_result_w): 
        num_data_2 = matrix_2.shape[0]
        num_ch = matrix_1_shape[1]
        num_h_2 = matrix_2.shape[2]
        num_w_2 = matrix_2.shape[3]

        num_data_1 = matrix_1_shape[0]
        num_h_1 = matrix_1_shape[2]
        num_w_1 = matrix_1_shape[3]

        ## Im2col_matrix_1
        if(mode == "Forward_WS" or mode =="Backward_WS"):
            new_h = math.floor((num_h_1 + self.pad*2 - self.filter_size) /self.stride) +1
            new_w = math.floor((num_w_1+ self.pad*2 - self.filter_size)/self.stride) +1
            Im2col_matrix_1 = np.full((num_data_1*new_h*new_w, num_ch*num_h_2*num_w_2), 0.)   
            for b in range(num_data_1):
                for h in range(new_h):
                    for w in range(new_w):
                        one_px = pad_matrix_1[b, 0:num_ch, h*self.stride:h*self.stride+num_h_2, w*self.stride:w*self.stride+num_w_2]
                        #print(one_px)
                        px_num = new_h * new_w * b + new_w * h + w
                        # put 4D matrix values in to Im2col matrix in one row
                        for i in range(num_ch):
                            for j in range(num_h_2):
                                for k in range(num_w_2):
                                    Im2col_matrix_1[px_num][num_w_2*num_h_2*i + num_w_2*j + k]= one_px[i][j][k]
        elif(mode == "Backward_OS"):
            Im2col_matrix_1 = np.full((num_ch*num_h_1*num_w_1, num_data_1* (OS_result_h//self.stride+1)*(OS_result_w//self.stride+1)), 0.) 
            for b in range(num_data_1):
                for h in range((OS_result_h//self.stride+1)):
                    for w in range((OS_result_w//self.stride+1)):
                        one_px = pad_matrix_1[b, 0:num_ch, h*self.stride:h*self.stride+num_h_2, w*self.stride:w*self.stride+num_w_2]
                        #print(one_px)
                        px_num = (OS_result_h//self.stride+1)*(OS_result_w//self.stride+1)*b + (OS_result_w//self.stride+1)*h + w
                        # put 4D matrix values in to Im2col matrix in one row
                        for i in range(num_ch):
                            for j in range(num_h_2):
                                for k in range(num_w_2):
                                    Im2col_matrix_1[num_w_2*num_h_2*i + num_w_2*j + k][px_num]= one_px[i][j][k]



        #Im2col_matrix_1[num_data_1*num_h_1*num_w_1][:] = np.full((1,num_ch*num_h_2*num_w_2), neg_inf) ## end point
        if(mode == "Forward_WS"):
            Im2col_matrix_2 = np.full((num_data_2, num_ch*num_h_2*num_w_2),0.)
            for n in range(num_data_2):
                for c in range(num_ch):
                    for h in range(num_h_2):
                        for w in range(num_w_2):
                            Im2col_matrix_2[n][num_h_2*num_w_2*c + num_w_2*h + w] = matrix_2[n][c][h][w]
        elif(mode=="Backward_WS"):
            Im2col_matrix_2 = np.full((num_ch*num_h_2*num_w_2, num_data_2), 0.)
            for n in range(num_data_2):
                for c in range(num_ch):
                    for h in range(num_h_2):
                        for w in range(num_w_2):
                            Im2col_matrix_2[num_h_2*num_w_2*c + num_w_2*h + w][n] = matrix_2[n][c][h][w]
        elif(mode=="Backward_OS"):
            Im2col_matrix_2 = np.full((num_ch*num_h_2*num_w_2, num_data_2), 0.)
            for n in range(num_data_2):
                for c in range(num_ch):
                    for h in range(num_h_2):
                        for w in range(num_w_2):
                            Im2col_matrix_2[num_h_2*num_w_2*c + num_w_2*h + w][n] = matrix_2[n][c][h][w]



        else:
            print("error: in Im2col function mode is not found")

        return Im2col_matrix_1, Im2col_matrix_2


    
    def preprocessing(self, matrix_1, matrix_2, mode, result_size, stride):
        self.stride = stride
        if(result_size[1]==0 and result_size[0]==0 and mode == "Backward_OS"):
            print("preprocessing: Output stationary, but no required result_size")
        if(matrix_1.shape[1] != matrix_2.shape[1]):
            print("error nor same num_channel")
        ####    padding on matrix1
        if(mode=="Backward_OS"):
            filter_size = result_size[1]
        else:
            filter_size = matrix_2.shape[2]
        pad_matrix_1 = self.padding_matrix(filter_size, matrix_1)
        print("TEST A : padding matrix of matrix_1")
        print("\t matrix_1 size change from ", matrix_1.shape, "to ", pad_matrix_1.shape,"with padding")
        print("\t matrix_2 size is ", matrix_2.shape,"\n")
        ####    Im2col
        Im2col_matrix_1, Im2col_matrix_2 = self.Im2col(pad_matrix_1, matrix_2, matrix_1.shape, mode, result_size[0], result_size[1])
        print("TEST B : Im2col of pad_matrix_1, matrix_2")
        print("\t size of Im2col_matrix_1: ", Im2col_matrix_1.shape," size of Im2col_matrix_2: " ,Im2col_matrix_2.shape)
        #np.savetxt('./Im2col_1', Im2col_matrix_1, fmt='%i', delimiter=',\t')
        #np.savetxt('./Im2col_2', Im2col_matrix_2, fmt='%i', delimiter=',\t')
        return Im2col_matrix_1, Im2col_matrix_2
    
    def Im2col_to_4Dmatrix(self, input, outputshape):
        ch = outputshape[1]
        da = outputshape[0]
        he = outputshape[2]
        wi = outputshape[3]
        output = np.full(outputshape,0.)
        print(output.shape)
        print(input.shape)
        for c in range(ch):
            for d in range(da):
                for h in range(he):
                    for w in range(wi):
                        output[d][c][h][w] = input[d*he*wi + h*wi + w][c]
        return output













if __name__ == "__main__":
    mode = input()
    if(mode != "Backward_OS" and mode != "Forward_WS" and mode != "Backward_WS"):
        print("input should be 'Backward_OS' or 'Forward_WS' or 'Backward_WS'")
        quit()
    print("FMAC simulator start with mode" , mode, ".")
######## hard_setting #########################################################################################
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
        result_size = np.zeros(2)
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
        result_size = np.array([W_width, W_height])
        print("1st operand=>","(",num_data1," * ",num_ch12," * ", num_h1," * ",num_h2,")" )
        print("2nd operand=>","(",num_data2," * ",num_ch12," * ", num_h2," * ",num_h2,")" )
        print("output G_weight is size of (",num_data1," * ",num_data2," * ", W_height," * ",W_width,")\n\n")
##########################   start of preprocessing   ###############################
    ####    Test for Forward_WS (matrix1 * matrix2)     
####    TEST START
    ##$$$$$$$$$$$$$$
    # pre-processing이 할 일
    # 
    # $$$$$$$$$$$$$$ 
    print("STEP: preprocessing")
    preprocessor = preprocess()
    matrix_1 = np.full((num_data1,num_ch12,num_h1,num_w1),1)
    # list_1 = [[[[1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21],[22,23,24,25,26,27,28],[29,30,31,32,33,34,35],[36,37,38,39,40,41,42],[43,44,45,46,47,48,49]]],[[[1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21],[22,23,24,25,26,27,28],[29,30,31,32,33,34,35],[36,37,38,39,40,41,42],[43,44,45,46,47,48,49]]]]
    # matrix_1 = np.array(list_1)
    #"BPF_import" 1  1이 아니라 임의의 숫자넣어야함
    if(mode == "Backward_OS"):
        matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),1)
    elif(mode == "Forward_WS" or mode == "Backward_WS"):
        matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),1)


    if(mode == "Forward_WS" or mode == "Backward_WS"):
        Im2col_matrix_1_WS, Im2col_matrix_2_WS = preprocessor.preprocessing(matrix_1, matrix_2, mode, result_size)
    elif(mode == "Backward_OS"):
        Im2col_matrix_1_OS, Im2col_matrix_2_OS = preprocessor.preprocessing(matrix_1, matrix_2, mode, result_size)

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
####    TEST END
     
##########################   end of preprocessing   ###############################