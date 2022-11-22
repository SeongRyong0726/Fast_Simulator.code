import numpy as np
from PE_Array_cycle import *
from tqdm import tqdm
from preprocess_cycle import *
from data_generator_cycle import *
from Simulator_Config import BfpConfig
import glob

# def im2col(X, filters, stride=1, pad=0):
#     n, c, h, w = X.shape
#     n_f, _, filter_h, filter_w = filters.shape

#     out_h = (h + 2 * pad - filter_h) // stride + 1
#     out_w = (w + 2 * pad - filter_w) // stride + 1

#     # add padding to height and width.
#     in_X = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values =0)
#     out = np.zeros((n, c, filter_h, filter_w, out_h, out_w))

#     for h in range(filter_h):
#         h_end = h + stride * out_h
#         for w in range(filter_w):
#             w_end = w + stride * out_w
#             out[:, :, h, w, :, :] = in_X[:, :, h:h_end:stride, w:w_end:stride]

#     out = out.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
#     return out


if __name__ == "__main__":
    # mode = input()
    # if(mode != "Backward_OS" and mode != "Forward_WS" and mode != "Backward_WS"):
    #     print("input should be 'Backward_OS' or 'Forward_WS' or 'Backward_WS'")
    #     quit()
    # print("FMAC simulator start with mode" , mode, ".")

    dim_col = BfpConfig.dim_col
    dim_row = BfpConfig.dim_row
    stride = 1#BfpConfig.strides
    group_size = BfpConfig.group_size
    num_of_exp_plus_man = 1+ group_size
    mantissa_bit = BfpConfig.mantissa_bit
    
    print("FMAC Systolic Array spec: dim_col=", dim_col, "dim_row=", dim_row, "group_size=",group_size)
    print("assume: stride is 1\n")
    ############  data_size_table  #############
    aC = 10
    aB = 15
    aN = 5
    d_H = 100
    d_W = 100
    w_H = 7
    w_W = 7
    ############################################
    ##For Cycle Simulator
    glob_list = sorted(glob.glob('/home/sroh/cycle-sim-data/resnet50-cifar100-b128/*.npy'))
    f = open('/home/sroh/sroh-backup-files/SR_sim/Cycle_sim_fp/bwd-a-result.txt', 'w')
    f.write("dim_col="+ str(dim_col) +  "dim_row=" + str(dim_row)+ "group_size="+ str(group_size)+"\n")
    # for sd in glob_list:
    #     print(sd)
    cycle_result = np.full(((155+156+156),5), 0)
    kkk = 54
    aaa = 53
    bbb = 53+54
    ccc = 53+54+54
    for number in range(kkk):  #155+156+156
        #TEST RESNET-50
        num_num = number+53+54
        print("======================================================================")
        print("START")
        print("dim_col=", dim_col, "dim_row=", dim_row, "group_size=",group_size)
        print("lhs:" ,glob_list[2*num_num])
        print("rhs:" ,glob_list[2*num_num+1])
        f.write("dim_col="+ str(dim_col) +  "dim_row=" + str(dim_row)+ "group_size="+ str(group_size)+"\n")
        f.write(str(number) + "\n")
        f.write("lhs:" + str(glob_list[2*num_num]) + "\n")
        f.write("rhs:" + str(glob_list[2*num_num+1]) + "\n")
        if(num_num<aaa):#bwd_a 53
            Im2col_matrix_1_WS = np.load(glob_list[2*num_num])
            Im2col_matrix_2_WS = np.load(glob_list[2*num_num+1])
            Im2col_matrix_2_WS = Im2col_matrix_2_WS.T
            print("lhs shape:", Im2col_matrix_1_WS.shape)
            print("rhs shape:", Im2col_matrix_2_WS.shape)
            mode = "Backward_WS"
        elif(aaa <= num_num<(bbb)):#bwd_w 54
            Im2col_matrix_1_OS = np.load(glob_list[2*num_num])
            Im2col_matrix_2_OS = np.load(glob_list[2*num_num+1])
            Im2col_matrix_1_OS = Im2col_matrix_1_OS.T
            Im2col_matrix_2_OS = Im2col_matrix_2_OS.T
            print("lhs shape:", Im2col_matrix_1_OS.shape)
            print("rhs shape:", Im2col_matrix_2_OS.shape)
            mode = "Backward_OS"
        elif((bbb)<=num_num< ccc):#fwd 54
            Im2col_matrix_1_WS = np.load(glob_list[2*num_num])
            Im2col_matrix_2_WS = np.load(glob_list[2*num_num+1])
            print("lhs shape:", Im2col_matrix_1_WS.shape)
            print("rhs shape:", Im2col_matrix_2_WS.shape)
            mode = "Forward_WS"
        cycle_result[number][0] = number

    ######## Start of hard_setting #########################################################################################
        # if(mode == "Forward_WS"):
        #     num_data1 = aB
        #     num_h1 = d_H
        #     num_w1 = d_W
        #     num_data2 = aN
        #     num_h2 = w_H
        #     num_w2 = w_W 
        #     num_ch12 = aC #common
        #     result_size = np.zeros(2)
        #     print("1st operand=>","(",num_data1," * ",num_ch12," * ", num_h1," * ",num_h2,")" )
        #     print("2nd operand=>","(",num_data2," * ",num_ch12," * ", num_h2," * ",num_h2,")\n\n" )
        # elif(mode == "Backward_WS"):
        #     num_data1 = aB
        #     num_h1 = d_H
        #     num_w1 = d_W
        #     num_data2 = aC
        #     num_h2 = w_H
        #     num_w2 = w_W 
        #     num_ch12 = aN #common
        #     result_size = np.zeros(2)
        #     print("1st operand=>","(",num_data1," * ",num_ch12," * ", num_h1," * ",num_h2,")" )
        #     print("2nd operand=>","(",num_data2," * ",num_ch12," * ", num_h2," * ",num_h2,")\n\n" )
        # elif(mode == "Backward_OS"):
        #     num_data1 = aC
        #     num_h1 = d_H
        #     num_w1 = d_W
        #     num_data2 = aN
        #     num_h2 = d_H
        #     num_w2 = d_W 
        #     num_ch12 = aB #common
        #     W_height = w_H
        #     W_width = w_W
        #     result_size = np.array([W_width, W_height])
        #     print("1st operand=>","(",num_data1," * ",num_ch12," * ", num_h1," * ",num_h2,")" )
        #     print("2nd operand=>","(",num_data2," * ",num_ch12," * ", num_h2," * ",num_h2,")" )
        #     print("output G_weight is size of (",num_data1," * ",num_data2," * ", W_height," * ",W_width,")\n\n")

    ##########################   start of preprocessing   ###############################
        #print("STEP: preprocessing")
        #preprocessor = preprocess()
        # MAKE matrix_1, matrix_2 
        #matrix_1 = np.random.rand(num_data1,num_ch12,num_h1,num_w1)
        #matrix_2 = np.random.rand(num_data2,num_ch12,num_h2,num_w2)
        #matrix_1 = np.random.randint(1,10, size=(num_data1,num_ch12,num_h1,num_w1))
        #matrix_2 = np.random.randint(1,10, size=(num_data2,num_ch12,num_h2,num_w2))
        #matrix_1 = np.full((num_data1,num_ch12,num_h1,num_w1),211)
        #matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),21)

        # preprocessing
        # if(mode == "Forward_WS" or mode == "Backward_WS"):
        #     #Im2col_matrix_1_WS, Im2col_matrix_2_WS = preprocessor.preprocessing(matrix_1, matrix_2, mode, result_size, stride)
        #     np.savetxt('./TEST_Im2col_1', Im2col_matrix_1_WS[:,:], fmt='%i', delimiter=',\t')
        #     np.savetxt('./TEST_Im2col_2', Im2col_matrix_2_WS[:,:], fmt='%i', delimiter=',\t')
        # elif(mode == "Backward_OS"):
        #     #Im2col_matrix_1_OS, Im2col_matrix_2_OS = preprocessor.preprocessing(matrix_1, matrix_2, mode, result_size, stride)
        #     np.savetxt('./TEST_Im2col_1', Im2col_matrix_1_OS[:,:], fmt='%i', delimiter=',\t')
        #     np.savetxt('./TEST_Im2col_2', Im2col_matrix_2_OS[:,:], fmt='%i', delimiter=',\t')
    ##########################   end of preprocessing   ###############################
    ######## End of hard_setting #########################################################################################





    ##########################   start of data_generator   ###############################

        #### Input 1. ####
        # print("\n\n\n")
        # print("=======================================================================")
        # print("START OF data_generator")
        # print("=======================================================================")
        # print("\n\nINPUT 1")
        if(mode == "Forward_WS"):
            DG_1 = Data_generator(dim_col, dim_row)
            fp_skewing_data_1 = DG_1.fp_process_input_data(Im2col_matrix_1_WS, "WIDTH", "COLUMN")
            # print("size of input1 matrix = (", fp_skewing_data_1.shape, ")")
        elif(mode == "Backward_WS"):
            DG_1 = Data_generator(dim_col, dim_row)
            fp_skewing_data_1 = DG_1.fp_process_input_data(Im2col_matrix_1_WS, "WIDTH", "ROW")
            # print("size of input1 matrix = (", fp_skewing_data_1.shape, ")")
        elif(mode == "Backward_OS"):
            DG_1 = Data_generator(dim_col, dim_row)
            fp_skewing_data_1 = DG_1.fp_process_input_data(Im2col_matrix_1_OS, "HEIGHT", "COLUMN")
            # print("size of input1 matrix = (", fp_skewing_data_1.shape, ")")
        
        
        #### input2. ####
        # print("\n\nINPUT 2")
        if(mode == "Forward_WS"):
            DG_2 = Data_generator(dim_col, dim_row)
            fp_skewing_data_2 = DG_2.fp_process_weight(Im2col_matrix_2_WS, "WIDTH")
            # print("size of input2 matrix = (", fp_skewing_data_2.shape, ")")
        elif(mode == "Backward_WS"):
            DG_2 = Data_generator(dim_col, dim_row)
            fp_skewing_data_2 = DG_2.fp_process_weight(Im2col_matrix_2_WS, "HEIGHT")
            # print("size of input2 matrix = (", fp_skewing_data_2.shape, ")")
        elif(mode == "Backward_OS"):
            DG_2 = Data_generator(dim_col, dim_row)
            fp_skewing_data_2 = DG_2.fp_process_input_data(Im2col_matrix_2_OS, "HEIGHT", "ROW")
            # print("size of input2 matrix = (", fp_skewing_data_2.shape, ")")
        # print("=======================================================================\n\n\n\n\n")
    ##########################   end of data_generator   ###############################







    ################################   START OF SYSTOLIC ARRAY #########################################
    # input : matrix_1, matrix_2
        # print("=======================================================================")
        # print("START OF SYSTOLIC ARRAY")
        # pad = math.ceil((w_W-1)/2)
    ####    case 1 : Forward_WS                                                                                                                                     (h + 2 * pad - filter_h) // stride + 1
        if(mode == "Forward_WS"):
            Sample = PE_Array(dim_col, dim_row, PE_Unit, group_size, "Forward_WS", fp_skewing_data_1, fp_skewing_data_2, None, None, fp_skewing_data_2.shape[1], fp_skewing_data_2.shape[0], fp_skewing_data_1.shape[0])#num_data1 * ((num_h1 +2*pad - w_H)//stride + 1) * ((num_w1 + 2*pad - w_H)//stride + 1))
            Sample.Forward_WS_Calculation()
    ####    case 2 : Backward_WS    
        elif(mode == "Backward_WS"):                                                                                   #col fold  #row fold 
            Sample = PE_Array(dim_col, dim_row, PE_Unit, group_size, "Backward_WS",  None, fp_skewing_data_2, fp_skewing_data_1, None,fp_skewing_data_2.shape[1] , fp_skewing_data_2.shape[0], fp_skewing_data_1.shape[0])#num_data1 * ((num_h1 +2*pad - w_H)//stride + 1) * ((num_w1 + 2*pad - w_H)//stride + 1))         
            Sample.Backward_WS_Calculation()
    ####    case 3 : Backward_OS 
        elif(mode == "Backward_OS"):
            Sample = PE_Array(dim_col, dim_row, PE_Unit, group_size, "Backward_OS",  fp_skewing_data_1, None, fp_skewing_data_2, (fp_skewing_data_1.shape[0], fp_skewing_data_2.shape[0]), fp_skewing_data_1.shape[1], fp_skewing_data_2.shape[1], None)
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

        # if(mode =="Backward_OS"):
        #     outputshape = (num_data1,num_data2,(W_height//stride + 1),(W_width//stride + 1))
        # else:
        #     outputshape = (num_data1,num_data2,((num_h1 +2*pad - w_H)//stride + 1),((num_w1 + 2*pad - w_H)//stride + 1))
        #print(Sample.return_result)
        #Basic code
        #result = preprocessor.Im2col_to_4Dmatrix(Sample.return_result, outputshape)
        #ResNet-50 Code get 2D result
        # result = Sample.return_result
        # Sample.clean_return_result()

        
    ################################   End OF Getting Result   #########################################


    ################################   START OF Sample_Result   #########################################
        # pad = math.ceil((w_W-1)/2)
        # X_col = im2col(matrix_1, matrix_2, stride=stride, pad=pad)
        # np.savetxt('./TRUE_Im2col_1', X_col, fmt='%i', delimiter=',\t')
        # n, c, h, w = matrix_1.shape
        # n_f, _, filter_h, filter_w = matrix_2.shape

        # out_h = (h + 2 * pad - filter_h) // stride + 1
        # out_w = (w + 2 * pad - filter_w) // stride + 1

        # out = np.matmul(X_col, matrix_2.reshape(n_f, -1).T)
        # np.savetxt('./TRUE_Im2col_2', matrix_2.reshape(n_f, -1).T, fmt='%i', delimiter=',\t')
        # out = out.reshape(n, out_h, out_w, n_f)
        # out = out.transpose(0, 3, 1, 2)
        # print('Sample_code:', out.shape)
        # print('Sample_code:',out)
    ################################   End OF Sample_Result   #########################################
        # print('fp_SIM:', result.shape)
        
        #print('fp_SIM:', result)
        
        # print("COMPARE")
        # print("RESULT: ", np.array_equal(result,out))
        # print("diff_max: ",(result - out).max())
        # print("diff_min: ",(result - out).min())
        # print("result_max_value: ",result.max())
        # print("result_min_value: ",result.min())
        # print("\n\n")
        # print("RESULT")
        # print("Cycle_PE_unit__: ", Sample.cycle_calculation_in_PE_unit, "\t*", BfpConfig.cycle_calculation_in_PE_unit)
        # print("Cycle_move_data: ", Sample.cycle_move_data, "\t*", BfpConfig.cycle_move_data)
        # print("Cycle_st_OS____: ", Sample.cycle_store_result_in_OS, "\t*", BfpConfig.cycle_store_result_in_OS)
        # print("______________________________________________________")
        # print("Total Cycle: " ,Sample.cycle,"cycles \n\n")
        cycle_result[number][1] = Sample.cycle_calculation_in_PE_unit
        cycle_result[number][2] = Sample.cycle_move_data
        cycle_result[number][3] = Sample.cycle_store_result_in_OS
        cycle_result[number][4] = Sample.cycle
    f.close()
    np.savetxt("cycle_result.csv", cycle_result, delimiter=",", fmt="%d")    
