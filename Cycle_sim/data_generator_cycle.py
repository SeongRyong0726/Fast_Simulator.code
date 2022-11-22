#from unittest import result
import numpy as np
import math
#from util.bfp.cuda_bfp_wrapper import *
from Simulator_Config import BfpConfig
from fp_container import fp_container
import sys


##################################################################################################
####    Class fMAC start    ######################################################################
#   group_type  ==> HEIGHT or WIDTH
#   component   ==> ROW or COLUMN       # row(gradient) , column(data, weight)
#   assuming    #
#   1. BFP converter가 없기 떄문에 self.mantissa_bit으로 통일해서 진행한다.
#   2. grouping된 data (exp, mantissa * group_size)는 numpy의 한 cell로 보관한다.

class Data_generator():
    def __init__(self,dim_col, dim_row):
        self.group_size = BfpConfig.group_size
        #self.mantissa_bit = BfpConfig.bfp_M_Bit
        self.dim_col = dim_col
        self.dim_row = dim_row
        
        # self.place_exp = 0
        # self.place_sign = 1
        # self.place_mantissa = 1 + self.group_size
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-    
    # MAIN FUNCTION
    # 만약 data와 type gradient(row), data, weight(column)이 주어지면: Grouping과 Skew를 완성하여 내보낸다.
    # data는 n * m numpy이고, 나오는 값은 grouping된 numpy(exp * 1, mantissa+sign * k)이어야 한다.
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # 1. BFP
    # def process_input_data(self,data,group_type,component):
    #     self.data = data
    #     self.group_type = group_type
    #     self.component = component

    #     self.bfp_group_matrix = self.BFP_data_grouping()  #2D
    #     self.bfp_skewing_data = self.BFP_data_skewing(self.bfp_group_matrix) 
    #     return self.bfp_skewing_data

    # def process_weight(self, data, group_type):
    #     self.data = data
    #     self.group_type = group_type

    #     self.bfp_group_matrix = self.BFP_data_grouping()
    #     return self.bfp_group_matrix

    # 2. FP
    def fp_process_input_data(self, data,group_type,component):
        self.fp_data = data
        self.fp_group_type = group_type
        self.fp_component = component

        self.fp_group_matrix = self.fp_data_grouping()
        self.fp_skewing_data = self.fp_data_skewing(self.fp_group_matrix)
        return self.container_np

    def fp_process_weight(self, data, group_type):
        self.fp_data = data
        self.fp_group_type = group_type

        self.container_np = self.fp_data_grouping()
        return self.container_np
    #SUB FP_function 1
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    #   component   ==> ROW or COLUMN       # row(gradient) , column(data, weight)
    def fp_data_grouping(self):
        data = self.fp_data
        M = data.shape[0]
        K = data.shape[1]
        print("______________________________________________________________________")
        print("1. FP_DATA_GROUPING\n")
        if(self.fp_group_type == "WIDTH"):
            #result = np.zeros(M, math.ceil(K/self.group_size), self.group_size) # M x ceil(K/g) cell matrix /and group_size array for every one cell 
            self.make_container_numpy(M, math.ceil(K/self.group_size))
            print("SIZE = ", self.container_np.shape)
            # for m in range(M):
            #     # print("(",m ,") \t", end = '')
            #     for k in range(math.ceil(K/self.group_size)):
            #         if(K % self.group_size == 0):
            #             for i in range(self.group_size):
            #                 self.container_np[m][k].insert_value(i, data[m][k*self.group_size + i])
            #                 # print(self.container_np[m][k].get_value(i)," ", end ='')
            #             # print("| ", end ='')
            #         else:
            #             if(k != (math.ceil(K/self.group_size)-1)):
            #                 for i in range(self.group_size):
            #                     self.container_np[m][k].insert_value(i, data[m][k*self.group_size + i])
            #                     # print(self.container_np[m][k].get_value(i)," ", end ='')
            #             elif(k == (math.ceil(K/self.group_size)-1)):
            #                 for i in range(K % self.group_size):
            #                     self.container_np[m][k].insert_value(i, data[m][k*self.group_size + i])
            #                     # print(self.container_np[m][k].get_value(i)," ", end ='')
            #             # print("| ", end ='')    
            #     # print("")
            return self.container_np

        elif(self.fp_group_type == "HEIGHT"):
            self.make_container_numpy(math.ceil(M/self.group_size), K)
            print("SIZE = ", self.container_np.shape)
            # for m in range(math.ceil(M/self.group_size)):
            #     # print("(",m ,") \t", end = '')
            #     for k in range(K):
            #         if(M%self.group_size == 0):
            #             for i in range(self.group_size):
            #                 self.container_np[m][k].insert_value(i, data[m * self.group_size + i][k])
            #                 # print(self.container_np[m][k].get_value(i)," ", end ='')
            #             # print("| ", end ='')
            #         else:
            #             if(m != math.ceil(M/self.group_size)-1 ):
            #                 for i in range(self.group_size):
            #                     self.container_np[m][k].insert_value(i, data[m * self.group_size + i][k])
            #                     # print(self.container_np[m][k].get_value(i)," ", end ='')
            #             elif(m == math.ceil(M/self.group_size)-1):
            #                 for i in range(M % self.group_size):
            #                     self.container_np[m][k].insert_value(i, data[m * self.group_size + i][k])
            #                     # print(self.container_np[m][k].get_value(i), " ",end ='')
            #             # print("| ", end ='')         
            #     # print("")
            return self.container_np

        else:
            print("error")
            exit(-1)    

    def fp_data_skewing(self, data):
    # data는 이미 grouping 된 상태 [2D : R x C]
        row = data.shape[0]
        col = data.shape[1]
        data = data
        self.make_container_numpy(row + self.dim_col + self.dim_row, col)#np.zeros(row + self.dim_col + self.dim_row, col)
        base = 0
        print("______________________________________________________________________")
        print("2. FP_DATA_SKEWING\n")
        # if(self.fp_component == "COLUMN"):
        #     for num_fold in range (math.ceil(col/self.dim_col)):
        #         if(num_fold != math.ceil(col/self.dim_col)-1):
        #             unit_num_to_process = self.dim_col
        #         else:
        #             if(col % self.dim_col == 0):
        #                 unit_num_to_process = self.dim_col
        #             else:
        #                 unit_num_to_process = col % self.dim_col
        #         for i in range(unit_num_to_process):
        #             for j in range(row):
        #                 #print("(",i+j , base + i,")", end = '')
        #                 for g in range(self.group_size):
        #                     self.container_np[i+j][base + i].insert_value(g,data[j][base + i].get_value(g))
        #                     #result[i+j][base + i][g] = data[j][base + i][g]
        #                 #print()
        #         base += self.dim_col

        # elif(self.fp_component == "ROW"):
        #     for num_fold in range (math.ceil(col/self.dim_row)):
        #         if(num_fold != (math.ceil(col/self.dim_row)-1)):
        #             unit_num_to_process = self.dim_row
        #         else:
        #             if(col%self.dim_row == 0):
        #                 unit_num_to_process = self.dim_row
        #             else:
        #                 unit_num_to_process = col % self.dim_row
        #         for i in range (unit_num_to_process):
        #             for j in range(row):
        #                 for g in range(self.group_size):
        #                     self.container_np[i+j][base + i].insert_value(g,data[j][base + i].get_value(g))
        #                     #result[i + j][base + i][g] = data[j][base + i][g]
        #         base += self.dim_row
        # else:
        #     print("error")
        #     exit(-1)
        print("skew_SIZE: ", self.container_np.shape)
        # for a in range (self.container_np.shape[0]):
        #     print("<",a,">", end='')
        #     for b in range (self.container_np.shape[1]):
        #         for g in range(self.group_size):
        #             print(self.container_np[a][b].get_value(g), end = '')
        #         print("|", end ='')
        #     print("")
        return self.container_np


    #sub function 1-2, 2-1
    def make_container_numpy(self, row, col):
        self.container_np = np.ndarray((row, col),dtype=np.object)#np.zeros(file_row, file_col, 2 + self.group_size)
        for i in range(row):
            for j in range(col):
                self.container_np[i][j] = fp_container()
    

    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    #SUB function 1
    # def BFP_data_grouping(self):
    #     # Converter parameter
    #     M = self.data.shape[0]
    #     K = self.data.shape[1]
    #     shape = (M, K)
    #     use_multi_exp = False
    #     apply_thresholding = False
    #     threshold = 5 #큰의미 없을듯 위에 false라서
    #     # convert fp to bfp
    #     if(self.group_type == "WIDTH"):
    #         matrix = self.data
    #         shape = (matrix.shape[0], matrix.shape[1])
    #         Converter = CudaBfpWrapper(shape, use_multi_exp, apply_thresholding, threshold)
    #         fp_input = torch.tensor(matrix, dtype=torch.float32).to('cuda')
    #         Converter.run_convert_fp_to_bfp(src_tensor=fp_input, is_stochastic_rounding=False)

    #         bfp_group_matrix = self.make_new_file(matrix, Converter)
    #         return bfp_group_matrix

    #     elif(self.group_type == "HEIGHT"): 
    #         matrix = np.transpose(self.data)  # transpose
    #         shape = (matrix.shape[0], matrix.shape[1])
    #         Converter = CudaBfpWrapper(shape, use_multi_exp, apply_thresholding, threshold)
    #         fp_input = torch.tensor(matrix, dtype=torch.float32).to('cuda')
    #         Converter.run_convert_fp_to_bfp(src_tensor=fp_input, is_stochastic_rounding=False)
    #         #make new file
    #         bfp_group_matrix = self.make_new_file(matrix, Converter)
    #         result = np.transpose(bfp_group_matrix) #transpose again
    #         return result
    #     else:
    #         print("error:data_grouping(self)")
    # #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # #SUB function 2
    # def BFP_data_skewing(self, bfp_matrix): 
    #     base = 0
    #     Height = bfp_matrix.shape[0] # row a
    #     Width = bfp_matrix.shape[1] # col b
    #     result = self.make_container_numpy(Height+self.dim_col+self.dim_row ,Width)


    #     if(self.component == "COLUMN"):#data_type == "Data"): ## col
    #         for num_fold in range (math.ceil(Width/self.dim_col)):
    #             if(num_fold != (math.ceil(Width/self.dim_col)-1)):
    #                 col_num_to_process = self.dim_col
    #             else:
    #                 col_num_to_process = Width % self.dim_col
    #             for i in range (col_num_to_process):
    #                 for j in range (Height):
    #                     data = bfp_matrix[j][base + i]
    #                     result[j + i][base + i].insert(data.sign_g, data.exp_g, data.mantissa_g)
    #             base += self.dim_col

    #     elif(self.component =="ROW"):#data_type == "Gradient"):
    #         for num_fold in range (math.ceil(Width/self.dim_row)):
    #             if(num_fold != (math.ceil(Width/self.dim_row)-1)):
    #                 row_num_to_process = self.dim_row
    #             else:
    #                 row_num_to_process = Width % self.dim_row
    #             for i in range (row_num_to_process):
    #                 for j in range (Height):
    #                     data = bfp_matrix[j][base + i]
    #                     result[j + i][base + i].insert(data.sign_g, data.exp_g, data.mantissa_g)
    #             base += self.dim_row
    #     else:
    #         "error"
    #     return result



    # #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # #SUB function 1-1
    # def make_new_file(self, matrix, Converter):
    #     file_row = matrix.shape[0]
    #     file_col = math.ceil(matrix.shape[1]/self.group_size)
    #     remaining = matrix.shape[1] % self.group_size
    #     # make return_file with Bfp_container class
    #     return_file = self.make_container_numpy(file_row, file_col)
    #     # make return_file
    #     np_bfp_S = Converter.bfp_S.cpu().numpy()
    #     np_bfp_E = Converter.bfp_E.cpu().numpy()
    #     np_bfp_M = Converter.bfp_M.cpu().numpy()
    #     print("S",np_bfp_S)
    #     print("E",np_bfp_E)
    #     print("M",np_bfp_M)
    #     for d_row in range(file_row):
    #         for d_col in range(file_col):
    #             temp_container_S = np.zeros(self.group_size) 
    #             temp_container_E = np.zeros(1)
    #             temp_container_M = np.zeros(self.group_size)
                
    #             temp_container_E = np_bfp_E[d_row, d_col * self.group_size] #exp
    #             if(d_col != (file_col - 1)):
    #                 for group in range(self.group_size):
    #                     temp_container_S[group] = np_bfp_S[d_row, d_col * self.group_size + group] #sign
    #                     temp_container_M[group] = np_bfp_M[d_row, d_col * self.group_size + group] # mantissa
    #             else: # end point
    #                 for group in range(remaining):
    #                     temp_container_S[group] = np_bfp_S[d_row, d_col * self.group_size + group] #sign
    #                     temp_container_M[group] = np_bfp_M[d_row, d_col * self.group_size + group] # mantissa
    #             return_file[d_row][d_col].insert(temp_container_S, temp_container_E, temp_container_M)
    #     return return_file
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # Debugging function
    # def show_array_of_what(self, data, what):
    #     if(what != "S" and what !="E" and what !="M"):
    #         print("error: show_array_of what")
    #     if(what == "S"): contents = "sign_g"
    #     elif(what == "E"): contents = "exp_g"
    #     elif(what == "M"): contents = "mantissa_g"
    #     result = np.zeros(data.shape)
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[1]):
    #             result[i][j] = data[i][j].exp_g
    #     print(result)

if __name__ == "__main__":
    converter = Data_generator(4,4)
    data = np.full((5,18), 128, dtype = np.float32)
    #data = 100 * np.random.rand(9,9) + 50
    t_data = np.transpose(data)
    D   =   converter.process_input_data(data,"HEIGHT", "ROW")
    #D_t =   converter.process_input_data(t_data,"HEIGHT", "ROW")
    converter.show_array_of_what(D, "E")


                # print("data")
            # print(self.data)
            # print("bfp_S")
            # print(Converter.bfp_S)
            # print("bfp_E")
            # print(Converter.bfp_E)
            # print("bfp_M")
            # print(Converter.bfp_M)
            # print("32_bit_M")
            # print(Converter.bfp_32_bit_M)