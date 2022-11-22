import numpy as np
import math
from fp_container import *
from Simulator_Config import BfpConfig

from ctypes import *

#bfp_so_file_path = "/home/sroh/sroh-backup-files/SR_sim/Lib_cuda_bfp_calculation.so"
#c_lib = ctypes.CDLL(bfp_so_file_path)

##################################################################################################
####    Class fMAC start    ######################################################################
#   mode = 3types "Forward_WS", "Backward_WS", "Backward_OS"
class PE_Unit():
    def __init__(self, row_num, col_num, group_size,mode):
        # Unit characteristic    
            # fMAC ID (col_num, row_num)
        self.col_num = col_num
        self.row_num = row_num
            # number of multipier
        self.group_size = BfpConfig.group_size
        #self.mantissa_bit = Simulator_Config.bfp_M_Bit
        #self.round = math.ceil(self.mantissa_bit/2)
            # mode 3가지
        self.mode = mode


        # operand_container [ E0 + M0 / E1 + M1 ]
        self.cont_value_left = fp_container()#Bfp_container(group_size)
        self.cont_value_down = fp_container()#Bfp_container(group_size)        
        self.cont_weight = fp_container()#Bfp_container(group_size)
        
        # operand_container => operand_np [for Dot product function]
        #self.cont_value_left_np = self.cont_to_np(self.cont_value_left)
        #self.cont_value_down_np = self.cont_to_np(self.cont_value_down)
        #self.cont_weight_np = self.cont_to_np(self.cont_weight)
        #self.DP_result 

        # component (fp type) [store value, out the value]
        self.accumulator = 0        
        self.result_to_right = 0    
        self.result_to_up = 0  


        #for calculation (E0, M0, E1, M1, register for Weight)
        # self.exponant_from_left = np.zeros(1)
        # self.mantissa_from_left = np.zeros(group_size)
        # self.exponant_from_down = np.zeros(1)
        # self.mantissa_from_down = np.zeros(group_size)
        # self.weight_exponant = np.zeros(1)
        # self.weight_mantissa = np.zeros(group_size)

    ############################################################
    ####    for FLOW of "OPERAND" + "direction of col, row"
    def get_operand_from_left(self):
        return self.cont_value_left
    def get_operand_from_down(self):
        return self.cont_value_down
    def load_operand_on_left_entrance(self, container):
        self.cont_value_left.insert(container)#container.sign_g, container.exp_g, container.mantissa_g)
    def load_operand_on_down_entrance(self, container):
        self.cont_value_down.insert(container)#container.sign_g, container.exp_g, container.mantissa_g)
 
    
    ####    for FLOW of "sub_RESULT" + "direction of col, row"
    def get_result_to_right(self):
        return self.result_to_right
    def get_result_to_up(self):
        return self.result_to_up
    def load_result_on_acc(self, result):
        self.accumulator = result
    
    ####    for FLOW of "WEIGHT"
    def get_Weight_from_down(self):
        return self.cont_weight
    def load_Weight_value(self, W):
        self.cont_weight.insert(W)#W.sign_g, W.exp_g, W.mantissa_g)
    ###########################################################

    # def fp_generator(self, exp, mantissa):
    #     return exp

    # def cont_to_np(self, cont):
    #     result = np.zeros(self.group_size*2+1)
    #     result[0] = cont.exp_g[0]
    #     for i in range (self.group_size):
    #         result[i+1] = cont.mantissa_g[i]
    #     for j in range (self.group_size):
    #         result[j+1+self.group_size] = cont.sign_g[j]
    #     return result


    #### operation 함수 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # now, have container  
    # (container, round) => 2bit mantissa+sign , exp를 내놓는 함수가 필요
    # 함수의 결과값으로 계산하고 
    # (결과 exp, mantissa+sign) ==> floating point만드는 함수가 필요
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    
    # container 는 numpy로 교체후 array로 넘겨준다. exp(1), man(g), sign(g)
    # void g_bfp_dot_product(
    # const int32_t* fst_container, const int32_t* snd_container, 
    # const int32_t group_size, const int32_t mantissa_bit, const float bfp_M_bits_to_divide, 
    # const int32_t round, 
    # const float* result){
    def dot_product(self, cont1, cont2):
        result = 0
        #print("_________")
        for g in range (self.group_size):
            #print(cont1.get_value(g), " * " ,cont2.get_value(g), "=",cont1.get_value(g) * cont2.get_value(g) )
            result += cont1.get_value(g) * cont2.get_value(g)
        return result

    def operation(self):
        if(self.mode == "Forward_WS"):
            result = self.dot_product(self.cont_value_down, self.cont_weight)
            self.result_to_right = self.accumulator + result

        elif(self.mode == "Backward_WS"):
            result = self.dot_product(self.cont_value_left, self.cont_weight)
            self.result_to_up = self.accumulator + result

        elif(self.mode == "Backward_OS"):
            result = self.dot_product(self.cont_value_down, self.cont_value_left)
            self.accumulator += result

    