import numpy as np
import math

##################################################################################################
####    Class fMAC start    ######################################################################
#   mode = 3types "Forward_WS", "Backward_WS", "Backward_OS"
class PE_Unit():
    def __init__(self, row_num, col_num, group_size,mode):
        # fMAC ID (col_num, row_num)
        self.col_num = col_num
        self.row_num = row_num
        # number of multipier
        self.group_size = group_size
        # mode 3가지
        self.mode = mode
        # operand (E0, M0, E1, M1, register for Weight)
        self.exponant_from_left = np.zeros(1)
        self.mantissa_from_left = np.zeros(group_size)
        self.exponant_from_down = np.zeros(1)
        self.mantissa_from_down = np.zeros(group_size)
        
        self.weight_exponant = np.zeros(1)
        self.weight_mantissa = np.zeros(group_size)
        # component (fp type)
        self.accumulator = 0        
        self.result_to_right = 0    
        self.result_to_up = 0       

    ############################################################
    ####    for flow of "OPERAND" + "direction of col, row"
    def get_operand_from_left(self):
        return self.exponant_from_left, self.mantissa_from_left
    def get_operand_from_down(self):
        return self.exponant_from_down, self.mantissa_from_down
    def load_operand_on_left_entrance(self, exp, mantissa):
        self.exponant_from_left = exp
        self.mantissa_from_left = mantissa
    def load_operand_on_down_entrance(self, exp, mantissa):
        self.exponant_from_down = exp
        self.mantissa_from_down = mantissa
    
    ####    for flow of "sub_RESULT" + "direction of col, row"
    def get_result_to_right(self):
        return self.result_to_right
    def get_result_to_up(self):
        return self.result_to_up
    def load_result_on_acc(self, result):
        self.accumulator = result
    
    ####    Weight
    def get_Weight_from_down(self):
        return self.weight_exponant, self.weight_mantissa
    def load_Weight_value(self, exp, mantissa):
        self.weight_exponant = exp
        self.weight_mantissa = mantissa
    ###########################################################

    def fp_generator(self, exp, mantissa):
        return exp
   


    #### operation 함수: input값을 이용하여 연산하고 결과를 내놓는다.
    def operation(self):
        if(self.mode == "Forward_WS"):
            exp = self.weight_exponant + self.exponant_from_down
            mantissa = np.dot(self.weight_mantissa, self.mantissa_from_down)
            result = self.accumulator + self.fp_generator(exp, mantissa.sum(axis=0))
            test_3 = self.accumulator + self.weight_exponant * self.exponant_from_down
            self.result_to_right =test_3 # result

        elif(self.mode == "Backward_WS"):
            exp = self.weight_exponant + self.exponant_from_left
            mantissa = np.dot(self.weight_mantissa, self.mantissa_from_left)
            result = self.accumulator + self.fp_generator(exp, mantissa.sum(axis=0))
            test_3 = self.accumulator + self.weight_exponant * self.exponant_from_left
            self.result_to_up = test_3 #result

        elif(self.mode == "Backward_OS"):
            exp = self.exponant_from_left + self.exponant_from_down
            mantissa = np.dot(self.mantissa_from_left, self.mantissa_from_down)
            result= self.fp_generator(exp, mantissa.sum(axis=0))
            test_3 = self.exponant_from_left * self.exponant_from_down
            self.accumulator+=  test_3#result

    