import numpy as np
import math

####    Class Systolic Array end    ##############################################################
##################################################################################################
####    Class fMAC start    ######################################################################
class fMAC():
    def __init__(self, row_num, col_num, group_size,mode):
        # fMAC ID (col_num, row_num)
        self.col_num = col_num
        self.row_num = row_num
        # number of multipier
        self.group_size = group_size
        # mode 3가지
        self.mode = mode
        # operand
        self.exponant_from_left = np.zeros(1)
        self.mantissa_from_left = np.zeros(group_size)
        self.exponant_from_down = np.zeros(1)
        self.mantissa_from_down = np.zeros(group_size)
        
        self.weight_exponant = np.zeros(1)
        self.weight_mantissa = np.zeros(group_size)

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
        do = 1
   


    #### operation 함수: input값을 이용하여 연산하고 결과를 내놓는다.
    def operation(self):
        if(self.mode == "Forward_WS"):
            exp = self.weight_exponant + self.exponant_from_down
            mantissa = self.weight_mantissa * self.mantissa_from_down
            result = self.fp_generator(exp, mantissa)
            self.result_to_right = result

        elif(self.mode == "Backward_WS"):
            exp = self.weight_exponant + self.exponant_from_left
            mantissa = self.weight_mantissa * self.mantissa_from_left
            result = self.fp_generator(exp, mantissa)
            self.result_to_up = result

        elif(self.mode == "Backward_OS"):
            exp = self.exponant_from_left + self.exponant_from_down
            mantissa = self.mantissa_from_left * self.mantissa_from_down
            result= self.fp_generator(exp, mantissa)
            self.accumulator+= result

    