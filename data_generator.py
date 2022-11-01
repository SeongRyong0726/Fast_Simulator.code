from unittest import result
import numpy as np

##################################################################################################
####    Class fMAC start    ######################################################################
#   group_type  ==> HEIGHT or WIDTH
#   componant   ==> ROW or COLUMN       # row(gradient) , column(data, weight)
#   assuming    #
#   1. BFP converter가 없기 떄문에 self.mantissa_bit으로 통일해서 진행한다.
#   2. grouping된 data (exp, mantissa * group_size)는 numpy의 한 cell로 보관한다.

class Data_generator():
    def __init__(self, group_size, dim_col, dim_row, componant):
        self.group_size = group_size
        self.dim_col = dim_col
        self.dim_row = dim_row
        self.componant = componant
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-    
    # 만약 data와 type gradient(row), data, weight(column)이 주어지면: Grouping과 Skew를 완성하여 내보낸다.
    # data는 n * m numpy이고, 나오는 값은 grouping된 numpy(exp * 1, mantissa+sign * k)이어야 한다.
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def process_input_data(self,data,group_type,mantissa_bit):
        self.data = data
        self.group_type = group_type
        self.mantissa_bit = mantissa_bit
        self.data_grouping()
        self.data_skewing() 

    def data_grouping(self):
        self.group_size
        self.mantissa_bit 
        self.data
        result =1
        # processing==> fp data를 BFP로 바꾸고 GROUPING한다. (2 bit씩 쪼개서) 중요
        self.grouping_data = result

    def data_skewing(self):
        self.grouping_data
        if(self.componant =="ROW"):
            fold_unit = self.dim_row
        elif(self.componant == "COLUMN"):
            fold_unit = self.dim_col
        #processing
    
    def load_data(self, data):
        self.grouping_data = data
        self.data_skewing()