import numpy as np
import math
from PE_Unit_cycle import PE_Unit
from fp_container import fp_container
from Simulator_Config import BfpConfig
from tqdm import tqdm

class PE_Array():                                                                         ## OS       #fold       #fold       ## WS                   
    def __init__(self, dim_col, dim_row, opUnit, groupsize, mode, Input_A, Input_W, Input_G, OS_result_shape, col_total, row_total, num_of_Activation_width):
        ####    System information
        self.mode = mode                    #{"Forward_WS", "Backward_WS", "Backward_OS"}
        self.group_size = groupsize         #

        self.dim_row = dim_row
        self.dim_col = dim_col
        self.Unit = PE_Unit 
        self.Unit_array = list()
                ####    Systolic Array fMAC 가기 직전 entrance
        self.left_side_entrance = np.ndarray(self.dim_row,dtype=np.object)
        self.bottom_side_entrance = np.ndarray(self.dim_col,dtype=np.object)  # Weight, Data
                ####    Systolic Array SRAM 가기 직전 entrance
        self.right_side_entrance = np.full(self.dim_row, 0.0)   
        self.up_side_entrance = np.full(self.dim_col, 0.0) 
        for i in range(self.dim_row):
            self.left_side_entrance[i] = fp_container()
        for j in range(self.dim_col):
            self.bottom_side_entrance[j] = fp_container() 

        #### fold
        self.row_total = row_total
        self.col_total = col_total
        #print("asdsadasdasd", row_total, col_total)          
        self.row_fold_end = math.ceil(self.row_total/self.dim_row)
        self.col_fold_end = math.ceil(self.col_total/self.dim_col)
        #print("fold_end", self.row_fold_end, self.col_fold_end) 
        self.row_fold_current = 0
        self.col_fold_current = 0
        self.fold_info_array = np.full((self.row_fold_end,self.col_fold_end,2),0)
        # self.round_for_1 = round_for_1
        # self.round_for_2 = round_for_2
        ####    Process Variable
        self.row_use = self.dim_row
        self.col_use = self.dim_col

             ####    SRAM
        self.make_fold_info_array()
        if(self.mode == "Backward_OS" or self.mode == "Forward_WS"):
            self.Data_SRAM = Input_A #self.make_skew_input(Input_A, "Data", self.round_for_1)                ##numpy
        self.Weight_SRAM = Input_W     
        if(self.mode == "Backward_WS"):                                         ##numpy
            self.Gradient_SRAM = Input_G #self.make_skew_input(Input_G, "Gradient", self.round_for_1)            ##numpy
        elif(self.mode == "Backward_OS"):
            self.Gradient_SRAM = Input_G #self.make_skew_input(Input_G, "Gradient", self.round_for_2)
        self.update_colrow_use_variable()
            ####    for OS type
        if(self.mode =="Backward_OS"):
            self.OS_result_height = Input_A.shape[1] #2D
            self.OS_result_width = Input_G.shape[1]  #2D
            ####    for WS type
        self.num_of_Activation_width = num_of_Activation_width

        ####    trace_변수
        self.cycle = 0              ##count_cycle
        self.cycle_move_data = 0
        self.cycle_calculation_in_PE_unit = 0
        self.cycle_store_result_in_OS = 0
        self.util = 0
        if(self.mode == "Forward_WS" or self.mode == "Backward_WS"):
            self.temp_result = np.full((self.num_of_Activation_width, max(self.dim_col,self.dim_row)),0.) 
            self.go_result = np.full(((self.num_of_Activation_width- self.dim_row - self.dim_col, max(self.dim_col,self.dim_row))),0.)

        if(self.mode == "Forward_WS"):
            self.result_height= Input_A.shape[0] - self.dim_row - self.dim_col
            self.result_width = Input_W.shape[0]
        elif (self.mode=="Backward_WS"):
            self.result_height= Input_G.shape[0] - self.dim_row - self.dim_col
            self.result_width = Input_W.shape[1]
        elif (self.mode=="Backward_OS"):
            self.result_height = self.OS_result_height 
            self.result_width = self.OS_result_width  
        self.return_result =np.full((self.result_height, self.result_width),0.)   #### TO_DO
    

        ####    Array에 Unit(fMAC)을 채우고 초기화 // 가로(cols) 세로(rows)
        for i in range (self.dim_row):
            self.Unit_array.append(list())
            for j in range (self.dim_col):
                self.Unit_array[i].append(self.Unit(i, j, groupsize, mode))
        if(BfpConfig.check_sram == True):
            print("_Check_SRAM : ON")
            if(self.mode == "Backward_OS" or self.mode == "Forward_WS"):
                self.print_Data_SRAM()
            if(self.mode == "Backward_OS" or self.mode == "Backward_WS"):
                self.print_Gradient_SRAM()
            if(self.mode =="Forward_WS" or self.mode == "Backward_WS"):
                self.print_Weight_SRAM()
        else:
            print("_Check_SRAM: OFF")

####    FOR_Debuging    ######################################################################################
    def print_Data_SRAM(self):
        #np.savetxt('./Data_SRAM', self.Data_SRAM[:,:].get_value(0), fmt='%i', delimiter=',\t')
        print("DATA")
        for i in range (self.Data_SRAM.shape[0]):
            print(i, "\t", end = '' )
            for j in range (self.Data_SRAM.shape[1]):
                print("(",i,j,")", end='')
                for g in range (self.group_size):
                    print(self.Data_SRAM[i][j].get_value(g), end = '')
                print("|", end = '')
            print()
    def print_Gradient_SRAM(self):
        #np.savetxt('./Gradient_SRAM', self.Gradient_SRAM[:,:], fmt='%i', delimiter=',\t')
        print("Gradient")
        for i in range (self.Gradient_SRAM.shape[0]):
            print(i, "\t", end = '' )
            for j in range (self.Gradient_SRAM.shape[1]):
                for g in range (self.group_size):
                    print(self.Gradient_SRAM[i][j].get_value(g), end = '')
                print("|", end = '')
            print()
    def print_Weight_SRAM(self):
        #np.savetxt('./Weight_SRAM', self.Weight_SRAM[:,:], fmt='%i', delimiter=',\t')
        print("Weight")
        for i in range (self.Weight_SRAM.shape[0]):
            print(i, "\t", end = '' )
            for j in range (self.Weight_SRAM.shape[1]):
                print("(",i,j,")", end='')
                for g in range (self.group_size):
                    print(self.Weight_SRAM[i][j].get_value(g), end = '')
                print("|", end = '')
            print()
####    array of fold_information    #########################################################################
    def make_fold_info_array(self):
        self.update_colrow_use_variable()
        for _ in range(self.col_fold_end*self.row_fold_end):
            self.fold_info_array[self.row_fold_current][self.col_fold_current][0] = self.row_use
            self.fold_info_array[self.row_fold_current][self.col_fold_current][1] = self.col_use
            if(self.mode == "Backward_WS"):
                self.update_fold_variable_B_WS()
            else:
                self.update_fold_variable_F_WS()
        self.update_colrow_use_variable()
        #print("fold_info_array")
        #print(self.fold_info_array)

####    data_generation으로 옯겨져야함    #########################################################################
    # def make_skew_input(self, Im2col_matrix, data_type, round): #Data, Gradient
    #     row_base_addr = 0
    #     col_base_addr = 0
    #     a = Im2col_matrix.shape[1]# "11.01"0]
    #     b = Im2col_matrix.shape[2]# "11.01"1]
    #     c = Im2col_matrix.shape[3]# "11.01"2]

    #     if(data_type == "Data"): ## col
    #         result = np.full((round ,a+self.dim_col+self.dim_row ,b ,c),0.)
    #         for r in range(round):  # "11.01"
    #             for i in range (self.col_fold_end):
    #                 num_col = self.fold_info_array[0][i][1]
    #                 #print("num_col", num_col)
    #                 for h in range (Im2col_matrix.shape[1]):# "11.01"0]):
    #                     for j in range (num_col):
    #                         result[r][h+j][j+col_base_addr][0] = Im2col_matrix[r][h][j+col_base_addr][0]
    #                 col_base_addr += self.dim_col
    #             col_base_addr = 0
    #     elif(data_type == "Gradient"):
    #         result = np.full((round ,a+self.dim_col+self.dim_row ,b ,c),0)
    #         for r in range(round):  # "11.01"
    #             for i in range (self.row_fold_end):
    #                 num_row = self.fold_info_array[i][0][0]
    #                 #print("num_row", num_row)
    #                 for h in range (Im2col_matrix.shape[1]):# "11.01"0]):
    #                     for j in range (num_row):
    #                         result[r][h+j][j+row_base_addr][0] = Im2col_matrix[r][h][j+row_base_addr][0]
    #                 row_base_addr += self.dim_row
    #             row_base_addr = 0
    #     else:
    #         "error"
    #     return result

####_______________________________________________________________________________________________________________________________________
#### Function: Function :   update next_fold, next row,col_use, Use when finishing every fold (n*folds = entire caculation)
####               변수  :  현제 row,col_fold / 총 row,col_fold 
####    Functions.py TEST1
####_______________________________________________________________________________________________________________________________________

    def update_colrow_use_variable(self):
        #### update row,col_use
        self.col_use = self.dim_col
        self.row_use = self.dim_row
        if(self.row_fold_current == (self.row_fold_end -1)):
            if(self.row_total % self.dim_row == 0):
                self.row_use = self.dim_row
            else:
                self.row_use = self.row_total % self.dim_row
        if(self.col_fold_current == (self.col_fold_end -1)):
            if(self.col_total % self.dim_col == 0):
                self.col_use = self.dim_col
            else:
                self.col_use = self.col_total % self.dim_col
    
    def update_fold_variable_B_WS(self):
        #### fold value update
        self.row_fold_current += 1
        if(self.row_fold_current == self.row_fold_end):
            self.row_fold_current = 0
            self.col_fold_current +=1
            if(self.col_fold_current == self.col_fold_end):
                #### end: reset "*_fold_current", "*_use"
                self.col_use = self.dim_col
                self.row_use = self.dim_row
                self.col_fold_current = 0
                self.row_fold_current = 0
                return False                                          ## 연산 끝
        self.update_colrow_use_variable()
        return True

    def update_fold_variable_F_WS(self):
        #### fold value update
        self.col_fold_current += 1
        if(self.col_fold_current == self.col_fold_end):
            self.col_fold_current = 0
            self.row_fold_current +=1
            if(self.row_fold_current == self.row_fold_end):
                #### end: reset "*_fold_current", "*_use"
                self.col_use = self.dim_col
                self.row_use = self.dim_row
                self.col_fold_current = 0
                self.row_fold_current = 0
                return False                                          ## 연산 끝
        self.update_colrow_use_variable()
        return True

####_______________________________________________________________________________________________________________________________________
#### Function:    Systolic Array에서 operand와 result를 flow하는 함수이다. (하나의 fMAC단위로 진행)
####              SRAM-->fMAC(operand)
####              fMAC-->fMAC(operand/result)
####              fMAC-->SRAM(result)
####     Function.py TEST2
####_______________________________________________________________________________________________________________________________________

    def separate_exp_man(self, list):
        ####  bottom_side_entrace에는 tuple exp_up, mantissa는 numpy ####
        exp = np.array(list[0])
        mantissa = np.full((self.group_size,), 0)
        for i in range (self.group_size):
            mantissa[i] = list[i+1]
            #mantissa = np.insert(mantissa, i , list[i+1])
        return exp, mantissa ## numpy리턴

    def fMAC_operand_pass_right(self, row_num, col_num):                                        #operand
        if row_num>=0 and row_num<=self.dim_row-1       and col_num > 0 and col_num<self.dim_col:  #case) fMAC-->fMAC  
            fp_container = self.Unit_array[row_num][col_num-1].get_operand_from_left()    #이전꺼에서 받아서 넣는다
            self.Unit_array[row_num][col_num].load_operand_on_left_entrance(fp_container)
        elif row_num>=0 and row_num<=self.dim_row-1     and col_num==0:                              #case) SRAM-->fMAC (load)  
            fp_container = self.left_side_entrance[row_num]
            #print(str(type(self.left_side_entrance[0])))
            self.Unit_array[row_num][col_num].load_operand_on_left_entrance(fp_container)                                                                                 
        else:
            print("error: function 'fMAC_operand_pass_right'")

    def fMAC_operand_pass_up(self, row_num=0, col_num=0):                                         #operand
        if row_num>0 and row_num<self.dim_row and col_num >= 0 and col_num<=self.dim_col-1:          #case) fMAC-->fMAC  
            fp_container = self.Unit_array[row_num-1][col_num].get_operand_from_down()
            self.Unit_array[row_num][col_num].load_operand_on_down_entrance(fp_container)
        elif row_num==0                         and col_num >= 0 and col_num<=self.dim_col-1:        #case) SRAM-->fMAC (load)
            fp_container = self.bottom_side_entrance[col_num]
            self.Unit_array[row_num][col_num].load_operand_on_down_entrance(fp_container)                                                                                   

        else:
            print("error: function 'fMAC_operand_pass_up'")

    def fMAC_result_pass_right(self, row_num=0, col_num=0):                                         #result (부분합)
        if row_num>=0 and row_num<=self.dim_row-1   and col_num >= 0 and col_num<self.dim_col-1:          #case) fMAC-->fMAC  
            result = self.Unit_array[row_num][col_num].get_result_to_right()
            self.Unit_array[row_num][col_num+1].load_result_on_acc(result)
        elif row_num>=0 and row_num<=self.dim_row-1 and col_num==self.dim_col-1:                       #case) SRAM직전 fMAC-->SRAM
            self.right_side_entrance[row_num] = self.Unit_array[row_num][col_num].get_result_to_right()
        else:
            print("error: function 'fMAC_result_pass_right'")

    def fMAC_result_pass_up(self, row_num=0, col_num=0):
        if row_num>=0 and row_num<self.dim_row-1 and col_num >= 0 and col_num<=self.dim_col-1:       #case) fMAC-->fMAC  
            result = self.Unit_array[row_num][col_num].get_result_to_up()
            self.Unit_array[row_num+1][col_num].load_result_on_acc(result)
        elif row_num==self.dim_row-1            and col_num >= 0 and col_num<=self.dim_col-1:       #case) SRAM직전 fMAC-->SRAM
            self.up_side_entrance[col_num] = self.Unit_array[row_num][col_num].get_result_to_up()
        else:
            print("error: function 'fMAC_result_pass_right'")


####_______________________________________________________________________________________________________________________________________
#### Function: Systolic_Array_1cycle_calculation 함수를 사용하면 1cycle에 걸친 Systolic Array 내부의 동작을 수행한다.
#### Functions.py TEST3
####_______________________________________________________________________________________________________________________________________

    def Systolic_Array_1cycle_calculation(self):
        ####    1. 내부에서 연산 수행
        self.cycle_calculation_in_PE_unit += 1
        for i in range(self.dim_row):
            for j in range(self.dim_col):
                self.Unit_array[i][j].operation()   ##실제값이 아니라 중간에 텅빈 공간을 operate하는 경우 error == True를 return
        ####    2. fMAC이 다음계산을 위해 값을 받아옴
        for i in range(self.dim_row):
            for j in range(self.dim_col):
                if(self.mode == "Backward_OS"):
                    self.fMAC_operand_pass_right(self.dim_row - i -1,self.dim_col - j -1)
                    self.fMAC_operand_pass_up(self.dim_row - i-1,self.dim_col - j-1)
                # WS type은 result_pass 함수에서 결과를 SRAM으로 넣어준다.
                elif(self.mode == "Forward_WS"):
                    self.fMAC_operand_pass_up(self.dim_row - i-1,self.dim_col - j-1)
                    self.fMAC_result_pass_right(self.dim_row - i-1,self.dim_col - j-1)
                elif(self.mode == "Backward_WS"):
                    self.fMAC_operand_pass_right(self.dim_row - i-1,self.dim_col - j-1)
                    self.fMAC_result_pass_up(self.dim_row - i-1,self.dim_col - j-1)
    
####_______________________________________________________________________________________________________________________________________
#### Function: WS
#### Functions.py TEST3 (Weight pre-load TEST)
#### Functions.py TEST4 (Forward WS)
#### Functions.py TEST5 (Backward WS)
####_______________________________________________________________________________________________________________________________________

    def preload_Weight_row_by_row(self, row_num, col_num):
        if row_num>0 and row_num<self.dim_row and col_num >= 0 and col_num<=self.dim_col-1:       #case) fMAC-->fMAC  
            fp_container = self.Unit_array[row_num-1][col_num].get_Weight_from_down()
            self.Unit_array[row_num][col_num].load_Weight_value(fp_container)
        elif row_num==0                         and col_num >= 0 and col_num<=self.dim_col-1:       #case) SRAM-->fMAC (load)
            fp_container = self.bottom_side_entrance[col_num]
            self.Unit_array[row_num][col_num].load_Weight_value(fp_container)
        else:
            print("error: function 'fMAC_result_pass_right'")

    def Systolic_preload_Weight(self):
        #### for TEST3
        ####    array_W = np.full((self.dim_row, self.dim_col),0)
        
        base_col = int(self.col_fold_current * self.dim_col)
        base_row = int(self.row_fold_current * self.dim_row) 
        ####    이전값 reset
        for i in range(self.dim_row):
            for j in range(self.dim_col):
                self.Unit_array[i][j].cont_weight = fp_container()
                
        ####    값 넣기        
        for k in range (self.row_use):
            temp_input_W = self.Weight_SRAM[base_row + self.row_use -1 - k][base_col: base_col + self.col_use]
            input_W = temp_input_W
            padding = fp_container()
            for _ in range(self.dim_col-self.col_use):
                input_W = np.append(input_W, [padding], axis = 0)
            #input_W = np.pad(temp_input_W, (0,self.dim_col-self.col_use), 'constant', constant_values = 0)
            self.bottom_side_entrance = input_W
            ##count_cycle
            self.cycle_move_data += 1
            for a in range(self.dim_row):
                for b in range(self.dim_col):
                    self.preload_Weight_row_by_row(self.dim_row-a-1,b)
            
    ####    1. for Forward WS
    def Systolic_Forward_WS_one_folds(self):
        ####    for TEST3
        # array_Flow = np.full((self.dim_row, self.dim_col),0)
        # array_Flow1 = np.full((self.dim_row, self.dim_col),0)
        expect_cycle = self.Data_SRAM.shape[0]

        #print(expect_cycle)
        #print(self.temp_result.shape)
        base_col = int(self.col_fold_current * self.dim_col)
        base_row = int(self.row_fold_current * self.dim_row)  
        pbar = tqdm(total=expect_cycle)
        for i in range(expect_cycle):
            pbar.update(1)
            ##count_cycle
            self.cycle_move_data += 1
            input_A = self.Data_SRAM[i][base_col: base_col + self.col_use]
            padding = fp_container()
            for _ in range(self.dim_col-self.col_use):
                input_A = np.append(input_A, [padding], axis = 0)
            #Debugging
            # print(i," ", end='')
            # for p_num in range(self.dim_col):
            #     for p_g in range(self.group_size):
            #         print(input_A[p_num].get_value(p_g), end='')
            #     print("| ",end='')
            # print()
            #input_A = np.pad(temp_input_A, ((0,self.dim_col-self.col_use),(0,0)), 'constant', constant_values = 0)
            self.bottom_side_entrance = input_A
            self.Systolic_Array_1cycle_calculation()
            self.temp_result[i][0:self.dim_row] += self.right_side_entrance[0:self.dim_row]
            #### for TEST3
            # for row in range(self.dim_row):
            #     for col in range(self.dim_col):
            #         array_Flow[row][col] = self.Unit_array[row][col].exponant_from_down
            #         array_Flow1[row][col] = self.Unit_array[row][col].result_to_right
            #print("Cycle:" , i)
            #print(array_Flow)
            #print(array_Flow1)  
        pbar.close()      

    ####    2. for Backward WS
    def Systolic_Backward_WS_one_folds(self):
        ####    for TEST3
        array_Flow = np.full((self.dim_row, self.dim_col),0)
        expect_cycle = self.Gradient_SRAM.shape[0]
        #print("G: ",self.Gradient_SRAM.shape)
        #print("Store: ", self.temp_result.shape)
        base_col = int(self.col_fold_current * self.dim_col)
        base_row = int(self.row_fold_current * self.dim_row)
        pbar = tqdm(total=expect_cycle)
        for i in range (expect_cycle):
            pbar.update(1)
            ##count_cycle
            self.cycle_move_data += 1
            input_G = self.Gradient_SRAM[i][base_row: base_row + self.row_use]
            padding = fp_container()
            for _ in range(self.dim_row-self.row_use):
                input_G = np.append(input_G, [padding], axis = 0)
            #Debugging
            # print(i," ", end='')
            # for j in range(self.dim_row):
            #     for g in range(self.group_size):
            #         print(input_G[j].get_value(g), end='')
            #     print("| ",end='')
            # print()
            #input_G = np.pad(temp_input_G, ((0,self.dim_row-self.row_use),(0,0)), 'constant', constant_values = 0)
            self.left_side_entrance = input_G
            self.Systolic_Array_1cycle_calculation()
            self.temp_result[i][0:self.dim_col] += self.up_side_entrance[0:self.dim_col]
            #### for TEST3
            # for row in range(self.dim_row):
            #     for col in range(self.dim_col):
            #         array_Flow[row][col] = self.Unit_array[row][col].exponant_from_left
            #print("Cycle:" , i)
            #print(array_Flow)
        pbar.close()



    ######################
    ####    기울어진 결과값을 저장하기 좋게 정 사각형으로(이 결과값은 전체가 아닌 일부이다)
    ####    1. for Forward WS
    def make_square_Forward_WS_result(self):
        base_col = self.dim_col
        for i in range(self.num_of_Activation_width-self.dim_col-self.dim_row):
            for j in range(self.row_use):
                self.go_result[i][j] += self.temp_result[base_col + i + j][j]

    ####    2. for Backward WS
    def make_square_Backward_WS_result(self):
        base_row = self.dim_row
        for i in range(self.num_of_Activation_width-self.dim_col-self.dim_row):
            for j in range(self.col_use):
                self.go_result[i][j] += self.temp_result[base_row + i + j][j]


    ######################
    ####    WS결과를 최종 장소에 저장
    ####    1. for Forward WS
    def store_Forward_WS_result(self):
        base_row = self.row_fold_current * self.dim_row 
        #print(self.return_result.shape)
        self.return_result[0:self.num_of_Activation_width-self.dim_row-self.dim_col , base_row : base_row + self.row_use] += self.go_result[0:self.num_of_Activation_width-self.dim_col-self.dim_row , 0:self.row_use]
    
    ####    2. for Backward WS
    def store_Backward_WS_result(self):
        base_col = self.col_fold_current * self.dim_col 
        self.return_result[0:self.num_of_Activation_width-self.dim_row-self.dim_col , base_col : base_col + self.col_use] += self.go_result[0:self.num_of_Activation_width-self.dim_col-self.dim_row , 0:self.col_use]


    ######################
    ####    1. for Forward WS
    def Forward_WS_Calculation(self):
        self.update_colrow_use_variable()
        i = 0
        while True:
            print("fold_count",i,"= col_fold",self.col_fold_current, "row_fold" , self.row_fold_current)
            #Cycle11.19 #self.Systolic_preload_Weight()
            self.cycle_move_data += self.row_use #Cycle11.19 
            #Cycle11.19 #self.Systolic_Forward_WS_one_folds()
            self.cycle_move_data += self.Data_SRAM.shape[0] #Cycle11.19
            self.cycle_calculation_in_PE_unit += self.Data_SRAM.shape[0] #Cycle11.19
            #Cycle11.19 #self.make_square_Forward_WS_result()
            #Cycle11.19 #self.store_Forward_WS_result()
            #Cycle11.19 #self.temp_result = np.full((self.dim_col+self.dim_row+self.num_of_Activation_width, max(self.dim_col,self.dim_row)),0.)# "11.01"
            #Cycle11.19 #self.go_result = np.full(((self.num_of_Activation_width, max(self.dim_col,self.dim_row))),0.)# "11.01"
            i+=1
            if self.update_fold_variable_F_WS() == True: continue
            break   #self.Systolic_preload_Weight()# False
        self.cycle = 0
        self.cycle = self.cycle_calculation_in_PE_unit * BfpConfig.cycle_calculation_in_PE_unit + self.cycle_move_data * BfpConfig.cycle_move_data + self.cycle_store_result_in_OS * BfpConfig.cycle_store_result_in_OS

    ######################
    ####    2. for Backward WS
    def Backward_WS_Calculation(self):
        i = 0
        while True:
            print("fold_count",i,"= col_fold",self.col_fold_current, "row_fold" , self.row_fold_current)
            #Cycle11.19 #self.Systolic_preload_Weight()
            self.cycle_move_data += self.row_use #Cycle11.19
            #Cycle11.19 #self.Systolic_Backward_WS_one_folds()
            self.cycle_move_data += self.Gradient_SRAM.shape[0] #Cycle11.19
            self.cycle_calculation_in_PE_unit += self.Gradient_SRAM.shape[0] #Cycle11.19
            #Cycle11.19 #self.make_square_Backward_WS_result()
            #Cycle11.19 #self.store_Backward_WS_result()
            #Cycle11.19 #self.temp_result = np.full((self.dim_col+self.dim_row+self.num_of_Activation_width, max(self.dim_col,self.dim_row)),0.)# "11.01"
            #Cycle11.19 #self.go_result = np.full(((self.num_of_Activation_width, max(self.dim_col,self.dim_row))),0.)# "11.01"
            i+=1
            if self.update_fold_variable_B_WS() == True: continue
            break   ## False
        self.cycle = 0
        self.cycle = self.cycle_calculation_in_PE_unit * BfpConfig.cycle_calculation_in_PE_unit + self.cycle_move_data * BfpConfig.cycle_move_data + self.cycle_store_result_in_OS * BfpConfig.cycle_store_result_in_OS


    def print_weight(self):
        print("Weight array")
        for r in range(self.dim_row):
            print("row: ", r, "\t" , end='')
            for c in range(self.dim_col):
                for g in range(self.group_size):
                    print(self.Unit_array[r][c].cont_weight.get_value(g), end = '')
                print("|", end = '')
            print()


####_______________________________________________________________________________________________________________________________________
#### Function: OS
####    Functions.py TEST 6
####_______________________________________________________________________________________________________________________________________
    def Systolic_Backward_OS_one_folds(self):
        ## expect_cycle은 정확해야 한다.
        expect_cycle = self.Gradient_SRAM.shape[0] #### TO_DO 
        base_col = int(self.col_fold_current * self.dim_col)
        base_row = int(self.row_fold_current * self.dim_row)
        #print("asdasdasdasd", self.col_use, base_col)
        #print("asdasdasdasd", self.row_use, base_row)
        for i in range (expect_cycle):
            ## cycle count
            self.cycle_move_data += 1
            input_A = self.Data_SRAM[i][base_col: base_col + self.col_use]
            padding = fp_container()
            for _ in range(self.dim_col-self.col_use):
                input_A = np.append(input_A, [padding], axis = 0)
            self.bottom_side_entrance = input_A
            
            input_G = self.Gradient_SRAM[i][base_row: base_row + self.row_use]
            #print(input_G.shape)
            padding = fp_container()
            for _ in range(self.dim_row-self.row_use):
                input_G = np.append(input_G, [padding], axis = 0)
            
            # for k in range(self.dim_row):
            #     for g in range(self.group_size):
            #         print(input_G[k].get_value(g), end ='')
            #     print("|", end='')
            # print()
            self.left_side_entrance = input_G


            self.Systolic_Array_1cycle_calculation()
        for i in range (self.dim_row + self.dim_col):
            ##cycle count
            self.cycle_move_data += 1

            padding = fp_container()
            for i in range(self.dim_row):
                self.left_side_entrance = np.append(self.left_side_entrance, [padding], axis = 0)
            for i in range(self.dim_col):
                self.bottom_side_entrance = np.append(self.bottom_side_entrance, [padding], axis = 0)
            self.Systolic_Array_1cycle_calculation()

    
    def store_OS_result(self):
        base_col_addr = int(self.col_fold_current * self.dim_col)
        base_row_addr = int(self.row_fold_current * self.dim_row)
        ##fold에 따라 해당 sram에 저장 (여러 사이클걸린다. 한줄씩 뺴야함.)
        for i in range(self.row_use):
            self.cycle_store_result_in_OS += 1
            for j in range(self.col_use):
               self.return_result[base_col_addr + j][base_row_addr + i] = self.Unit_array[i][j].accumulator
               self.Unit_array[i][j].accumulator = 0
       
    def Backward_OS_Calculation(self):
        self.update_colrow_use_variable()
        #array_W = np.full((self.dim_row, self.dim_row), 0.0)
        i = 0
        while True:
            print("fold_count",i,"= col_fold",self.col_fold_current, "row_fold" , self.row_fold_current)
            i+=1
            #Cycle11.19 #self.Systolic_Backward_OS_one_folds()
            self.cycle_move_data += (self.Gradient_SRAM.shape[0] + self.dim_row + self.dim_col) #Cycle11.19
            self.cycle_calculation_in_PE_unit += (self.Gradient_SRAM.shape[0] + self.dim_row + self.dim_col) #Cycle11.19
            #Cycle11.19 #self.store_OS_result()
            self.cycle_store_result_in_OS += self.row_use #Cycle11.19
            if self.update_fold_variable_F_WS() == True: continue
            break   ## False
        self.cycle = 0
        self.cycle = self.cycle_calculation_in_PE_unit * BfpConfig.cycle_calculation_in_PE_unit + self.cycle_move_data * BfpConfig.cycle_move_data + self.cycle_store_result_in_OS * BfpConfig.cycle_store_result_in_OS




    def clean_return_result(self):
        self.return_result =np.full((self.result_height, self.result_width),0.)