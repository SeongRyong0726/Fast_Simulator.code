import numpy as np
import math
from fMAC import fMAC


class Systolic_array():                                                                         ## OS       #fold       #fold       ## WS                   
    def __init__(self, dim_col, dim_row, opUnit, groupsize, mode, Input_A, Input_W, Input_G, OS_result_shape, col_total, row_total, num_of_Activation_width):
        ####    System information
        self.mode = mode                    #{"Forward_WS", "Backward_WS", "Backward_OS"}
        self.group_size = groupsize         #

        self.dim_row = dim_row
        self.dim_col = dim_col
        self.Unit = opUnit
        self.Unit_array = list()
            ####    SRAM
        self.Data_SRAM = Input_A                ##numpy
        self.Weight_SRAM = Input_W              ##numpy
        self.Gradient_SRAM = Input_G            ##numpy
                ####    Systolic Array fMAC 가기 직전 entrance
        self.bottom_side_entrance = np.zeros(self.dim_col)  # Weight, Data
        self.left_side_entrance = np.zeros(self.dim_row)    # Gradient
                ####    Systolic Array SRAM 가기 직전 entrance
        self.up_side_entrance = np.zeros(self.dim_col) 
        self.right_side_entrance = np.zeros(self.dim_row)
        #### fold
        self.col_total = col_total          
        self.row_total = row_total
        self.col_fold_end = math.ceil(self.col_total/dim_col)
        self.row_fold_end = math.ceil(self.row_total/dim_row)
        self.col_fold_current = 0
        self.row_fold_current = 0
        ####    Process Variable
        self.col_use = self.dim_col
        self.row_use = self.dim_row

        self.update_colrow_use_variable()
            ####    for OS type
        if(self.mode =="Backward_OS"):
            self.OS_result_height = Input_A.shape[1] #C,H,w
            self.OS_result_width = Input_G.shape[1]  #N
            ####    for WS type
        self.num_of_Activation_width = num_of_Activation_width

        ####    trace_변수
        self.cycle = 0              ##count_cycle
        self.util = 0
        if(self.mode == "Forward_WS" or self.mode == "Backward_WS"):
            self.temp_result = np.full((self.dim_col+self.dim_row+self.num_of_Activation_width, max(self.dim_col,self.dim_row)),0) 
            self.go_result = np.full(((self.num_of_Activation_width, max(self.dim_col,self.dim_row))),0)

        if(self.mode == "Forward_WS"):
            result_height= Input_A.shape[0] 
            result_width = Input_W.shape[0]
        elif (self.mode=="Backward_WS"):
            result_height= Input_G.shape[0] 
            result_width = Input_W.shape[1]
        elif (self.mode=="Backward_OS"):
            result_height = self.OS_result_height 
            result_width = self.OS_result_width  
        self.return_result =np.full((result_height, result_width),0)   #### TO_DO
    

        ####    Array에 Unit(fMAC)을 채우고 초기화 // 가로(cols) 세로(rows)
        for i in range (self.dim_row):
            self.Unit_array.append(list())
            for j in range (self.dim_col):
                self.Unit_array[i].append(opUnit(i, j, groupsize, mode))


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
            self.row_use = self.row_total % self.dim_row
        if(self.col_fold_current == (self.col_fold_end -1)):
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

    def fMAC_operand_pass_right(self, row_num=0, col_num=0):                                        #operand
        if row_num>=0 and row_num<=self.dim_row-1       and col_num > 0 and col_num<self.dim_col:  #case) fMAC-->fMAC  
            exp_left, mantissa_left = self.Unit_array[row_num][col_num-1].get_operand_from_left()    #이전꺼에서 받아서 넣는다
            self.Unit_array[row_num][col_num].load_operand_on_left_entrance(exp_left, mantissa_left)
        elif row_num>=0 and row_num<=self.dim_row-1     and col_num==0:                              #case) SRAM-->fMAC (load)  
            exp_up, mantissa_up = self.separate_exp_man(self.left_side_entrance[row_num])
            self.Unit_array[row_num][col_num].load_operand_on_left_entrance(exp_up, mantissa_up)                                                                                 
        else:
            print("error: function 'fMAC_operand_pass_right'")

    def fMAC_operand_pass_up(self, row_num=0, col_num=0):                                         #operand
        if row_num>0 and row_num<self.dim_row and col_num >= 0 and col_num<=self.dim_col-1:          #case) fMAC-->fMAC  
            exp_up, mantissa_up = self.Unit_array[row_num-1][col_num].get_operand_from_down()
            self.Unit_array[row_num][col_num].load_operand_on_down_entrance(exp_up, mantissa_up)
        elif row_num==0                         and col_num >= 0 and col_num<=self.dim_col-1:        #case) SRAM-->fMAC (load)
            exp_up, mantissa_up = self.separate_exp_man(self.bottom_side_entrance[col_num])
            self.Unit_array[row_num][col_num].load_operand_on_down_entrance(exp_up, mantissa_up)                                                                                   

        else:
            print("error: function 'fMAC_operand_pass_up'")

    def fMAC_result_pass_right(self, row_num=0, col_num=0):                                         #result (부분합)
        if row_num>=0 and row_num<=self.dim_row-1   and col_num >= 0 and col_num<self.dim_col-1:          #case) fMAC-->fMAC  
            result = self.Unit_array[row_num][col_num].get_result_to_right()
            self.Unit_array[row_num][col_num+1].load_result_on_acc(result)
        elif row_num>=0 and row_num<=self.dim_row-1 and col_num==self.dim_col-1:                       #case) SRAM직전 fMAC-->SRAM
            ####  TO DO ####                                                                        # SRAM에 값 넣기 SRAM의 base addr//진행에 따라 시작점이 다르므로 pointer를 지정해야하마
            self.right_side_entrance[row_num] = self.Unit_array[row_num][col_num].get_result_to_right()
        else:
            print("error: function 'fMAC_result_pass_right'")

    def fMAC_result_pass_up(self, row_num=0, col_num=0):
        if row_num>=0 and row_num<self.dim_row-1 and col_num >= 0 and col_num<=self.dim_col-1:       #case) fMAC-->fMAC  
            result = self.Unit_array[row_num][col_num].get_result_to_up()
            self.Unit_array[row_num+1][col_num].load_result_on_acc(result)
        elif row_num==self.dim_row-1            and col_num >= 0 and col_num<=self.dim_col-1:       #case) SRAM직전 fMAC-->SRAM
            ####  TO DO ####                                                                        # SRAM에 값 넣기
            self.up_side_entrance[col_num] = self.Unit_array[row_num][col_num].get_result_to_up()
        else:
            print("error: function 'fMAC_result_pass_right'")


####_______________________________________________________________________________________________________________________________________
#### Function: Systolic_Array_1cycle_calculation 함수를 사용하면 1cycle에 걸친 Systolic Array 내부의 동작을 수행한다.
#### Functions.py TEST3
####_______________________________________________________________________________________________________________________________________

    def Systolic_Array_1cycle_calculation(self):
        ####    1. 내부에서 연산 수행
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
            exp, mantissa = self.Unit_array[row_num-1][col_num].get_Weight_from_down()
            self.Unit_array[row_num][col_num].load_Weight_value(exp, mantissa)
        elif row_num==0                         and col_num >= 0 and col_num<=self.dim_col-1:       #case) SRAM-->fMAC (load)
            set = self.bottom_side_entrance[col_num]
            exp = set[0]
            mantissa = set[1:self.group_size+1]
            self.Unit_array[row_num][col_num].load_Weight_value(exp, mantissa)
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
                self.Unit_array[i][j].weight_exponant = np.zeros(1)
                self.Unit_array[i][j].weight_mantissa = np.zeros(self.group_size)
        ####    값 넣기        
        for k in range (self.row_use):
            temp_input_W = self.Weight_SRAM[base_row + k][base_col: base_col + self.col_use]
            input_W = temp_input_W
            padding = np.full((self.group_size+1,), 0)
            for i in range(self.dim_col-self.col_use):
                input_W = np.append(input_W, [padding], axis = 0)
            #input_W = np.pad(temp_input_W, (0,self.dim_col-self.col_use), 'constant', constant_values = 0)
            self.bottom_side_entrance = input_W
            
            ##count_cycle
            self.cycle +=1 
            ##self.Systolic_Array_1cycle_calculation()
            for a in range(self.dim_row):
                for b in range(self.dim_col):
                    self.preload_Weight_row_by_row(self.dim_row-a-1,b)
                
                ####    for TEST3
                # for row in range(self.dim_row):
                #     for col in range(self.dim_col):
                #         array_W[row][col] = self.Unit_array[row][col].weight_exponant
                # print(array_W)
            
    ####    1. for Forward WS
    def Systolic_Forward_WS_one_folds(self):
        ####    for TEST3
        array_Flow = np.full((self.dim_row, self.dim_col),0)
        array_Flow1 = np.full((self.dim_row, self.dim_col),0)

        base_col = int(self.col_fold_current * self.dim_col)
        base_row = int(self.row_fold_current * self.dim_row)
        for i in range (self.num_of_Activation_width):
            ##count_cycle
            self.cycle += 1
            temp_input_A = self.Data_SRAM[i][base_col: base_col + self.col_use][:]
            input_A = np.pad(temp_input_A, ((0,self.dim_col-self.col_use),(0,0)), 'constant', constant_values = 0)
            self.bottom_side_entrance = input_A
            self.Systolic_Array_1cycle_calculation()
            ####    entrance에 저장된 값 result에 저장
            
            self.temp_result[i][0:self.dim_row] = self.right_side_entrance[0:self.dim_row]
            #### for TEST3
            for row in range(self.dim_row):
                for col in range(self.dim_col):
                    array_Flow[row][col] = self.Unit_array[row][col].exponant_from_down
                    array_Flow1[row][col] = self.Unit_array[row][col].result_to_right
            #print("Cycle:" , i)
            #print(array_Flow)
            #print(array_Flow1)
        for i in range (self.dim_col + self.row_use):
            ##count_cycle
            self.cycle += 1
            self.bottom_side_entrance = np.full((self.dim_col,self.group_size+1),0)
            self.Systolic_Array_1cycle_calculation()
            ####    entrance에 저장된 값 result에 저장
            
            self.temp_result[i+self.num_of_Activation_width][0:self.dim_row] = self.right_side_entrance[0:self.dim_row]
            #### for TEST3
            for row in range(self.dim_row):
                for col in range(self.dim_col):
                    array_Flow[row][col] = self.Unit_array[row][col].exponant_from_down
            #print("Cycle:!!" , i+self.num_of_Activation_width)
            #print(array_Flow)


    ####    2. for Backward WS
    def Systolic_Backward_WS_one_folds(self):
        ####    for TEST3
        array_Flow = np.full((self.dim_row, self.dim_col),0)
        base_col = int(self.col_fold_current * self.dim_col)
        base_row = int(self.row_fold_current * self.dim_row)
        for i in range (self.num_of_Activation_width):
            ##count_cycle
            self.cycle += 1
            temp_input_G = self.Gradient_SRAM[i][base_row: base_row + self.row_use][:]
            input_G = np.pad(temp_input_G, ((0,self.dim_row-self.row_use),(0,0)), 'constant', constant_values = 0)
            self.left_side_entrance = input_G
            self.Systolic_Array_1cycle_calculation()
            self.temp_result[i][0:self.dim_col] = self.up_side_entrance[0:self.dim_col]
            #### for TEST3
            for row in range(self.dim_row):
                for col in range(self.dim_col):
                    array_Flow[row][col] = self.Unit_array[row][col].exponant_from_left
            #print("Cycle:" , i)
            #print(array_Flow)

        for i in range (self.dim_row + self.col_use):
            ##count_cycle
            self.cycle += 1
            self.left_side_entrance = np.full((self.dim_row,self.group_size+1),0)
            self.Systolic_Array_1cycle_calculation()
            self.temp_result[i+self.num_of_Activation_width][0:self.dim_col] = self.up_side_entrance[0:self.dim_col]
            #### for TEST3
            for row in range(self.dim_row):
                for col in range(self.dim_col):
                    array_Flow[row][col] = self.Unit_array[row][col].exponant_from_left
            #print("Cycle:!!" , i+self.num_of_Activation_width)
            #print(array_Flow)


    ######################
    ####    기울어진 결과값을 저장하기 좋게 정 사각형으로(이 결과값은 전체가 아닌 일부이다)
    ####    1. for Forward WS
    def make_square_Forward_WS_result(self):
        base_col = self.dim_col
        for i in range(self.num_of_Activation_width):
            for j in range(self.row_use):
                self.go_result[i][j] = self.temp_result[base_col + i + j][j]

    ####    2. for Backward WS
    def make_square_Backward_WS_result(self):
        base_row = self.dim_row
        for i in range(self.num_of_Activation_width):
            for j in range(self.col_use):
                self.go_result[i][j] = self.temp_result[base_row + i + j][j]


    ######################
    ####    WS결과를 최종 장소에 저장
    ####    1. for Forward WS
    def store_Forward_WS_result(self):
        base_row = self.row_fold_current * self.dim_row 
        self.return_result[0:self.num_of_Activation_width , base_row : base_row + self.row_use] += self.go_result[0:self.num_of_Activation_width , 0:self.row_use]
    
    ####    2. for Backward WS
    def store_Backward_WS_result(self):
        base_col = self.col_fold_current * self.dim_col 
        self.return_result[0:self.num_of_Activation_width , base_col : base_col + self.col_use] += self.go_result[0:self.num_of_Activation_width , 0:self.col_use]



    ######################
    ####    1. for Forward WS
    def Forward_WS_Calculation(self):
        while True:
            self.Systolic_preload_Weight()
            self.Systolic_Forward_WS_one_folds()
            self.make_square_Forward_WS_result()
            self.store_Forward_WS_result()

            if self.update_fold_variable_F_WS() == True: continue
            break   ## False

    ######################
    ####    2. for Backward WS
    def Backward_WS_Calculation(self):
        while True:
            self.Systolic_preload_Weight()
            self.Systolic_Backward_WS_one_folds()
            self.make_square_Backward_WS_result()
            self.store_Backward_WS_result()

            if self.update_fold_variable_B_WS() == True: continue
            break   ## False





####_______________________________________________________________________________________________________________________________________
#### Function: OS
####    Functions.py TEST 6
####_______________________________________________________________________________________________________________________________________
    def Systolic_Backward_OS_one_folds(self):
        ## expect_cycle은 정확해야 한다.
        expect_cycle = self.Gradient_SRAM.shape[0] #### TO_DO 
        base_col = int(self.col_fold_current * self.dim_col)
        base_row = int(self.row_fold_current * self.dim_row)
        
        for i in range (expect_cycle):
            ## cycle count
            self.cycle += 1

            temp_input_A = self.Data_SRAM[i][base_col: base_col + self.col_use][:]
            input_A = np.pad(temp_input_A, ((0,self.dim_col-self.col_use),(0,0)), 'constant', constant_values = 0)
            self.bottom_side_entrance = input_A
            temp_input_G = self.Gradient_SRAM[i][base_row: base_row + self.row_use][:]
            input_G = np.pad(temp_input_G, ((0,self.dim_row-self.row_use),(0,0)), 'constant', constant_values = 0 )
            self.left_side_entrance = input_G


            self.Systolic_Array_1cycle_calculation()
        for i in range (self.dim_row + self.dim_col):
            ##cycle count
            self.cycle += 1

            self.left_side_entrance = np.full((self.dim_row, self.group_size + 1),0)
            self.bottom_side_entrance = np.full((self.dim_col, self.group_size + 1),0)
            self.Systolic_Array_1cycle_calculation()

    
    def store_OS_result(self):
        base_col_addr = int(self.col_fold_current * self.dim_col)
        base_row_addr = int(self.row_fold_current * self.dim_row)
        ##fold에 따라 해당 sram에 저장 (여러 사이클걸린다. 한줄씩 뺴야함.)
        for i in range(self.row_use):
            self.cycle += 1
            for j in range(self.col_use):
               self.return_result[base_col_addr + j][base_row_addr + i] = self.Unit_array[i][j].accumulator
               self.Unit_array[i][j].accumulator = 0
       
    def Backward_OS_Calculation(self):
        while True:
            self.Systolic_Backward_OS_one_folds()
            self.store_OS_result()
            
            if self.update_fold_variable_F_WS() == True: continue
            break   ## False

