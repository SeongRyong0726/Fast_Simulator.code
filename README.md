Fast_simulator.code
====================
Class "fMAC"
======================
###### 기본적인 fMAC의 동작을 수행, 값을 저장하는 레지스터를 변수로 + 값을 연산하고 flow시키는 동작을 수행.

init
----
#### ID : self.col_num, self.row_num
#### self.group_size
#### self.mode
#### operand
>#### self.exponant_from_left, self.mantissa_from_left                  
>#### self.exponant_from_down, self.mantissa_from_down
>#### self.weight_exponant, self.weight_mantissa
#### result
>#### self.accumulator
>#### self.result_to_right, self.result_to_up

methods
-------
#### flow operand
>#### get_operand_from_left(self), get_operand_from_down(self)
>#### load_operand_on_left_entrance(self, exp, mantissa), load_operand_on_down_entrance(self, exp, mantissa)
>#### get_Weight_from_down(self), load_Weight_value(self, exp, mantissa)
#### flow_result
>#### get_result_to_right(self), get_result_to_up(self)
>#### load_result_on_acc(self, result)
#### operation
>#### operation(self)
>#### fp_generator(self, exp, mantissa) $$$$미완성 


Class "Systolic-array"
======================
init
----
#### System information
>#### self.mode                         {"Forward_WS", "Backward_WS", "Backward_OS"}
>#### self.group_size                   {type:int}
>#### self.dim_row, self.dim_col        {type:int, int}
>#### self.Unit                         {type:Class}
>#### self.Unit_array                   {type:}
>#### self.Data_SRAM                    {type:numpy array (width(row), hight(col), groupsize + 1)}   
>#### self.Weight_SRAM                  {type:numpy array (width(row), hight(col), groupsize + 1)} 
>>#### self.bottom_side_entrance @load  {type:numpy array (width(row), 1, groupsize + 1)}
>>#### self.up_side_entrance     @store {type:numpy array (width(row), 1, groupsize + 1)}
>#### self.Gradient_SRAM                {type:numpy array (width(col), hight(row), groupsize + 1)}
>>#### self.left_side_entrance   @load  {type:numpy array (width(col), 1, groupsize + 1)}
>>#### self.right_side_entrance  @store {type:numpy array (width(col), 1, groupsize + 1)}
#### fold Variable
>#### self.col_total,        self.row_total             {type:int, int}
>#### self.col_fold_current, self.row_fold_current      {type:int, int}
>#### self.col_fold_end,     self.row_fold_end          {type:int, int}
#### Process Variable
>#### self.col_use, self.row_use                                          {type:int, int}
>#### self.target_count (for OS type)                                     {type:int, int} DP 부분합개수
>#### self.num_of_Activation_width, self.WS_Output_height (for WS type)   {type:int, int}
#### trace variable
>#### self.cycle                {type:int}
>#### self.util                 {type:int}
>#### self.temp_result    @WS: 기울어진 partial 결과
>>##### Forward WS: (dim_row, expect_cycle)
>>##### Backward WS : (dim_col, expect_cycle)
>#### self.go_result      @WS: 직사각형 partial 결과
>>##### Forward WS: (row_use, num_of_Activation_width)
>>##### Backward WS : (col_use, num_of_Activation_width)
>#### self.return_result  @WS, OS: entire 결과
        
methods
-------
## Update next "num_fold (col, row)" & "num_use (col, row)"
#### update_fold_variable_B_WS(self)
#### update_fold_variable_F_WS(self)
> #### update_colrow_use_variable(self)

## flow of operand, result of each fMAC
#### separate_exp_man(self, tuple)
#### fMAC_operand_pass_right(self, row_num=0, col_num=0)
#### fMAC_operand_pass_up(self, row_num=0, col_num=0)
#### fMAC_result_pass_right(self, row_num=0, col_num=0)
#### fMAC_result_pass_up(self, row_num=0, col_num=0):

## '1 cycle' Calculation
#### Systolic_Array_1cycle_calculation(self)


## Output stationary
#### 1. Backward_OS_Calculation(self)
##### Do calculation + store for all folds (iter Unit = one pair of fold)
> #### a. Systolic_Backward_OS_one_folds(self)
>> ###### expect_cycle동안 
>> ###### 한줄씩 넣고 계산(self.left_side_entrance, self.bottom_side_entrance)
>> ###### 각 fMAC의 accumulator에 저장
> #### b. store_OS_result(self)


## Weight Stationary
#### 1. Systolic_preload_Weight(self)
> #### preload_Weight_row_by_row(self, row_num, col_num) (for 1 FMAC)
### Weight stationary(Forward)
#### 1. Forward_WS_Calculation(self)
> #### a. Systolic_Forward_WS_one_folds(self)
>> ###### expect_cycle동안 
>> ###### 한줄씩 넣고 계산(self.left_side_entrance, self.bottom_side_entrance)
>> ###### (self.right_side_entrance, self.top_side_entrance)를 통해 저장
> #### b. make_square_Forward_WS_result(self)
> #### c. store_Forward_WS_result(self)
### Weight stationary(Backward)
#### 1. Forward_WS_Calculation(self)
> #### a. Systolic_Backward_WS_one_folds(self)
>> ###### expect_cycle동안 
>> ###### 한줄씩 넣고 계산(self.left_side_entrance, self.bottom_side_entrance)
>> ###### (self.right_side_entrance, self.top_side_entrance)를 통해 저장
> #### b. make_square_Backward_WS_result(self)
> #### c. store_Backward_WS_result(self)

