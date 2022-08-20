Fast_simulator.code
====================
Class "fMAC"
======================
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
>#### fp_generator(self, exp, mantissa) $$$$
Class "Systolic-array"
======================
init
----
#### System information
>#### self.mode         {"Forward_WS", "Backward_WS", "Backward_OS"}
>#### self.group_size
>#### self.dim_row, self.dim_col
>#### self.Unit
>#### self.Unit_array
>#### self.Data_SRAM      @numpy
>#### self.Weight_SRAM    @numpy
>>#### self.bottom_side_entrance @load
>>#### self.up_side_entrance     @store
>#### self.Gradient_SRAM  @numpy
>>#### self.left_side_entrance   @load
>>#### self.right_side_entrance  @store
#### fold Variable
>#### self.col_total,        self.row_total
>#### self.col_fold_current, self.row_fold_current
>#### self.col_fold_end,     self.row_fold_end
#### Process Variable
>#### self.col_use, self.row_use
>#### self.target_count (for OS type)
>#### self.num_of_Activation_width, self.WS_Output_height
#### trace variable
>#### self.cycle
>#### self.util
>#### self.temp_result    @WS: 기울어진 partial결과
>#### self.go_result      @WS: 직사각형 partial결과
>#### self.return_result  @WS, OS: 진짜 저장소
        
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
> #### b. store_OS_result(self)

## Weight Stationary
#### 1. Systolic_preload_Weight(self)
> #### preload_Weight_row_by_row(self, row_num, col_num) (for 1 FMAC)

### Weight stationary(Forward)
#### 1. Forward_WS_Calculation(self)
> #### a. Systolic_Forward_WS_one_folds(self)
> #### b. make_square_Forward_WS_result(self)
> #### c. store_Forward_WS_result(self)


### Weight stationary(Backward)
#### 1. Forward_WS_Calculation(self)
> #### a. Systolic_Backward_WS_one_folds(self)
> #### b. make_square_Backward_WS_result(self)
> #### c. store_Backward_WS_result(self)

