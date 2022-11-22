class BfpConfig:
    group_size = 16
    dim_col = 128
    dim_row = 128
    strides = 1

    mantissa_bit = 4
    # bfp_M_Bit = 8
    # chunk_size_to_sort = 1024
    check_sram = False

    #cycle
    cycle_move_data = 1
    cycle_calculation_in_PE_unit = 4
    cycle_store_result_in_OS = dim_row



class PrecisionFlag:
    FP = 0
    BFP = 1