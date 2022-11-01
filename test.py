import numpy as np
import math
def im2col(X, filters, stride=1, pad=0):
    n, c, h, w = X.shape
    n_f, _, filter_h, filter_w = filters.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # add padding to height and width.
    in_X = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values =0)
    out = np.zeros((n, c, filter_h, filter_w, out_h, out_w))

    for h in range(filter_h):
        h_end = h + stride * out_h
        for w in range(filter_w):
            w_end = w + stride * out_w
            out[:, :, h, w, :, :] = in_X[:, :, h:h_end:stride, w:w_end:stride]

    out = out.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return out








if __name__ == "__main__":
    mode = input()
    if(mode != "Backward_OS" and mode != "Forward_WS" and mode != "Backward_WS"):
        print("input should be 'Backward_OS' or 'Forward_WS' or 'Backward_WS'")
        quit()
    print("FMAC simulator start with mode" , mode, ".")
######## hard_setting #########################################################################################
    dim_col = 4
    dim_row = 4
    strides = 1
    group_size = 16
    num_of_exp_plus_man = 1+ group_size
    mantissa_bit = 8 # "11.01"
    round1 = int(mantissa_bit/2)
    round2 = int(mantissa_bit/2)
    #w_W는 홀수
    aC = 2
    aB = 2
    aN = 2
    d_H = 14
    d_W = 14
    w_H = 7
    w_W = 7


    if(mode != "Backward_OS"):
        num_data1 = aB
        num_h1 = d_H
        num_w1 = d_W
        num_data2 = aN
        num_h2 = w_H
        num_w2 = w_W 
        num_ch12 = aC #common
        result_size = np.zeros(2)
    elif(mode == "Backward_OS"):
        num_data1 = aC
        num_h1 = d_H
        num_w1 = d_W
        num_data2 = aN
        num_h2 = d_H
        num_w2 = d_W 
        num_ch12 = aB #common
        W_height = w_H
        W_width = w_W
        result_size = np.array([W_width, W_height]) #OS에서만
    matrix_1 = np.full((num_data1,num_ch12,num_h1,num_w1),1)
    if(mode == "Backward_OS"):
        matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),1)
    elif(mode == "Forward_WS" or mode == "Backward_WS"):
        matrix_2 = np.full((num_data2,num_ch12,num_h2,num_w2),1)


    print('Filters:', matrix_2.shape)

    stride = 1
    pad = math.ceil((w_W-1)/2)
    X_col = im2col(matrix_1, matrix_2, stride=stride, pad=pad)

    n, c, h, w = matrix_1.shape
    n_f, _, filter_h, filter_w = matrix_2.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    out = np.matmul(X_col, matrix_2.reshape(n_f, -1).T)
    out = out.reshape(n, out_h, out_w, n_f)
    out = out.transpose(0, 3, 1, 2)

    print('Output:', out.shape)
    print(out)
   