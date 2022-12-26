Fast_simulator.code
====================
1. Input
> ###### Activation(Width ,Hight, channel, B) (현재까진 B=1)
> ###### Weight(Width, Hight, channel, num_filt)
> ###### True_Output (Width, Hight, Channel, B)     
2. Output
> ###### 3가지 연산의 결과 (Output, Gradient_Activation, Gradient_Weight)
3. Simulator Overview
> ###### Froward_WS = Activation(+padding) --> precision_decision <A,W> --(im2col)--> Systolic_Class(시뮬레이션) --> 결과(Output) --> Gradient_Output 계산 (Output - True_Output) 
> ###### Backward WS = Gradient_Output(+padding) -->  precision_decision <G_O,W> --(im2col)--> Systolic_Class(시뮬레이션) --> 결과(Gradient_Activation)
> ###### Backward OS = Activation_T(+padding) --> precision_decision <A_T, G_O> --(im2col)--> Systolic_Class(시뮬레이션) --> 결과(Gradient_Weight)
