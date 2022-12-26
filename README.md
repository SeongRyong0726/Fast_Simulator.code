Fast_simulator.code
====================
TO run the code.
<run_sim.py> : this is start from 4D fp array. caculation is only for fp not BFP, yet.

1. write "python run_sim.py" on cmd
2. write one of "Backward_OS", "Forward_WS", or "Backward_WS" which you want
3. done

<run_sim_cycle.py> : this is for get cycle of one layer caculation. TO run this. you need files @ see run_sim_cycle.py#L56
this simulation is start from 2D GEMM input.

1. write "python run_sim_cycle.py" on cmd
2. if "right file set + right path", it may run 
3. done (get cycle)

