"""
    iPEPS input file 

    Authoer : Wangwei Lan
"""


#!
#!
#!           Load packages 
#!
#!
using MKL
using IterativeSolvers
using TensorOperations
using LinearAlgebra
using HDF5
using TSVD
using Revise
using Plots


#!
#!
#! include files 
#!
#!
include("./src/all.jl")


#! enable cache
enable_cache(maxrelsize=0.30)





#!
#!
#!  plot detail
#!
#!
EnergyPlot = plot()
Magnetization = Array{Float64}(undef,0)
Energy = 0.01



for j = 2.90:0.05:3.00




    #!
    #!
    #!      Set calculation details
    #!
    #!

    param = SetParam(
        
                    # spin = 1/2
                    spin     = 0.5,                                                    
                
                    # physical dimension, dphy = 2*spin+1
                    dphy      = 2,                                                      
                    
                    # Dstep : at certain steps, virtual dimension change to correspoinding ones at Dlink 
                    Dstep     = [14,  41,  91,  121,  301,  400,  500,  Inf],           

                    # Dlink : virtual dimension 
                    Dlink     = [3,   3,   3,    3,    3,    3,     3,    3],           

                    # chistep : at certain steps, boundary dimension change to correspoinding ones at chimax
                    chistep   = [31,  41,  61,  81,  91,    151, 201,   251,   Inf],    
                    
                    # chimax : maximum boundary dimension
                    chimax    = [20,  20,  20,  20,  20,    20,   20,    20,     20],   

                    # HamiType : Hamiltonian type 
                    HamiType  = "Ising",   
                    
                    # step: at certain steps, tao change to correspoinding ones at tao 
                    step      = [35,   61,  121,     201,  221,    251, 301, 351,    501,     851,     1001,    550,     650,    750,      800,     650,     Inf],
                    
                    # tao ； tao at certain steps 
                    tao       = [0.05, 0.05,0.01,     0.005, 0.005,  0.005,0.005, 0.005,  0.005,  0.005,  0.005,  0.05,    0.001,   0.001,    0.001,   0.001,  0.005],
                    
                    # MaxLoop : maximum loop numbers
                    MaxLoop   = 500,
                    
                    # Tol : truncation error during update step 
                    Tol       = 1.0e-9,
                    
                    # MoverType ： different ways to perform "move" in iPEPS
                        #  SingleLayer : Sequential way
                        #  2x2UnitCell
                        #  2x1UnitCell
                    MoverType = "SingleLayer",
                    #MoverType ="2x1UnitCell",                    


                    # ProjectorFunction : projector types 
                        # ConstructProjector_Unitary : 2x2UnitCell, use unitary as projectors 
                        # ConstructProjector_Philippe_Corboz: use Philippe_Corboz way to construct projectors
                        # ConstructProjector_Unitary_reduced_env : 2x1UnitCell, using a smaller environment to construct projector 
                    #ProjectorFunction = ConstructProjector_Unitary,
                    #ProjectorFunction = ConstructProjector_Philippe_Corboz,
                    ProjectorFunction = ConstructProjector_Unitary_reduced_env,

                    # SimpleUpdateOn : whether Simple Update is on or off
                    SimpleUpdateOn = "true",
                    
                    FullUpdateOn   = "true",
                    
                    # Simple update accuracy
                    SimpTol        = 1.0e-12,
                    
                    MaxSimpLoop    = 1000,
                    
                    # only for quantum Ising Model 
                    Magnetism      = j)



    #!
    #!
    #!
    #!
    paramc = deepcopy(param);
    
    
    
    #!
    #!
    #!  run iPEPS algorithm
    #!
    #!
    println("Start Calculation: ")
    iPEPS(param,paramc);
    #magnetism = compute_physical(param,Operator="Sz")
    
    
    
    #! save file 
    @save "iPEPS_D_3_full_update_2x1UnitCell_Sequential_j_$(j)_chi_30.jld2" param

end







x = 1
