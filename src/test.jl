using TensorOperations
function TestFunc(A::Array{Float64})
    @tensor test[] := A[1,2,3,4,5]*A[1,2,3,4,5]
end
include("../src/all.jl")
include("./test1.jl")
param = SetParam(spin = 0.5,
                    dphy = 2,
                    Dlink = 2,
                    Dp = 2,
                    chimax = 4,
                    HamiType = "Heisenberg",
                    step = [20,50,60,80,Inf],
                    tao = [0.02,0.02,0.02,0.02,0.02],
                    MaxLoop = 120,
                    Tol = 1.0e-7,
                    ProjectorFunction=ConstructProjectorIsometry,
                    #ProjectorFunction=ConstructProjector2,
                    SimpleUpdateOn = "true",
                    SimpTol = 1.0e-10,
                    MaxSimpLoop = 500);
println("Start")
A = param.UnitCell[1,1]
B = param.UnitCell[1,2]
hr = param.Hamiltonian
C1 = param.CornerEnvironment[1,1][1]
#@tensor test[] := C1[1,2]*C1[1,2]
C2 = param.CornerEnvironment[1,1][2]
C3 = param.CornerEnvironment[1,2][3]
C4 = param.CornerEnvironment[1,2][4]
E1 = param.EdgeEnvironment[1,2][1]
E2 = param.EdgeEnvironment[1,1][1]
E3 = param.EdgeEnvironment[1,1][2]
E5 = param.EdgeEnvironment[1,1][3]
E6 = param.EdgeEnvironment[1,2][3]
E8 = param.EdgeEnvironment[1,2][4]

@tensor Numerator[] := C1[13,14]*C2[19,20]*C3[10,9]*C4[1,2]*A[28,16,15,25,22]*A[27,18,17,24,21]*
                    B[8,25,3,4,12]*B[7,24,5,6,11]*E1[2,5,3,23]*E2[23,17,15,13]*E3[14,18,16,19]*E5[20,21,22,26]*E6[26,11,12,10]*E8[9,6,4,1]*hr[7,27,8,28]
@tensor Denominator[] := C1[12,13]*C2[19,20]*C3[9,8]*C4[1,2]*A[18,15,14,25,22]*A[18,17,16,24,21]*B[7,25,3,4,11]*
                B[7,24,5,6,10]*E1[2,5,3,23]*E2[23,16,14,12]*E3[13,17,15,19]*E5[20,21,22,26]*E6[26,10,11,9]*E8[8,6,4,1]
energy1 = Numerator[1]/Denominator[1]
println(energy1)
#
println("Start")
ComputeEnergyTest(A,B,C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,hr)



#A = rand(2,2,2,2,2);
#TestFunc(A)