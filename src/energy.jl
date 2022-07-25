#=
"""
    ComputeEnergy from ComputeEnvironmentPlane
"""
function ComputeEnergy(A::Array{Float64},B::Array{Float64},Environment,hr)

    C1 = Environment[1,1].C1
    C2 = Environment[1,1].C2
    C3 = Environment[1,2].C3
    C4 = Environment[1,2].C4
    E1 = Environment[1,2].E1
    E2 = Environment[1,1].E1
    E3 = Environment[1,1].E2
    E5 = Environment[1,1].E3
    E6 = Environment[1,2].E3
    E8 = Environment[1,2].E4

    @tensor Numerator[] := C1[13,14]*C2[19,20]*C3[10,9]*C4[1,2]*A[28,16,15,25,22]*A[27,18,17,24,21]*B[8,25,3,4,12]*B[7,24,5,6,11]*
                        E1[2,5,3,23]*E2[23,17,15,13]*E3[14,18,16,19]*E5[20,21,22,26]*E6[26,11,12,10]*E8[9,6,4,1]*hr[7,27,8,28]
    @tensor Denominator[] := C1[12,13]*C2[19,20]*C3[9,8]*C4[1,2]*A[18,15,14,25,22]*A[18,17,16,24,21]*B[7,25,3,4,11]*
                    B[7,24,5,6,10]*E1[2,5,3,23]*E2[23,16,14,12]*E3[13,17,15,19]*E5[20,21,22,26]*E6[26,10,11,9]*E8[8,6,4,1]
    energy1 = Numerator[1]/Denominator[1]

    C1 = Environment[1,1].C1
    C2 = Environment[2,1].C2
    C3 = Environment[2,1].C3
    C4 = Environment[1,1].C4
    E2 = Environment[1,1].E1
    E3 = Environment[1,1].E2
    E4 = Environment[2,1].E2
    E5 = Environment[2,1].E3
    E7 = Environment[2,1].E4
    E8 = Environment[1,1].E4
    @tensor Denominator[] := C1[8,9]*C2[20,19]*C3[13,12]*C4[1,2]*E2[2,5,3,8]*E3[9,10,11,23]*E4[23,21,22,20]*E5[19,17,15,13]*E7[12,16,14,26]*
                        E8[26,6,4,1]*A[7,11,3,4,25]*B[18,22,25,14,15]*A[7,10,5,6,24]*B[18,21,24,16,17]
    @tensor Numerator[] := C1[9,10]*C2[20,19]*C3[14,13]*C4[1,2]*E2[2,5,3,9]*E3[10,11,12,23]*E4[23,21,22,20]*E5[19,18,16,14]*
                        E7[13,17,15,26]*E8[26,6,4,1]*A[8,12,3,4,25]*A[7,11,5,6,24]*B[28,22,25,15,16]*B[27,21,24,17,18]*hr[7,27,8,28]
    energy2 = Numerator[1]/Denominator[1]


    C1 = Environment[2,1].C1
    C2 = Environment[2,1].C2
    C3 = Environment[2,2].C3
    C4 = Environment[2,2].C4
    E1 = Environment[2,2].E1
    E2 = Environment[2,1].E1
    E4 = Environment[2,1].E2
    E5 = Environment[2,1].E3
    E6 = Environment[2,2].E3
    E7 = Environment[2,2].E4


    @tensor Numerator[] := C1[13,14]*C2[19,20]*C3[10,9]*C4[1,2]*B[28,16,15,25,22]*B[27,18,17,24,21]*A[8,25,3,4,12]*A[7,24,5,6,11]*
                        E1[2,5,3,23]*E2[23,17,15,13]*E4[14,18,16,19]*E5[20,21,22,26]*E6[26,11,12,10]*E7[9,6,4,1]*hr[7,27,8,28]
    @tensor Denominator[] := C1[12,13]*C2[19,20]*C3[9,8]*C4[1,2]*B[18,15,14,25,22]*B[18,17,16,24,21]*A[7,25,3,4,11]*
                    A[7,24,5,6,10]*E1[2,5,3,23]*E2[23,16,14,12]*E4[13,17,15,19]*E5[20,21,22,26]*E6[26,10,11,9]*E7[8,6,4,1]
    energy3 = Numerator[1]/Denominator[1]

    C1 = Environment[1,2].C1
    C2 = Environment[2,2].C2
    C3 = Environment[2,2].C3
    C4 = Environment[1,2].C4

    E1 = Environment[1,2].E1
    E3 = Environment[1,2].E2
    E4 = Environment[2,2].E2
    E6 = Environment[2,2].E3
    E7 = Environment[2,2].E4
    E8 = Environment[1,2].E4

    @tensor Denominator[] := C1[8,9]*C2[20,19]*C3[13,12]*C4[1,2]*E1[2,5,3,8]*E3[9,10,11,23]*E4[23,21,22,20]*E6[19,17,15,13]*E7[12,16,14,26]*
                        E8[26,6,4,1]*B[7,11,3,4,25]*A[18,22,25,14,15]*B[7,10,5,6,24]*A[18,21,24,16,17]
    @tensor Numerator[] := C1[9,10]*C2[20,19]*C3[14,13]*C4[1,2]*E1[2,5,3,9]*E3[10,11,12,23]*E4[23,21,22,20]*E6[19,18,16,14]*
                        E7[13,17,15,26]*E8[26,6,4,1]*B[8,12,3,4,25]*B[7,11,5,6,24]*A[28,22,25,15,16]*A[27,21,24,17,18]*hr[7,27,8,28]
    energy4 = Numerator[1]/Denominator[1]
    return energy1,energy2,energy3,energy4#(energy1+energy2+energy3+energy4)/4
end
=#





#"""
#    ComputeEnergy from ComputeEnvironmentPlane
#"""
function ComputePhysical(param::PEPSParam;Operator="Energy")
    
    A = param.UnitCell[1,1]
    B = param.UnitCell[1,2]
    if Operator == "Energy"
        hr = param.Hamiltonian
        Id = reshape(Matrix(1.0I,4,4),param.dphy,param.dphy,param.dphy,param.dphy)
    elseif Operator =="Mag"
        hr = reshape(kron(Matrix(1.0I,param.dphy,param.dphy),param.Sz),param.dphy,param.dphy,param.dphy,param.dphy)
        Id = reshape(Matrix(1.0I,param.dphy^2,param.dphy^2),param.dphy,param.dphy,param.dphy,param.dphy)
    end
    C1 = param.CornerEnvironment[1,1][1]
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
    #
    @tensor Denominator[] := C1[12,13]*C2[19,20]*C3[9,8]*C4[1,2]*A[18,15,14,25,22]*A[18,17,16,24,21]*B[7,25,3,4,11]*
                    B[7,24,5,6,10]*E1[2,5,3,23]*E2[23,16,14,12]*E3[13,17,15,19]*E5[20,21,22,26]*E6[26,10,11,9]*E8[8,6,4,1]
    energy1 = Numerator[1]/Denominator[1]

    #
    C1 = param.CornerEnvironment[1,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,1][3]
    C4 = param.CornerEnvironment[1,1][4]
    E2 = param.EdgeEnvironment[1,1][1]
    E3 = param.EdgeEnvironment[1,1][2]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E7 = param.EdgeEnvironment[2,1][4]
    E8 = param.EdgeEnvironment[1,1][4]
    @tensor Denominator[] := C1[8,9]*C2[20,19]*C3[13,12]*C4[1,2]*E2[2,5,3,8]*E3[9,10,11,23]*E4[23,21,22,20]*E5[19,17,15,13]*E7[12,16,14,26]*
                        E8[26,6,4,1]*A[7,11,3,4,25]*B[18,22,25,14,15]*A[7,10,5,6,24]*B[18,21,24,16,17]
    @tensor Numerator[] := C1[9,10]*C2[20,19]*C3[14,13]*C4[1,2]*E2[2,5,3,9]*E3[10,11,12,23]*E4[23,21,22,20]*E5[19,18,16,14]*
                        E7[13,17,15,26]*E8[26,6,4,1]*A[8,12,3,4,25]*A[7,11,5,6,24]*B[28,22,25,15,16]*B[27,21,24,17,18]*hr[7,27,8,28]
    energy2 = Numerator[1]/Denominator[1]

    C1 = param.CornerEnvironment[2,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[2,2][4]
    E1 = param.EdgeEnvironment[2,2][1]
    E2 = param.EdgeEnvironment[2,1][1]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]


    @tensor Numerator[] := C1[13,14]*C2[19,20]*C3[10,9]*C4[1,2]*B[28,16,15,25,22]*B[27,18,17,24,21]*A[8,25,3,4,12]*A[7,24,5,6,11]*
                        E1[2,5,3,23]*E2[23,17,15,13]*E4[14,18,16,19]*E5[20,21,22,26]*E6[26,11,12,10]*E7[9,6,4,1]*hr[7,27,8,28]
    @tensor Denominator[] := C1[12,13]*C2[19,20]*C3[9,8]*C4[1,2]*B[18,15,14,25,22]*B[18,17,16,24,21]*A[7,25,3,4,11]*
                    A[7,24,5,6,10]*E1[2,5,3,23]*E2[23,16,14,12]*E4[13,17,15,19]*E5[20,21,22,26]*E6[26,10,11,9]*E7[8,6,4,1]
    energy3 = Numerator[1]/Denominator[1]

    C1 = param.CornerEnvironment[1,2][1]
    C2 = param.CornerEnvironment[2,2][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[1,2][4]

    E1 = param.EdgeEnvironment[1,2][1]
    E3 = param.EdgeEnvironment[1,2][2]
    E4 = param.EdgeEnvironment[2,2][2]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    E8 = param.EdgeEnvironment[1,2][4]

    @tensor Denominator[] := C1[8,9]*C2[20,19]*C3[13,12]*C4[1,2]*E1[2,5,3,8]*E3[9,10,11,23]*E4[23,21,22,20]*E6[19,17,15,13]*E7[12,16,14,26]*
                        E8[26,6,4,1]*B[7,11,3,4,25]*A[18,22,25,14,15]*B[7,10,5,6,24]*A[18,21,24,16,17]
    @tensor Numerator[] := C1[9,10]*C2[20,19]*C3[14,13]*C4[1,2]*E1[2,5,3,9]*E3[10,11,12,23]*E4[23,21,22,20]*E6[19,18,16,14]*
                        E7[13,17,15,26]*E8[26,6,4,1]*B[8,12,3,4,25]*B[7,11,5,6,24]*A[28,22,25,15,16]*A[27,21,24,17,18]*hr[7,27,8,28]
    energy4 = Numerator[1]/Denominator[1]
    return energy2,energy1,energy3,energy4#(energy1+energy2+energy3+energy4)/4
    #
end






function iterative_update_energy()


end




function compute_projector_energy(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,chitemp)

 

    P1 = rand(size(C1,1),size(E2,3),chitemp)
    @tensor c1e2[:] := C1[1,-3]*E2[-1,-2,-4,1]
    c1e2 = c1e2/maximum(c1e2)
    @tensor temp[:] := E1[8,10,11,-1]*C4[6,8]*E8[2,4,5,6]*C3[1,2]*E1[9,10,11,-2]*C4[7,9]*E8[3,4,5,7]*C3[1,3]
    
    temp = temp/maximum(temp)
    for j in 1:15
        @tensor Env_P1[:] := temp[3,4]*c1e2[4,5,-1,-2]*c1e2[3,5,1,2]*P1[1,2,-3] 
        sizeEnv_P1 = size(Env_P1)
        Rlt = svd(reshape(Env_P1,prod(sizeEnv_P1[1:2]),sizeEnv_P1[3]))
        P1 = reshape(Rlt.U*Rlt.V',sizeEnv_P1...)
    end

    

    P2 = rand(chitemp,size(E2,2),chitemp)
    @tensor c1e2P1[:] := c1e2[-1,-3,1,2]*P1[1,2,-2]
    for j in 1:15 
        @tensor Env_P2[:] := c1e2P1[3,1,2]*c1e2P1[4,-1,-2]*P2[1,2,-3]*temp[3,4]
        sizeEnv_P2 = size(Env_P2)
        Rlt = svd(reshape(Env_P2,prod(sizeEnv_P2[1:2]),sizeEnv_P2[3]))
        P2 = reshape(Rlt.U*Rlt.V',sizeEnv_P2...)
    end
    
    return P1,P2


end



function energy_dn(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,A,B,hr,Id,chimax)


    PL1,PL2 = compute_projector_energy(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,chimax)
    PR1,PR2 = compute_projector_energy(permutedims(C2,[2,1]),permutedims(C1,[2,1]),permutedims(C4,[2,1]),permutedims(C3,[2,1]),
            permutedims(E6,[4,2,3,1]),permutedims(E5,[4,2,3,1]),permutedims(E3,[4,2,3,1]),permutedims(E2,[4,2,3,1]),
            permutedims(E1,[4,2,3,1]),permutedims(E8,[4,2,3,1]),chimax)
    PLDn1,PLDn2 = compute_projector_energy(permutedims(C4,[2,1]),permutedims(C3,[2,1]),permutedims(C2,[2,1]),permutedims(C1,[2,1]),
            permutedims(E2,[4,2,3,1]),permutedims(E1,[4,2,3,1]),permutedims(E8,[4,2,3,1]),permutedims(E6,[4,2,3,1]),
            permutedims(E5,[4,2,3,1]),permutedims(E3,[4,2,3,1]),chimax)
    PRDn1,PRDn2 = compute_projector_energy(C3,C4,C1,C2,E5,E6,E8,E1,E2,E3,chimax)


        @tensor Numerator[:] := C1[2,1]*C2[18,20]*C3[41,40]*C4[25,26]*A[17,8,7,49,9]*A[16,13,12,48,15]*B[52,49,31,32,34]*B[51,48,36,37,39]*E1[26,29,27,47]*E2[47,5,3,2]*
                            E3[6,13,8,10]*E5[20,21,19,50]*E6[50,44,42,41]*E8[33,37,32,30]*hr[51,16,52,17]*PL1[1,3,4]*PL2[4,5,24]*PL1[6,7,11]*PL2[11,12,24]*PR1[10,9,14]*
                            PR2[14,15,23]*PR1[18,19,22]*PR2[22,21,23]*PLDn1[25,27,28]*PLDn2[28,29,46]*PLDn1[30,31,35]*PLDn2[35,36,46]*PRDn1[33,34,38]*PRDn2[38,39,45]*
                            PRDn1[40,42,43]*PRDn2[43,44,45]
        @tensor Denominator[:] :=  C1[2,1]*C2[18,20]*C3[41,40]*C4[25,26]*A[17,8,7,49,9]*A[16,13,12,48,15]*B[52,49,31,32,34]*B[51,48,36,37,39]*E1[26,29,27,47]*E2[47,5,3,2]*
                            E3[6,13,8,10]*E5[20,21,19,50]*E6[50,44,42,41]*E8[33,37,32,30]*Id[51,16,52,17]*PL1[1,3,4]*PL2[4,5,24]*PL1[6,7,11]*PL2[11,12,24]*PR1[10,9,14]*
                            PR2[14,15,23]*PR1[18,19,22]*PR2[22,21,23]*PLDn1[25,27,28]*PLDn2[28,29,46]*PLDn1[30,31,35]*PLDn2[35,36,46]*PRDn1[33,34,38]*PRDn2[38,39,45]*
                            PRDn1[40,42,43]*PRDn2[43,44,45]


        energy1 = Numerator[1]/Denominator[1]
    return energy1

end


function compute_physical(param::PEPSParam;Operator="Energy")

    if Operator == "Energy"
        hr = param.Hamiltonian
        Id = reshape(Matrix(1.0I,param.dphy^2,param.dphy^2),param.dphy,param.dphy,param.dphy,param.dphy)
    elseif Operator =="Sz"
        hr = reshape(kron(Matrix(1.0I,param.dphy,param.dphy),param.Sz),param.dphy,param.dphy,param.dphy,param.dphy)
        Id = reshape(Matrix(1.0I,param.dphy^2,param.dphy^2),param.dphy,param.dphy,param.dphy,param.dphy)
    elseif Operator == "Sy"
        Sy = real(im*param.Sy)
        hr = reshape(kron(Matrix(1.0I,2,2),Sy),2,2,2,2)
        Id = reshape(Matrix(1.0I,4,4),2,2,2,2)
    elseif Operator == "Sx"
        hr = reshape(kron(Matrix(1.0I,2,2),param.Sx),2,2,2,2)
        Id = reshape(Matrix(1.0I,4,4),2,2,2,2)
    end


    A = param.UnitCell[1,1];
    B = param.UnitCell[1,2];
    C1 = param.CornerEnvironment[1,1][1];
    C2 = param.CornerEnvironment[1,1][2];
    C3 = param.CornerEnvironment[1,2][3];
    C4 = param.CornerEnvironment[1,2][4];
    E1 = param.EdgeEnvironment[1,2][1];
    E2 = param.EdgeEnvironment[1,1][1];
    E3 = param.EdgeEnvironment[1,1][2];
    E5 = param.EdgeEnvironment[1,1][3];
    E6 = param.EdgeEnvironment[1,2][3];
    E8 = param.EdgeEnvironment[1,2][4];
    energy1 = energy_dn(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,A,B,hr,Id,size(C1,1)) ;


    #
    A = permutedims(param.UnitCell[2,1],[1,5,2,3,4])
    B = permutedims(param.UnitCell[1,1],[1,5,2,3,4])
    C1 = param.CornerEnvironment[1,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,1][3]
    C4 = param.CornerEnvironment[1,1][4]
    E2 = param.EdgeEnvironment[1,1][1]
    E3 = param.EdgeEnvironment[1,1][2]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E7 = param.EdgeEnvironment[2,1][4]
    E8 = param.EdgeEnvironment[1,1][4]
    energy2 = energy_dn(C2,C3,C4,C1,E3,E4,E5,E7,E8,E2,A,B,hr,Id,size(C1,1))



    #
    A = param.UnitCell[2,1]
    B = param.UnitCell[2,2]
    C1 = param.CornerEnvironment[2,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[2,2][4]
    E1 = param.EdgeEnvironment[2,2][1]
    E2 = param.EdgeEnvironment[2,1][1]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    energy3 = energy_dn(C1,C2,C3,C4,E1,E2,E4,E5,E6,E7,A,B,hr,Id,size(C1,1))


    A = permutedims(param.UnitCell[2,2],[1,5,2,3,4])
    B = permutedims(param.UnitCell[1,2],[1,5,2,3,4])
    C1 = param.CornerEnvironment[1,2][1]
    C2 = param.CornerEnvironment[2,2][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[1,2][4]

    E1 = param.EdgeEnvironment[1,2][1]
    E3 = param.EdgeEnvironment[1,2][2]
    E4 = param.EdgeEnvironment[2,2][2]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    E8 = param.EdgeEnvironment[1,2][4]
    energy4 = energy_dn(C2,C3,C4,C1,E3,E4,E6,E7,E8,E1,A,B,hr,Id,size(C1,1))


    return energy1,energy2,energy3,energy4
end


function one_site_reduced_density_dn(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,A,B,hr,Id,chimax)


    PL1,PL2 = compute_projector_energy(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,chimax)
    PR1,PR2 = compute_projector_energy(permutedims(C2,[2,1]),permutedims(C1,[2,1]),permutedims(C4,[2,1]),permutedims(C3,[2,1]),
            permutedims(E6,[4,2,3,1]),permutedims(E5,[4,2,3,1]),permutedims(E3,[4,2,3,1]),permutedims(E2,[4,2,3,1]),
            permutedims(E1,[4,2,3,1]),permutedims(E8,[4,2,3,1]),chimax)
    PLDn1,PLDn2 = compute_projector_energy(permutedims(C4,[2,1]),permutedims(C3,[2,1]),permutedims(C2,[2,1]),permutedims(C1,[2,1]),
            permutedims(E2,[4,2,3,1]),permutedims(E1,[4,2,3,1]),permutedims(E8,[4,2,3,1]),permutedims(E6,[4,2,3,1]),
            permutedims(E5,[4,2,3,1]),permutedims(E3,[4,2,3,1]),chimax)
    PRDn1,PRDn2 = compute_projector_energy(C3,C4,C1,C2,E5,E6,E8,E1,E2,E3,chimax)


        @tensor rho[:] := C1[2,1]*C2[18,20]*C3[41,40]*C4[25,26]*A[17,8,7,49,9]*A[17,13,12,48,15]*B[-2,49,31,32,34]*B[-1,48,36,37,39]*E1[26,29,27,47]*E2[47,5,3,2]*
                            E3[6,13,8,10]*E5[20,21,19,50]*E6[50,44,42,41]*E8[33,37,32,30]*PL1[1,3,4]*PL2[4,5,24]*PL1[6,7,11]*PL2[11,12,24]*PR1[10,9,14]*
                            PR2[14,15,23]*PR1[18,19,22]*PR2[22,21,23]*PLDn1[25,27,28]*PLDn2[28,29,46]*PLDn1[30,31,35]*PLDn2[35,36,46]*PRDn1[33,34,38]*PRDn2[38,39,45]*
                            PRDn1[40,42,43]*PRDn2[43,44,45]
    return rho/tr(rho)
end


function compute_one_site_reduced_density(param::PEPSParam;Operator="Energy")

    if Operator == "Energy"
        hr = param.Hamiltonian
        Id = reshape(Matrix(1.0I,param.dphy^2,param.dphy^2),param.dphy,param.dphy,param.dphy,param.dphy)
    elseif Operator =="Sz"
        hr = reshape(kron(Matrix(1.0I,param.dphy,param.dphy),param.Sz),param.dphy,param.dphy,param.dphy,param.dphy)
        Id = reshape(Matrix(1.0I,param.dphy^2,param.dphy^2),param.dphy,param.dphy,param.dphy,param.dphy)
    elseif Operator == "Sy"
        Sy = real(im*param.Sy)
        hr = reshape(kron(Matrix(1.0I,2,2),Sy),2,2,2,2)
        Id = reshape(Matrix(1.0I,4,4),2,2,2,2)
    elseif Operator == "Sx"
        hr = reshape(kron(Matrix(1.0I,2,2),param.Sx),2,2,2,2)
        Id = reshape(Matrix(1.0I,4,4),2,2,2,2)
    end


    A = param.UnitCell[1,1];
    B = param.UnitCell[1,2];
    C1 = param.CornerEnvironment[1,1][1];
    C2 = param.CornerEnvironment[1,1][2];
    C3 = param.CornerEnvironment[1,2][3];
    C4 = param.CornerEnvironment[1,2][4];
    E1 = param.EdgeEnvironment[1,2][1];
    E2 = param.EdgeEnvironment[1,1][1];
    E3 = param.EdgeEnvironment[1,1][2];
    E5 = param.EdgeEnvironment[1,1][3];
    E6 = param.EdgeEnvironment[1,2][3];
    E8 = param.EdgeEnvironment[1,2][4];
    rho1 = one_site_reduced_density_dn(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,A,B,hr,Id,size(C1,1)) ;


    #
    A = permutedims(param.UnitCell[2,1],[1,5,2,3,4])
    B = permutedims(param.UnitCell[1,1],[1,5,2,3,4])
    C1 = param.CornerEnvironment[1,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,1][3]
    C4 = param.CornerEnvironment[1,1][4]
    E2 = param.EdgeEnvironment[1,1][1]
    E3 = param.EdgeEnvironment[1,1][2]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E7 = param.EdgeEnvironment[2,1][4]
    E8 = param.EdgeEnvironment[1,1][4]
    rho2 = one_site_reduced_density_dn(C2,C3,C4,C1,E3,E4,E5,E7,E8,E2,A,B,hr,Id,size(C1,1))



    #
    A = param.UnitCell[2,1]
    B = param.UnitCell[2,2]
    C1 = param.CornerEnvironment[2,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[2,2][4]
    E1 = param.EdgeEnvironment[2,2][1]
    E2 = param.EdgeEnvironment[2,1][1]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    rho3 = one_site_reduced_density_dn(C1,C2,C3,C4,E1,E2,E4,E5,E6,E7,A,B,hr,Id,size(C1,1))


    A = permutedims(param.UnitCell[2,2],[1,5,2,3,4])
    B = permutedims(param.UnitCell[1,2],[1,5,2,3,4])
    C1 = param.CornerEnvironment[1,2][1]
    C2 = param.CornerEnvironment[2,2][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[1,2][4]

    E1 = param.EdgeEnvironment[1,2][1]
    E3 = param.EdgeEnvironment[1,2][2]
    E4 = param.EdgeEnvironment[2,2][2]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    E8 = param.EdgeEnvironment[1,2][4]
    rho4 = one_site_reduced_density_dn(C2,C3,C4,C1,E3,E4,E6,E7,E8,E1,A,B,hr,Id,size(C1,1))


    return rho1,rho2,rho3,rho4 
end







function two_site_reduced_density_dn(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,A,B,hr,Id,chimax)


    PL1,PL2 = compute_projector_energy(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,chimax)
    PR1,PR2 = compute_projector_energy(permutedims(C2,[2,1]),permutedims(C1,[2,1]),permutedims(C4,[2,1]),permutedims(C3,[2,1]),
            permutedims(E6,[4,2,3,1]),permutedims(E5,[4,2,3,1]),permutedims(E3,[4,2,3,1]),permutedims(E2,[4,2,3,1]),
            permutedims(E1,[4,2,3,1]),permutedims(E8,[4,2,3,1]),chimax)
    PLDn1,PLDn2 = compute_projector_energy(permutedims(C4,[2,1]),permutedims(C3,[2,1]),permutedims(C2,[2,1]),permutedims(C1,[2,1]),
            permutedims(E2,[4,2,3,1]),permutedims(E1,[4,2,3,1]),permutedims(E8,[4,2,3,1]),permutedims(E6,[4,2,3,1]),
            permutedims(E5,[4,2,3,1]),permutedims(E3,[4,2,3,1]),chimax)
    PRDn1,PRDn2 = compute_projector_energy(C3,C4,C1,C2,E5,E6,E8,E1,E2,E3,chimax)


        @tensor rho[:] := C1[2,1]*C2[18,20]*C3[41,40]*C4[25,26]*A[-4,8,7,49,9]*A[-2,13,12,48,15]*B[-3,49,31,32,34]*B[-1,48,36,37,39]*E1[26,29,27,47]*E2[47,5,3,2]*
                            E3[6,13,8,10]*E5[20,21,19,50]*E6[50,44,42,41]*E8[33,37,32,30]*PL1[1,3,4]*PL2[4,5,24]*PL1[6,7,11]*PL2[11,12,24]*PR1[10,9,14]*
                            PR2[14,15,23]*PR1[18,19,22]*PR2[22,21,23]*PLDn1[25,27,28]*PLDn2[28,29,46]*PLDn1[30,31,35]*PLDn2[35,36,46]*PRDn1[33,34,38]*PRDn2[38,39,45]*
                            PRDn1[40,42,43]*PRDn2[43,44,45]
    return rho/tr(reshape(rho,4,4))
end


function compute_two_site_reduced_density(param::PEPSParam;Operator="Energy")

    if Operator == "Energy"
        hr = param.Hamiltonian
        Id = reshape(Matrix(1.0I,param.dphy^2,param.dphy^2),param.dphy,param.dphy,param.dphy,param.dphy)
    elseif Operator =="Sz"
        hr = reshape(kron(Matrix(1.0I,param.dphy,param.dphy),param.Sz),param.dphy,param.dphy,param.dphy,param.dphy)
        Id = reshape(Matrix(1.0I,param.dphy^2,param.dphy^2),param.dphy,param.dphy,param.dphy,param.dphy)
    elseif Operator == "Sy"
        Sy = real(im*param.Sy)
        hr = reshape(kron(Matrix(1.0I,2,2),Sy),2,2,2,2)
        Id = reshape(Matrix(1.0I,4,4),2,2,2,2)
    elseif Operator == "Sx"
        hr = reshape(kron(Matrix(1.0I,2,2),param.Sx),2,2,2,2)
        Id = reshape(Matrix(1.0I,4,4),2,2,2,2)
    end


    A = param.UnitCell[1,1];
    B = param.UnitCell[1,2];
    C1 = param.CornerEnvironment[1,1][1];
    C2 = param.CornerEnvironment[1,1][2];
    C3 = param.CornerEnvironment[1,2][3];
    C4 = param.CornerEnvironment[1,2][4];
    E1 = param.EdgeEnvironment[1,2][1];
    E2 = param.EdgeEnvironment[1,1][1];
    E3 = param.EdgeEnvironment[1,1][2];
    E5 = param.EdgeEnvironment[1,1][3];
    E6 = param.EdgeEnvironment[1,2][3];
    E8 = param.EdgeEnvironment[1,2][4];
    rho1 = two_site_reduced_density_dn(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,A,B,hr,Id,size(C1,1)) ;


    #
    A = permutedims(param.UnitCell[2,1],[1,5,2,3,4])
    B = permutedims(param.UnitCell[1,1],[1,5,2,3,4])
    C1 = param.CornerEnvironment[1,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,1][3]
    C4 = param.CornerEnvironment[1,1][4]
    E2 = param.EdgeEnvironment[1,1][1]
    E3 = param.EdgeEnvironment[1,1][2]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E7 = param.EdgeEnvironment[2,1][4]
    E8 = param.EdgeEnvironment[1,1][4]
    rho2 = two_site_reduced_density_dn(C2,C3,C4,C1,E3,E4,E5,E7,E8,E2,A,B,hr,Id,size(C1,1))



    #
    A = param.UnitCell[2,1]
    B = param.UnitCell[2,2]
    C1 = param.CornerEnvironment[2,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[2,2][4]
    E1 = param.EdgeEnvironment[2,2][1]
    E2 = param.EdgeEnvironment[2,1][1]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    rho3 = two_site_reduced_density_dn(C1,C2,C3,C4,E1,E2,E4,E5,E6,E7,A,B,hr,Id,size(C1,1))


    A = permutedims(param.UnitCell[2,2],[1,5,2,3,4])
    B = permutedims(param.UnitCell[1,2],[1,5,2,3,4])
    C1 = param.CornerEnvironment[1,2][1]
    C2 = param.CornerEnvironment[2,2][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[1,2][4]

    E1 = param.EdgeEnvironment[1,2][1]
    E3 = param.EdgeEnvironment[1,2][2]
    E4 = param.EdgeEnvironment[2,2][2]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    E8 = param.EdgeEnvironment[1,2][4]
    rho4 = two_site_reduced_density_dn(C2,C3,C4,C1,E3,E4,E6,E7,E8,E1,A,B,hr,Id,size(C1,1))


    return rho1,rho2,rho3,rho4 
end































function compute_projector_edge(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,chitemp)

    D = size(E1,2)
    @tensor uptemp[:] := C1[-1,10]*E3[10,8,9,6]*C2[6,4]*E5[4,2,3,1]*C1[-2,11]*E3[11,8,9,7]*C2[7,5]*E5[5,2,3,1];
    @tensor dntemp[:] := E6[1,2,3,4]*C3[4,6]*E8[6,8,9,10]*C4[10,-1]*E6[1,2,3,5]*C3[5,7]*E8[7,8,9,11]*C4[11,-2];
    @tensor Env_P1[:] := uptemp[1,2]*dntemp[7,6]*E1[7,-1,-2,5]*E2[5,3,4,1]*E1[6,-3,-4,8]*E2[8,3,4,2];
    Rlt = svd(reshape(Env_P1,D^2,D^2))
    P1 = reshape(Rlt.U[:,1:chitemp],D,D,chitemp)

    @tensor Env_P2[:] := uptemp[6,8]*dntemp[1,2]*E1[1,3,4,7]*E2[7,-1,-2,6]*E1[2,3,4,5]*E2[5,-3,-4,8];
    Rlt = svd(reshape(Env_P2,D^2,D^2))
    P2 = reshape(Rlt.U[:,1:chitemp],D,D,chitemp)


    @tensor Env_P3[:] := E8[13,14,15,16]*C4[16,18]*E1[18,20,21,22]*E2[22,24,25,26]*C1[26,29]*E3[29,-1,-2,12]*C2[12,10]*E5[10,8,9,6]*
                    E6[6,4,5,2]*C3[2,1]*E8[13,14,15,17]*C4[17,19]*E1[19,20,21,23]*E2[23,24,25,27]*C1[27,28]*E3[28,-3,-4,30]*
                    C2[30,11]*E5[11,8,9,7]*E6[7,4,5,3]*C3[3,1];
    Rlt = svd(reshape(Env_P3,D^2,D^2))
    P3 = reshape(Rlt.U[:,1:chitemp],D,D,chitemp)


    return P1,P2,P3
end



function correlation(param)

    a = param.UnitCell[1,1]
    b = param.UnitCell[2,1]
    up = param.CornerEnvironment[1,1][1] 
    dn = param.CornerEnvironment[1,1][3]
    mid = param.EdgeEnvironment[1,1][1]
    mid_impurity = param.EdgeEnvironment[1,1][1]
    
    #edge_up_a = param.
    #edge_up_b = param.
    #edge_dn_a = param.
    #edge_dn_b = param.

end


function energy_dn_twotoone(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,A,B,hr,Id,chitemp)

    D = size(A,2);
    P1,P2,P3 = compute_projector_edge(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,chitemp);
    P5,P6,P8 = compute_projector_edge(C3,C4,C1,C2,E5,E6,E8,E1,E2,E3,chitemp);


    @tensor DATemp[:] := A[8,1,5,-2,2]*A[7,3,6,-1,4]*P2[6,5,-5]*P3[3,1,-6]*P5[4,2,-7]*hr[-3,7,-4,8];
    @tensor UpNumerator[:] := E2[-1,1,2,3]*C1[3,6]*E3[6,4,5,12]*C2[12,11]*E5[11,9,10,-2]*P2[1,2,7]*P3[4,5,8]*
                    P5[9,10,13]*DATemp[-3,-4,-5,-6,7,8,13];
    @tensor DBTemp[:] := B[-2,-4,1,2,6]*B[-1,-3,3,4,5]*P1[3,1,-5]*P6[5,6,-7]*P8[4,2,-6];
    @tensor DnNumerator[:] := C4[12,11]*E1[11,9,10,-1]*E6[-2,1,2,3]*C3[3,6]*E8[6,4,5,12]*DBTemp[-5,-6,-3,-4,13,7,8]*P1[9,10,13]*P6[1,2,8]*P8[4,5,7];


    
    @tensor DATemp[:] := A[8,1,5,-2,2]*A[7,3,6,-1,4]*P2[6,5,-5]*P3[3,1,-6]*P5[4,2,-7]*Id[-3,7,-4,8];
    @tensor UpDenominator[:] := E2[-1,1,2,3]*C1[3,6]*E3[6,4,5,12]*C2[12,11]*E5[11,9,10,-2]*P2[1,2,7]*P3[4,5,8]*
                    P5[9,10,13]*DATemp[-3,-4,-5,-6,7,8,13];

    @tensor Numerator[:] := UpNumerator[1,2,3,4,5,6]*DnNumerator[1,2,3,4,5,6];
    @tensor Denominator[:] := UpDenominator[1,2,3,4,5,6]*DnNumerator[1,2,3,4,5,6];


    return Numerator[1]/Denominator[1]
end




#! ComputeEnergyTwoTOOne tries to reduce the virtual bond dimension first and then 
#! do movers in iPEPS, however, this is not working properly for 
function ComputeEnergyTwoTOOne(param,chitemp;Operator="Energy")

    if Operator == "Energy"
        hr = param.Hamiltonian
        Id = reshape(Matrix(1.0I,4,4),2,2,2,2)
    elseif Operator =="Mag"
        hr = reshape(kron(Matrix(1.0I,2,2),param.Sz),2,2,2,2)
    end


    A = param.UnitCell[1,1];
    B = param.UnitCell[1,2];
    C1 = param.CornerEnvironment[1,1][1];
    C2 = param.CornerEnvironment[1,1][2];
    C3 = param.CornerEnvironment[1,2][3];
    C4 = param.CornerEnvironment[1,2][4];
    E1 = param.EdgeEnvironment[1,2][1];
    E2 = param.EdgeEnvironment[1,1][1];
    E3 = param.EdgeEnvironment[1,1][2];
    E5 = param.EdgeEnvironment[1,1][3];
    E6 = param.EdgeEnvironment[1,2][3];
    E8 = param.EdgeEnvironment[1,2][4];
    energy1 = energy_dn_twotoone(C1,C2,C3,C4,E1,E2,E3,E5,E6,E8,A,B,hr,Id,chitemp) ;


    #
    A = permutedims(param.UnitCell[2,1],[1,5,2,3,4])
    B = permutedims(param.UnitCell[1,1],[1,5,2,3,4])
    C1 = param.CornerEnvironment[1,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,1][3]
    C4 = param.CornerEnvironment[1,1][4]
    E2 = param.EdgeEnvironment[1,1][1]
    E3 = param.EdgeEnvironment[1,1][2]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E7 = param.EdgeEnvironment[2,1][4]
    E8 = param.EdgeEnvironment[1,1][4]
    energy2 = energy_dn_twotoone(C2,C3,C4,C1,E3,E4,E5,E7,E8,E2,A,B,hr,Id,chitemp);



    #
    A = param.UnitCell[2,1]
    B = param.UnitCell[2,2]
    C1 = param.CornerEnvironment[2,1][1]
    C2 = param.CornerEnvironment[2,1][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[2,2][4]
    E1 = param.EdgeEnvironment[2,2][1]
    E2 = param.EdgeEnvironment[2,1][1]
    E4 = param.EdgeEnvironment[2,1][2]
    E5 = param.EdgeEnvironment[2,1][3]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    energy3 = energy_dn_twotoone(C1,C2,C3,C4,E1,E2,E4,E5,E6,E7,A,B,hr,Id,chitemp)


    A = permutedims(param.UnitCell[2,2],[1,5,2,3,4])
    B = permutedims(param.UnitCell[1,2],[1,5,2,3,4])
    C1 = param.CornerEnvironment[1,2][1]
    C2 = param.CornerEnvironment[2,2][2]
    C3 = param.CornerEnvironment[2,2][3]
    C4 = param.CornerEnvironment[1,2][4]

    E1 = param.EdgeEnvironment[1,2][1]
    E3 = param.EdgeEnvironment[1,2][2]
    E4 = param.EdgeEnvironment[2,2][2]
    E6 = param.EdgeEnvironment[2,2][3]
    E7 = param.EdgeEnvironment[2,2][4]
    E8 = param.EdgeEnvironment[1,2][4]
    energy4 = energy_dn_twotoone(C2,C3,C4,C1,E3,E4,E6,E7,E8,E1,A,B,hr,Id,chitemp)


    return energy1,energy2,energy3,energy4




end
















function ComputeMagnetism(A,Environment,Op)
    mz = Array{Float64}(undef,0)
    for i in 1:2
        for j in 1:2
            c1 = Environment[i,j].C1;c2 = Environment[i,j].C2;c3 = Environment[i,j].C3;c4 = Environment[i,j].C4;
            e1 = Environment[i,j].E1;e2 = Environment[i,j].E2;e3 = Environment[i,j].E3;e4 = Environment[i,j].E4
            a = A[i,j]
            @tensor Numerator[]:= c1[9,10]*c2[13,14]*c3[17,18]*c4[1,2]*e1[2,6,3,9]*e2[10,11,12,13]*e3[14,15,16,17]*
                                    e4[18,7,4,1]*Op[5,8]*a[5,12,3,4,16]*a[8,11,6,7,15]
            @tensor Denominator[] := c1[8,9]*c2[12,13]*c3[17,16]*c4[1,2]*e1[2,5,3,8]*e2[9,10,11,12]*e3[13,14,15,17]*
                                    e4[16,6,4,1]*a[7,11,3,4,15]*a[7,10,5,6,14]
            append!(mz,[Numerator[1]/Denominator[1]])
        end
    end
    return mz
end



function compute_virtual_projector(param)

    A = param.UnitCell[1,1];
    B = param.UnitCell[2,1];
    D = size(A,2)

    # A right bond;
    @tensor rho_L[:] := B[7,1,2,3,-2]*B[7,4,5,6,-1]*B[8,1,2,3,-4]*B[8,4,5,6,-3]
    for j in 1:0
        @tensor rho_L[:] := rho_L[7,3,8,4]*A[9,1,3,2,-2]*A[10,1,4,2,-4]*A[9,5,7,6,-1]*A[10,5,8,6,-3]
        @tensor rho_L[:] := rho_L[7,3,8,4]*B[9,1,3,2,-2]*B[10,1,4,2,-4]*B[9,5,7,6,-1]*B[10,5,8,6,-3]
        rho_L = rho_L/maximum(rho_L)
    end
    @tensor rho_L[:] := rho_L[7,3,8,4]*A[9,1,3,2,-2]*A[10,1,4,2,-4]*A[9,5,7,6,-1]*A[10,5,8,6,-3]
    Rlt_L = svd(reshape(rho_L,D^2,D^2))

    # B left bond;
    @tensor rho_R[:] := A[8,1,-4,3,2]*A[8,4,-3,6,5]*A[7,1,-2,3,2]*A[7,4,-1,6,5]
    for j in 1:0 
        @tensor rho_R[:] := rho_R[7,3,8,4]*B[10,1,-4,2,4]*B[10,5,-3,6,8]*B[9,5,-1,6,7]*B[9,1,-2,2,3]
        @tensor rho_R[:] := rho_R[7,3,8,4]*A[10,1,-4,2,4]*A[10,5,-3,6,8]*A[9,5,-1,6,7]*A[9,1,-2,2,3]
        rho_R = rho_R/maximum(rho_R)
    end
    @tensor rho_R[:] := rho_R[7,3,8,4]*B[10,1,-4,2,4]*B[10,5,-3,6,8]*B[9,5,-1,6,7]*B[9,1,-2,2,3]
    Rlt_R = svd(reshape(rho_R,D^2,D^2))

    R = diagm(Rlt_L.S)*Rlt_L.Vt*Rlt_R.U*diagm(Rlt_R.S)

    Rlt = svd(R)

    return Rlt
end



function compute_virtual_projector_environment(param)

    C1 = param.CornerEnvironment[1,1][1]
    E1 = param.EdgeEnvironment[1,1][1]  
    C4 = param.CornerEnvironment[1,1][4]

    C2 = param.CornerEnvironment[2,1][2]
    E3 = param.EdgeEnvironment[2,1][3]
    C3 = param.CornerEnvironment[2,1][3]

    @tensor Temp_L[:] := C1[1,5]*E1[2,-1,-2,1]*C4[6,2]*C1[3,5]*E1[4,-3,-4,3]*C4[6,4]
    R_L = svd(reshape(Temp_L,16,16))

    @tensor Temp_R[:] := C2[5,3]*E3[3,-3,-4,4]*C3[4,6]*C2[5,1]*E3[1,-1,-2,2]*C3[2,6]
    R_R = svd(reshape(Temp_R,16,16))

    Rlt = svd(diagm(R_L.S)*R_L.Vt*R_R.U*diagm(R_R.S))

    #PL = 
    #PR = 
    return  
end