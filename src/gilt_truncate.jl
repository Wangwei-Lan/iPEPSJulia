"""
    The idea of using GILT to reduce the bond dimension of environment tensors 
seems not working. i.e. we cannot furthur remove the circle environemnts within 
the loops.  
"""
function compute_projector_gilt(c1,c2,c3,c4,e_up_1,e_up_2,e_dn_1,e_dn_2,chikept)

    #! try to explore the idea of gilt truncations for 
    #! ipeps
    #=
    c1 = param.CornerEnvironment[1,2][1];
    c2 = param.CornerEnvironment[2,2][2];
    c3 = param.CornerEnvironment[1,1][4];
    c4 = param.CornerEnvironment[2,1][3];
    
    e_up_1 = param.EdgeEnvironment[1,2][2];
    e_up_2 = param.EdgeEnvironment[2,2][2];
    e_dn_1 = param.EdgeEnvironment[1,1][4];
    e_dn_2 = param.EdgeEnvironment[2,1][4];
    #
    #! perform svd on internal indices 

    #
    @tensor temp[:] := c2[2,1]*c3[1,7]*c2[4,3]*c3[3,8]*e_up_2[-1,5,6,2]*e_dn_2[7,9,10,-2]*
                    e_up_2[-3,5,6,4]*e_dn_2[8,9,10,-4];
    temp = temp/maximum(temp);
    @tensor temp[:] := temp[1,5,2,8]*e_up_1[9,3,4,1]*e_dn_1[5,6,7,10]*e_up_1[11,3,4,2]*
                    e_dn_1[8,6,7,12]*c1[-1,9]*c4[10,-2]*c1[-3,11]*c4[12,-4];
    temp = temp/maximum(temp);
    #
    sizetemp = size(temp);
    temp = reshape(temp,prod(sizetemp[1:2]),prod(sizetemp[3:4]));
    =#

    c1 = param.CornerEnvironment[1,1][1];
    c2 = param.CornerEnvironment[2,1][2];
    c3 = param.CornerEnvironment[2,1][3];
    c4 = param.CornerEnvironment[1,1][4];

    e1 = param.EdgeEnvironment[1,1][1];
    e2 = param.EdgeEnvironment[1,1][2];
    e3 = param.EdgeEnvironment[2,1][2];
    e4 = param.EdgeEnvironment[2,1][3];
    e5 = param.EdgeEnvironment[2,1][4];
    e6 = param.EdgeEnvironment[1,1][4];

    @tensor temp[:] := c2[-1,1]*c2[-3,3]*c3[2,-2]*c3[4,-4]*
                e4[1,5,6,2]*e4[3,5,6,4];
    temp = temp/maximum(temp);

    @tensor temp[:] := temp[1,5,2,8]*e3[-1,3,4,1]*e3[-3,3,4,2]*
                e5[5,6,7,-2]*e5[8,6,7,-4];
    temp = temp/maximum(temp);

    @tensor temp[:] := temp[1,5,2,8]*e2[-1,3,4,1]*e2[-3,3,4,2]*
                e6[5,6,7,-2]*e6[8,6,7,-4];
    temp = temp/maximum(temp);

    @tensor temp[:] := temp[1,7,2,8]*c1[3,1]*c1[4,2]*
                c4[7,-2]*c4[8,-4]*e1[-1,5,6,3]*e1[-3,5,6,4];
    temp = temp/maximum(temp);
    sizetemp = size(temp)
    Rlt = svd(reshape(temp,prod(sizetemp[1:2]),prod(sizetemp[3:4])));
    U = reshape(Rlt.U,sizetemp[1],sizetemp[2],prod(sizetemp[1:2]));
    @tensor t_i[:] := U[1,1,-1];
    S = sqrt.(Rlt.S)
    #! set truncation epsilon
    epsilon = 1.0e-4
    t_i_new = t_i .* (S.^2 ./(S.^2 .+ (epsilon*maximum(S)).^2)); 

    @tensor R[:] := t_i_new[1]*U[-1,-2,1]; 
    Rlt_R = svd(R);
    println(Rlt_R.S)
    #! do the truncations 
    chikept = 40
    P = Rlt_R.U[:,1:chikept]*diagm(Rlt_R.S[1:chikept])
    Pdagger = Rlt_R.V[:,1:chikept]*diagm(Rlt_R.S[1:chikept])


    @tensor exact[:] := c1[1,2]*c2[10,12]*c3[12,11]*c4[3,1]*e_up_1[2,4,5,6]*
                    e_up_2[6,8,9,10]*e_dn_1[7,4,5,3]*e_dn_2[11,8,9,7]
    @tensor truncate[:] := P[1,3]*Pdagger[2,3]*c1[1,4]*c2[12,14]*c3[14,13]*c4[5,2]*
                    e_up_1[4,6,7,8]*e_up_2[8,10,11,12]*e_dn_1[9,6,7,5]*e_dn_2[13,10,11,9]


    return P,Pdagger
end




function GILT_TRUNC()
    
    x = size(UnitCell,1);y = size(UnitCell,2)
    for i in 1:x
        isometryd1 = Array{Array{Float64}}(undef,0)
        isometrydaggerd1 = Array{Array{Float64}}(undef,0)
        isometryd2 = Array{Array{Float64}}(undef,0)
        isometrydaggerd2 = Array{Array{Float64}}(undef,0)
        for j in 1:y
            i == x ? inext = 1 : inext = i+1; j == y ? jnext = 1 : jnext = j+1 # define inext and jnext
            #----------- Single Layer
            #---- Construct Projector: will be deleted
            #---- Extract Environment Tensor: cij and cinextj
            c1 = param.CornerEnvironment[i,jnext][1]
            c2 = param.CornerEnvironment[inext,jnext][2]
            c3 = param.CornerEnvironment[inext,j][3]
            c4 = param.CornerEnvironment[i,j][4]

            e_up_1 = param.EdgeEnvironment[i,jnext][2]
            e_up_2 = param.EdgeEnvironment[inext,jnext][2]
            e_dn_1 = param.EdgeEnvironment[i,j]
            e_dn_2 = param.EdgeEnvironment[inext,j]

            #! construct P & Pdagger (GILT) to reduce the bond dimension of environment tensors 
            #! and at the same time remove the circle entanglements
            P,Pdagger = compute_projector_gilt(c1,c2,c3,c4,e_up_1,e_up_2,e_dn_1,e_dn_2,chikept)
    
            append!(isometry,[P])
            append!(isometrydagger,[Pdagger])
            #---- Update Corner Environment
            @tensor param.CornerEnvironment[inext,j][1] := param.CornerEnvironment[inext,j][1][1,-2]*isometry[end][1,-1]
            @tensor param.CornerEnvironment[i,j][3] := param.CornerEnvironment[i,j][3][-1,1]*isometrydagger[end][1,-2]

            #---- Update Edge Environment
            if j > 1
                @tensor param.EdgeEnvironment[i,j][1] := param.EdgeEnvironment[i,j][1][2,-2,-3,1]*isometry[end][2,-1]*
                                isometrydagger[end-1][1,-4]
                param.EdgeEnvironment[i,j][1][:] = param.EdgeEnvironment[i,j][1]/maximum(param.EdgeEnvironment[i,j][1])
            end
            if j == y

                @tensor param.EdgeEnvironment[][] := param.EdgeEnvironment[]*isometry*isometrydagger[]

            end

        end
    end

   return CornerEnvironment,EdgeEnvironment
end

# TODO: I should test the idea of removing short range entanglemnts in a circle.