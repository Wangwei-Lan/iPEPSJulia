"""
    ConstructProjector_Glen_Evenbly is based on the website: https://www.tensors.net/peps
"""
function ConstructProjector_Glen_Evenbly(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)
    Dlinkau = size(a,2);Dlinkal = size(a,3);Dlinkad = size(a,4);Dlinkar = size(a,5);
    Dlinkbu = size(b,2);Dlinkbl = size(b,3);Dlinkbd = size(b,4);Dlinkbr = size(b,5);
    dphy = size(a,1);chi = size(c1,1)
    @tensor CornerU[-1,-2,-3,-4,-5,-6] := c1[1,2]*e2[-1,5,3,1]*e3[2,6,4,-4]*a[7,4,3,-3,-6]*a[7,6,5,-2,-5]
    @tensor CornerD[-1,-2,-3,-4,-5,-6] := c4[1,2]*e1[2,5,3,-1]*e8[-4,6,4,1]*b[7,-3,3,4,-6]*b[7,-2,5,6,-5]
    CornerU = reshape(CornerU,chi*Dlinkad^2,chi*Dlinkar^2);#CornerU = CornerU/maximum(abs.(CornerU));
    CornerD = reshape(CornerD,chi*Dlinkbu^2,chi*Dlinkbr^2);#CornerD = CornerD/maximum(abs.(CornerD));
    vD = rand(chi*Dlinkad^2,chimax);vU = rand(chi*Dlinkbu^2,chimax);
    CorDtmp = Array{Float64}(undef,0)
    for j in 1:5
        #println("------------------------------------This is j $j")
        CorUtmp = CornerU'*vU
        vD = ((CorUtmp'*CorUtmp)\(CorUtmp'*CornerU'))'
        CorDtmp = vD'*CornerD
        vU = ((CorDtmp*CorDtmp')\(CorDtmp*CornerD'))'
    end
    CorUtmp = CornerU'*vU
    CU = CorUtmp'*CorUtmp;CD  = CorDtmp*CorDtmp'
    (dU,uU) = eigen(CU)
    (dD,uD) = eigen(CD)
    sum(dU .== 0.0) !=0 ? (ind = findall(dU .== 0.0);dU[ind] .= 1.0e-16;dU = abs.(dU)) : dU = abs.(dU)
    sum(dD .== 0.0) !=0 ? (ind = findall(dD .== 0.0);dD[ind] .= 1.0e-16;dD = abs.(dD)) : dD = abs.(dD)
    F = svd(Matrix(Diagonal(sqrt.(dU)))*uU'*uD*Matrix(Diagonal(sqrt.(dD))))
    P = reshape(vU*uU*Matrix(Diagonal(1 ./sqrt.(dU)))*F.U*Matrix(Diagonal(sqrt.(F.S))),chi,Dlinkad,Dlinkad,chimax)
    Pdagger = reshape(vD*uD*Matrix(Diagonal(1 ./sqrt.(dD)))*F.V*Matrix(Diagonal(sqrt.(F.S))),chi,Dlinkbu,Dlinkbu,chimax)

    return P, Pdagger
end



"""
    ConstructProjector_Unitary

    Use only half of the environments to obtain a unitary projectors
"""
function ConstructProjector_Unitary(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)

    #! define dimensions
    Dlinkau = size(a,2);Dlinkal = size(a,3);Dlinkad = size(a,4);Dlinkar = size(a,5);
    Dlinkbu = size(b,2);Dlinkbl = size(b,3);Dlinkbd = size(b,4);Dlinkbr = size(b,5);
    dphy = size(a,1);chi1 = size(e1,4); chi2 = size(e5,4)
    
    #! tensor UpperPlane 
    @tensor UpperPlane[:] := c1[2,1]*c2[8,9]*e2[-1,5,3,2]*e3[1,6,4,15]*e4[15,12,11,8]*e5[9,13,10,-4]*
                a[7,4,3,-3,17]*a[7,6,5,-2,16]*b[14,11,17,-6,10]*b[14,12,16,-5,13]
    UpperPlane = reshape(UpperPlane,chi1*Dlinkad^2,chi2*Dlinkbd^2)    
    UpperPlane = UpperPlane/maximum(UpperPlane)
    
    #! full svd to construct unitary as isometries
    Rlt = svd(UpperPlane)
    P = reshape(Rlt.U[:,1:chimax],chi1,Dlinkad,Dlinkad,chimax)
    Pdagger = deepcopy(P)

    return P,Pdagger
end



"""
    ConstructProjector_Unitary_reduced_env


    use reduced environment method to obtain a unitary projector
"""
function ConstructProjector_Unitary_reduced_env(c1,c2,c3,c4,e1,e2,e3,e4,a,chimax)
    

    Dlinkau = size(a,2);Dlinkal = size(a,3);Dlinkad = size(a,4);Dlinkar = size(a,5);
    dphy = size(a,1);chi1 = size(e1,1); chi2 = size(e3,4)


    #! UpperPlane for reduced environment 
    @tensor UpperPlane[:] := c1[2,1]*c2[8,9]*a[7,4,3,-3,11]*a[7,6,5,-2,10]*e1[-1,5,3,2]*e2[1,6,4,8]*e3[9,10,11,-4]
    UpperPlane = reshape(UpperPlane,chi1*Dlinkad^2,chi2)
    Rlt = svd(UpperPlane)

    #! reshape to projects
    P = reshape(Rlt.U,chi1,Dlinkad,Dlinkad,chi2)
    Pdagger = deepcopy(P) 
    return P,Pdagger
end




"""
    philippe corboz's way to do renormalization for double layer projector
"""
function ConstructProjector_Philippe_Corboz(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)

    #! define the bondary dimensions
    Dlinkau = size(a,2);Dlinkal = size(a,3);Dlinkad = size(a,4);Dlinkar = size(a,5);
    Dlinkbu = size(b,2);Dlinkbl = size(b,3);Dlinkbd = size(b,4);Dlinkbr = size(b,5);
    dphy = size(a,1);chi1 = size(e1,4); chi2 = size(e5,4)

    #! UpperPlane and LowerPlane are the environments to compute projectors
    @tensor UpperPlane[-4,-5,-6,-1,-2,-3] := c1[2,1]*c2[8,9]*e2[-1,5,3,2]*e3[1,6,4,15]*e4[15,12,11,8]*e5[9,13,10,-4]*
                a[7,4,3,-3,17]*a[7,6,5,-2,16]*b[14,11,17,-6,10]*b[14,12,16,-5,13]
    @tensor LowerPlane[-4,-5,-6,-1,-2,-3] := c4[1,2]*c3[9,8]*e6[-4,14,11,9]*e7[8,13,10,15]*e8[15,6,4,1]*e1[2,5,3,-1]*
                a[12,-6,17,10,11]*a[12,-5,16,13,14]*b[7,-3,3,4,17]*b[7,-2,5,6,16]
    UpperPlane = reshape(UpperPlane,chi2*Dlinkbd^2,chi1*Dlinkad^2)
    LowerPlane = reshape(LowerPlane,chi2*Dlinkau^2,chi1*Dlinkbu^2)

    #! use philippe's way to compute the projectors
    Fupper = LinearAlgebra.qr(UpperPlane/maximum(UpperPlane))
    Flower = LinearAlgebra.qr(LowerPlane/maximum(LowerPlane))
    R = Fupper.R
    Rtilt = Flower.R

    #! standard way to do svd 
    @tensor TEMP[-1,-2] := R[-1,1]*Rtilt[-2,1]#inv(Rtilt)[1,-2]*inv(R)[1,-1]
    Ftmp = LinearAlgebra.svd(TEMP)
    P2 = Rtilt'*Ftmp.V[:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp.S[1:chimax]))))
    Pdagger2 = R'*Ftmp.U[:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp.S[1:chimax]))))

    #! simplified way to do svd 
    P2 = reshape(P2,chi1,Dlinkad,Dlinkad,chimax)[:,:,:,1:chimax]
    Pdagger2 = reshape(Pdagger2,chi1,Dlinkbu,Dlinkbu,chimax)[:,:,:,1:chimax]

    return P2,Pdagger2
end




"""
    Single layer based on 

"""
function ConstructProjectorSingleLayer(cij1::Array{Float64},cij2::Array{Float64},cij3::Array{Float64},cij4::Array{Float64},
                    eij1::Array{Float64},eij2::Array{Float64},eij3::Array{Float64},eij4::Array{Float64},
                    cijnext1::Array{Float64},cijnext2::Array{Float64},cijnext3::Array{Float64},cijnext4::Array{Float64},
                    eijnext1::Array{Float64},eijnext2::Array{Float64},eijnext3::Array{Float64},eijnext4::Array{Float64},
                    a::Array{Float64},b::Array{Float64},chimax::Int64)




    ####################################################
    D = size(a,2)
    chitemp = chimax
    #=
    PL_dn,PL_up = compute_projector_single_layer_appox(cij1,cij2,cij3,cij4,eij1,eij2,eij3,eij4,chitemp)
    PR_dn,PR_up = compute_projector_single_layer_appox(permutedims(cij2,[2,1]),permutedims(cij1,[2,1]),permutedims(cij4,[2,1]),permutedims(cij3,[2,1]),
                    permutedims(eij3,[4,2,3,1]),permutedims(eij2,[4,2,3,1]),permutedims(eij1,[4,2,3,1]),permutedims(eij4,[4,2,3,1]),chitemp)  
    =#
    #
    PL_dn,PL_up = compute_projector_single_layer(cij1,cij2,cij3,cij4,eij1,eij2,eij3,eij4,chitemp)
    PR_dn,PR_up = compute_projector_single_layer(permutedims(cij2,[2,1]),permutedims(cij1,[2,1]),permutedims(cij4,[2,1]),permutedims(cij3,[2,1]),
                    permutedims(eij3,[4,2,3,1]),permutedims(eij2,[4,2,3,1]),permutedims(eij1,[4,2,3,1]),permutedims(eij4,[4,2,3,1]),chitemp)  
    #=
    PR_Dn_dn,PR_Dn_up = compute_projector_single_layer(cijnext3,cijnext4,cijnext1,cijnext2,eijnext3,eijnext4,eijnext1,eijnext2,chitemp)
    PL_Dn_dn,PL_Dn_up = compute_projector_single_layer(permutedims(cijnext4,[2,1]),permutedims(cijnext3,[2,1]),permutedims(cijnext2,[2,1]),permutedims(cijnext1,[2,1]),
                    permutedims(eijnext1,[4,2,3,1]),permutedims(eijnext4,[4,2,3,1]),permutedims(eijnext3,[4,2,3,1]),permutedims(eijnext2,[4,2,3,1]),chitemp)
    =#
    ##############################
    # update up environment 
    @tensor C1t[:] := cij1[2,1]*eij1[-1,4,3,2]*PL_dn[1,3,5]*PL_up[5,4,-2] 
    @tensor E2t[:] := eij2[1,8,3,4]*PL_dn[1,2,6]*PL_up[6,7,-1]*
                    PR_dn[4,5,11]*PR_up[11,10,-4]*a[9,3,2,-3,5]*a[9,8,7,-2,10]
    @tensor C2t[:] := cij2[1,3]*eij3[3,5,2,-2]*PR_dn[1,2,4]*PR_up[4,5,-1]
    
    #=
    @tensor C3t[:] := cijnext3[3,1]*eijnext3[-1,5,2,3]*PR_Dn_dn[1,2,4]*PR_Dn_up[4,5,-2]
    @tensor E4t[:] := eijnext4[1,7,2,4]*PR_Dn_dn[1,3,6]*PR_Dn_up[6,8,-1]*
                    PL_Dn_dn[4,5,10]*PL_Dn_up[10,11,-4]*b[9,-3,5,2,3]*b[9,-2,11,7,8]
    @tensor C4t[:] := cijnext4[1,2]*eijnext1[2,4,3,-2]*PL_Dn_dn[1,3,5]*PL_Dn_up[5,4,-1]
    =#


    #! use full environemnts: seems not so accurate according to Glen's explanation;
    #=
    @tensor Dn[:] := cij3[-1,1]*eij4[1,-2,-4,2]*cij4[2,-3]
    Dn = Dn/maximum(Dn)
    sizeUp = size(Up)
    P1 = randn(sizeUp[3],sizeUp[4],chimax)
    P1 = update_P1_iterative(Up,Dn,P1)

    @tensor Up[:] :=  C1t[1,2]*E2t[2,-3,3,4]*C2t[4,-1]*P1[1,3,-2]
    @tensor Dn[:] := cij3[-1,4]*eij4[4,-3,3,2]*cij4[2,1]*P1[1,3,-2]

    sizeUp = size(Up)
    P2 = randn(sizeUp[2],sizeUp[3],chimax)
    P2 = update_P2_iterative(Up,Dn,P2)
    P1dagger = deepcopy(P1); P2dagger = deepcopy(P2)
    =#


    #! update P1 and P2 separately
    #
    @tensor Up[:] := C1t[-3,2]*E2t[2,-2,-4,1]*C2t[1,-1]
    Up = Up/maximum(Up)
    #@tensor Dn[:] := cij3[-1,1]*eij4[1,-2,-4,2]*cij4[2,-3]
    #Dn = Dn/maximum(Dn)



    #=
    sizeUp = size(Up)
    P1 = randn(prod(sizeUp[3:4]),chimax)
    P1,trunc = iterative_update(reshape(Up,prod(sizeUp[1:2]),prod(sizeUp[3:4])),P1,chimax)
    P1 = reshape(P1,sizeUp[3],sizeUp[4],chimax)

    @tensor Up[:] := Up[-1,-3,1,2]*P1[1,2,-2]
    sizeUp = size(Up)
    P2 = randn(prod(sizeUp[2:3]),chimax)
    P2,trunc = iterative_update(reshape(Up,sizeUp[1],prod(sizeUp[2:3])),P2,chimax)
    P2 = reshape(P2,sizeUp[2],sizeUp[3],chimax)
    P1dagger = deepcopy(P1); P2dagger = deepcopy(P2)
    =#    

    #! update P1 and P2 simutaneously
    #
    P1,P2 = iterative_update(C1t,E2t,C2t,chimax)
    P1dagger = deepcopy(P1); P2dagger = deepcopy(P2)
    #


    #! use Philippe Corboz's way to renormalize

    #=
    @tensor Up[:] := C1t[-3,2]*E2t[2,-2,-4,1]*C2t[1,-1]
    Up = Up/maximum(Up)
    #@tensor Dn[:] := C3t[-1,1]*E4t[1,-2,-4,2]*C4t[2,-3]
    #Dn = Dn/maximum(Dn)
    #
    @tensor Dn[:] := cij3[-1,1]*eij4[1,-2,-4,2]*cij4[2,-3]
    Dn = Dn/maximum(Dn)

    sizeUp = size(Up)
    sizeDn = size(Dn)
    Up = reshape(Up,prod(sizeUp[1:2]),prod(sizeUp[3:4]))
    Dn = reshape(Dn,prod(sizeDn[1:2]),prod(sizeDn[3:4]))
    
    P_up = rand(size(Up,1),chitemp)
    P_dn = rand(size(Dn,1),chitemp)
    @tensor Up_P_up[:] := Up[1,-2]*P_up[1,-1]
    @tensor Dn_P_dn[:] := Dn[1,-2]*P_dn[1,-1]
    @tensor Env_up[:] := Up_P_up[-2,1]*Dn_P_dn[3,1]*Dn_P_dn[3,2]*Up[-1,2]
    @tensor Env_dn[:] := Up_P_up[3,1]*Up_P_up[3,2]*Dn_P_dn[-2,1]*Dn[-1,2]
    for j in 1:20
        @tensor Up_P_up[:] = Up[1,-2]*P_up[1,-1]
        @tensor Dn_P_dn[:] = Dn[1,-2]*P_dn[1,-1]
        @tensor Env_up[:] = Up_P_up[-2,1]*Dn_P_dn[3,1]*Dn_P_dn[3,2]*Up[-1,2]
        #@tensor Env_up[:] := Up[-1,5]*Dn[4,5]*Up[1,3]*Dn[2,3]*P_up[1,-2]*P_dn[4,6]*P_dn[2,6]
        Rlt = svd(Env_up)
        P_up = Rlt.U*Rlt.V'
        #
        @tensor Up_P_up[:] = Up[1,-2]*P_up[1,-1]
        @tensor Dn_P_dn[:] = Dn[1,-2]*P_dn[1,-1]
        @tensor Env_dn[:] = Up_P_up[3,1]*Up_P_up[3,2]*Dn_P_dn[-2,1]*Dn[-1,2]
        #@tensor Env_dn[:] := Up[4,5]*Dn[-1,5]*Up[1,3]*Dn[2,3]*P_up[4,6]*P_up[1,6]*P_dn[2,-2] 
        Rlt = svd(Env_dn)
        P_dn = Rlt.U*Rlt.V'
    end
    Rlt = svd(P_up'*Up*Dn'*P_dn)
    U = P_up*Rlt.U; V = P_dn*Rlt.V; S = Rlt.S
    P1 = reshape(Dn'*V*diagm(S.^(-1/2)),sizeUp[3],sizeUp[4],chitemp)
    P1dagger = reshape(Up'*U*diagm(S.^(-1/2)),sizeDn[3],sizeDn[4],chitemp)
    #
    
    Up = reshape(Up,sizeUp...)
    Dn = reshape(Dn,sizeDn...)
    @tensor UpP1[:] := Up[-1,-3,1,2]*P1[1,2,-2]        
    @tensor DnP1dagger[:] := Dn[-1,-3,1,2]*P1dagger[1,2,-2]
    sizeUpP1 = size(UpP1)
    sizeDnP1dagger = size(DnP1dagger)    

    P_up = rand(sizeUpP1[2],sizeUpP1[3],chitemp)
    @tensor Env[:] := UpP1[3,1,2]*UpP1[3,-1,-2]*P_up[1,2,-3]
    for j in 1:20
        @tensor Env[:] = UpP1[3,1,2]*UpP1[3,-1,-2]*P_up[1,2,-3]
        Rlt = svd(reshape(Env,prod(sizeUpP1[2:3]),size(Env,3)))
        P_up = reshape(Rlt.U*Rlt.V',size(Env)...)
    end
    P2 = deepcopy(P_up)
    P2dagger = deepcopy(P_up)
    #

    #
    P_dn = rand(sizeDnP1dagger[2],sizeDnP1dagger[3],chitemp)
    for j in 1:20
        @tensor Env[:] := DnP1dagger[3,1,2]*DnP1dagger[3,-1,-2]*P_dn[1,2,-3]
        Rlt = svd(reshape(Env,prod(sizeDnP1dagger[2:3]),size(Env,3)))
        P_dn = reshape(Rlt.U*Rlt.V',size(Env)...)
    end
    @tensor temp[:] := P_up[1,2,-1]*P_dn[1,2,-2]
    Rlt = svd(temp)    
    @tensor P2[:] := P_up[-1,-2,1]*Rlt.U[1,2]*diagm(sqrt.(Rlt.S))[2,-3]
    @tensor P2dagger[:] := P_dn[-1,-2,1]*Rlt.V[1,2]*diagm(sqrt.(Rlt.S))[2,-3]
    =#
    return P1,P1dagger,P2,P2dagger
end

function iterative_update(C1,E2,C2,chimax)

    P_dn = rand(size(C1,1),size(E2,3),chimax)
    P_up = rand(chimax,size(E2,2),chimax)
    @tensor Env_P_dn[:] := C1[-1,11]*E2[11,9,-2,8]*C2[8,10]*C1[1,2]*E2[2,5,3,6]*C2[6,10]*P_up[4,5,7]*P_up[-3,9,7]*P_dn[1,3,4]
    @tensor Env_P_up[:] := C1[7,8]*E2[8,-2,9,10]*C2[10,11]*C1[1,2]*E2[2,5,3,6]*C2[6,11]*P_up[4,5,-3]*P_dn[1,3,4]*P_dn[7,9,-1]
    for j in 1:15
        #print(" $j \n")
        @tensor Env_P_dn[:] = C1[-1,11]*E2[11,9,-2,8]*C2[8,10]*C1[1,2]*E2[2,5,3,6]*C2[6,10]*P_up[4,5,7]*P_up[-3,9,7]*P_dn[1,3,4]
        Rlt = svd(reshape(Env_P_dn/maximum(Env_P_dn),prod(size(Env_P_dn)[1:2]),size(Env_P_dn)[3]))
        P_dn = reshape(Rlt.U*Rlt.V',size(Env_P_dn)...)

        #
        @tensor Env_P_up[:] = C1[7,8]*E2[8,-2,9,10]*C2[10,11]*C1[1,2]*E2[2,5,3,6]*C2[6,11]*P_up[4,5,-3]*P_dn[1,3,4]*P_dn[7,9,-1]
        Rlt = svd(reshape(Env_P_up/maximum(Env_P_up),prod(size(Env_P_up)[1:2]),size(Env_P_up)[3]))
        P_up = reshape(Rlt.U*Rlt.V',size(Env_P_up)...)
        #
    end

    #! use QR decomposition to 
    #=
    @tensor Env_P_up[:] := C1[1,2]*E2[2,-2,3,4]*C2[4,-3]*P_dn[1,3,-1]
    Rlt = qr(reshape(Env_P_up,prod(size(Env_P_up)[1:2]),size(Env_P_up,3)))
    P_up = reshape(Rlt.Q[:,1:chimax],size(Env_P_up,1),size(Env_P_up,2),chimax)
    =#
    #print("\n")
    return P_dn,P_up
end



function compute_projector_single_layer_appox(c1,c2,c3,c4,e1,e2,e3,e4,chimax)
    #! update PL_dn and PL_up simutaneously: note, this is not update projectors for renormalization 
    #! this is just for approximations of the environemnts
    PL_dn = rand(size(c1,2),size(e1,3),chimax)
    PL_up = rand(chimax,size(e1,2),chimax)    
    temp = Matrix(1.0I,size(e1,1),size(e1,1))        #
    #@tensor temp[:] := c4[1,-1]*c4[1,-2]
    #@tensor temp[:] := 
    for j in 1:15
        
        @tensor Env_PL_dn[:] := c1[10,-1]*e1[9,8,-2,10]*c1[2,1]*e1[6,5,3,2]*temp[9,6]*PL_dn[1,3,4]*PL_up[-3,8,7]*PL_up[4,5,7]
        Rlt = svd(reshape(Env_PL_dn,prod(size(Env_PL_dn)[1:2]),size(Env_PL_dn)[3]))
        PL_dn = reshape(Rlt.U*Rlt.V',size(Env_PL_dn)...)

        @tensor Env_PL_up[:] := c1[8,7]*e1[10,-2,9,8]*c1[2,1]*e1[6,5,3,2]*temp[10,6]*PL_dn[7,9,-1]*PL_dn[1,3,4]*PL_up[4,5,-3]
        Rlt = svd(reshape(Env_PL_up,prod(size(Env_PL_up)[1:2]),size(Env_PL_up)[3]))
        PL_up = reshape(Rlt.U*Rlt.V',size(Env_PL_up)...)
    end

    return PL_dn,PL_up
end




function iterative_update(M::Array{Float64},P,chimax::Int64)

    @tensor Mtemp[:] := M[2,-1]*M[2,1]*P[1,-2]
    Rlt = svd(Mtemp)
    for j in 1:15
        @tensor Mtemp[:] = M[2,-1]*M[2,1]*P[1,-2]
        #Rlt = svd(M'*(M*P))
        Rlt = svd(Mtemp)
        P = Rlt.U*Rlt.V'
    end
    trunc =  (norm(M*P)-norm(M))/norm(M)

    return P,trunc
end




function compute_projector_single_layer(c1,c2,c3,c4,e1,e2,e3,e4,chitemp)

    #! update PL_dn and PL_up separately

    P1 = rand(size(c1,2),size(e1,3),chitemp)
    @tensor c1e1[:] := c1[1,-3]*e1[-1,-2,-4,1]
    c1e1 = c1e1#/maximum(c1e1)
    #=
    @tensor temp[:] := c4[16,-1]*e4[12,14,15,16]*c3[10,12]*e3[6,8,9,10]*c2[4,6]*e2[1,2,3,4]*
                c4[17,-2]*e4[13,14,15,17]*c3[11,13]*e3[7,8,9,11]*c2[5,7]*e2[1,2,3,5]
    =#
    #@tensor temp[:] := c4[10,-1]*e4[6,8,9,10]*c3[4,6]*e3[1,2,3,4]*c4[11,-2]*e4[7,8,9,11]*c3[5,7]*e3[1,2,3,5]
    temp = Matrix(1.0I,size(e1,1),size(e1,1))
    #@tensor temp[:] := c4[1,-1]*c4[1,-2]
    temp = temp/maximum(temp)
    @tensor Env_P1[:] := temp[3,4]*c1e1[4,5,-1,-2]*c1e1[3,5,1,2]*P1[1,2,-3] 
    for j in 1:15
        @tensor Env_P1[:] = temp[3,4]*c1e1[4,5,-1,-2]*c1e1[3,5,1,2]*P1[1,2,-3] 
        sizeEnv_P1 = size(Env_P1)
        try
            Rlt = svd(reshape(Env_P1,prod(sizeEnv_P1[1:2]),sizeEnv_P1[3]))
            P1 = reshape(Rlt.U*Rlt.V',sizeEnv_P1...)
        catch
            @save "Matrix.jld2" Env_P1
        end
    end
    
    P2 = rand(chitemp,size(e1,2),chitemp)
    @tensor c1e1P1[:] := c1e1[-1,-3,1,2]*P1[1,2,-2]
    @tensor Env_P2[:] := c1e1P1[3,1,2]*c1e1P1[4,-1,-2]*P2[1,2,-3]*temp[3,4]
    for j in 1:15
        @tensor Env_P2[:] = c1e1P1[3,1,2]*c1e1P1[4,-1,-2]*P2[1,2,-3]*temp[3,4]
        sizeEnv_P2 = size(Env_P2)
        try
            Rlt = svd(reshape(Env_P2,prod(sizeEnv_P2[1:2]),sizeEnv_P2[3]))
            P2 = reshape(Rlt.U*Rlt.V',sizeEnv_P2...)
        catch
            @save "Matrix.jld2" Env_P2
        end
    end
    
    return P1,P2
end




function update_P1_iterative(Up,Dn,P1)
    for j in 1:15
        @tensor Env[:] := Up[4,3,1,2]*Dn[4,3,-1,-2]*P1[1,2,-3]
        Rlt = svd(reshape(Env,prod(size(Env)[1:2]),size(Env)[3]))
        P1 = reshape(Rlt.U*Rlt.V',size(Env)...)
    end
    return P1
end


function update_P2_iterative(Up,Dn,P2)
    for j in 1:15
        @tensor Env[:] := Up[3,1,2]*Dn[3,-1,-2]*P2[1,2,-3]
        Rlt = svd(reshape(Env,prod(size(Env)[1:2]),size(Env)[3]))
        P2 = reshape(Rlt.U*Rlt.V',size(Env)...)
    end
    return P2
end









#-----------------------------------------------------------------------------------------------------------------------
#
#
#
#
#
#
#
#
#------------------------------------------------------------------------------------------------------------------------
"""
    Belowing function are used for TwoToOne truncations
    
        #! it looks this way is not working. However, I still keep the code here for later reference
"""
function ConstructProjectorIsometry(c1::Array{Float64},c2::Array{Float64},c3::Array{Float64},c4::Array{Float64},
                                e1::Array{Float64},e2::Array{Float64},e3::Array{Float64},e4::Array{Float64},e5::Array{Float64},
                                e6::Array{Float64},e7::Array{Float64},e8::Array{Float64},a::Array{Float64},b::Array{Float64},
                                chimax::Int64,Dp::Int64)
    Dlinkau = size(a,2);Dlinkal = size(a,3);Dlinkad = size(a,4);Dlinkar = size(a,5);
    Dlinkbu = size(b,2);Dlinkbl = size(b,3);Dlinkbd = size(b,4);Dlinkbr = size(b,5);
    dphy = size(a,1);chi = size(c1,1)
    
    aU = ConstructIsometries(a,b,Dp,direction=1)
    aL = ConstructIsometries(a,b,Dp,direction=2)
    aD = ConstructIsometries(a,b,Dp,direction=3)
    aR = ConstructIsometries(a,b,Dp,direction=4)
    #aU = reshape(aU,Dlinkau,Dlinkau,Dlinkau^2)
    #----- A isometry
    #=
    @tensor UpperPlane[:] := c1[3,6]*c2[23,20]*e2[-3,1,2,3]*e3[6,4,5,36]*e4[36,21,22,23]*e5[20,19,18,-1]*
                a[11,7,15,13,8]*a[11,9,14,12,10]*b[28,24,32,29,25]*b[28,26,31,30,27]*aU[4,5,17]*aU[9,7,17]*
                aL[1,2,16]*aL[14,15,16]*aR[10,8,35]*aR[31,32,35]*aD[12,13,-4]*bU[21,22,33]*bU[26,24,33]*bD[30,29,-2]*
                bR[27,25,34]*bR[19,18,34]
    @tensor LowerPlane[:] := c3[20,23]*c4[6,3]*e1[3,1,2,-3]*e6[-1,18,19,20]*e7[23,21,22,36]*e8[36,4,5,6]*
                b[11,14,7,8,12]*b[11,15,9,10,13]*a[28,24,32,30,25]*a[28,26,31,29,27]*bL[1,2,16]*bL[9,7,16]*
                bU[15,14,-4]*bD[10,8,17]*bD[4,5,17]*bR[13,12,35]*bR[31,32,35]*aU[26,24,-2]*aR[27,25,34]*aR[18,19,34]*
                aD[29,30,33]*aD[21,22,33]
    =#
    #------- AB isometry

    bU = ConstructIsometries(b,a,Dp,direction=1)
    bL = ConstructIsometries(b,a,Dp,direction=2)
    bD = ConstructIsometries(b,a,Dp,direction=3)
    bR = ConstructIsometries(b,a,Dp,direction=4)

    @tensor temp[:] := aU[1,2,-2]*bD[1,2,-1] 
    F = svd(temp)
    @tensor bD[:] := bD[-1,-2,1]*F.U[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    @tensor aU[:] := aU[-1,-2,1]*F.V[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]

    @tensor temp[:] := bU[1,2,-2]*aD[1,2,-1]
    F = svd(temp)
    @tensor aD[:] := aD[-1,-2,1]*F.U[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    @tensor bU[:] := bU[-1,-2,1]*F.V[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    #
    @tensor UpperPlane[:] := c1[3,6]*c2[23,20]*e2[-3,1,2,3]*e3[6,4,5,36]*e4[36,21,22,23]*e5[20,19,18,-1]*
                a[11,7,15,13,8]*a[11,9,14,12,10]*b[28,24,32,29,25]*b[28,26,31,30,27]*bD[4,5,17]*aU[9,7,17]*
                aL[1,2,16]*aL[14,15,16]*aR[10,8,35]*aR[31,32,35]*aD[12,13,-4]*aD[21,22,33]*bU[26,24,33]*bD[30,29,-2]*
                bR[27,25,34]*bR[19,18,34]
    @tensor LowerPlane[:] := c3[20,23]*c4[6,3]*e1[3,1,2,-3]*e6[-1,18,19,20]*e7[23,21,22,36]*e8[36,4,5,6]*
                b[11,14,7,8,12]*b[11,15,9,10,13]*a[28,24,32,30,25]*a[28,26,31,29,27]*bL[1,2,16]*bL[9,7,16]*
                bU[15,14,-4]*bD[10,8,17]*aU[4,5,17]*bR[13,12,35]*bR[31,32,35]*aU[26,24,-2]*aR[27,25,34]*aR[18,19,34]*
                aD[29,30,33]*bU[21,22,33]
    UpperPlane = reshape(UpperPlane,chi*Dp,chi*Dp)
    LowerPlane = reshape(LowerPlane,chi*Dp,chi*Dp)
    
    #=
    @tensor temp[:] := aR[1,2,-1]*bL[1,2,-2]
    F = svd(temp)
    @tensor aR[:] := aR[-1,-2,1]*F.U[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    @tensor bL[:] := bL[-1,-2,1]*F.V[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    
    @tensor temp[:] := bR[1,2,-1]*aL[1,2,-2]
    F = svd(temp)
    @tensor bR[:] := bR[-1,-2,1]*F.U[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    @tensor aL[:] := aL[-1,-2,1]*F.V[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    
    @tensor UpperPlane[:] := c1[3,6]*c2[23,20]*e2[-3,1,2,3]*e3[6,4,5,36]*e4[36,21,22,23]*e5[20,19,18,-1]*
                a[11,7,15,13,8]*a[11,9,14,12,10]*b[28,24,32,29,25]*b[28,26,31,30,27]*bD[4,5,17]*aU[9,7,17]*
                bR[1,2,16]*aL[14,15,16]*aR[10,8,35]*bL[31,32,35]*aD[12,13,-4]*aD[21,22,33]*bU[26,24,33]*bD[30,29,-2]*
                bR[27,25,34]*aL[19,18,34]
    @tensor LowerPlane[:] := c3[20,23]*c4[6,3]*e1[3,1,2,-3]*e6[-1,18,19,20]*e7[23,21,22,36]*e8[36,4,5,6]*
                b[11,14,7,8,12]*b[11,15,9,10,13]*a[28,24,32,30,25]*a[28,26,31,29,27]*aR[1,2,16]*bL[9,7,16]*
                bU[15,14,-4]*bD[10,8,17]*aU[4,5,17]*bR[13,12,35]*aL[31,32,35]*aU[26,24,-2]*aR[27,25,34]*bL[18,19,34]*
                aD[29,30,33]*bU[21,22,33]
    UpperPlane = reshape(UpperPlane,chi*Dp,chi*Dp)
    LowerPlane = reshape(LowerPlane,chi*Dp,chi*Dp)
    =#

    Fupper = LinearAlgebra.qr(UpperPlane/maximum(UpperPlane))
    Flower = LinearAlgebra.qr(LowerPlane/maximum(LowerPlane))
    R = Fupper.R
    Rtilt = Flower.R

    @tensor TEMP[-1,-2] := R[-1,1]*Rtilt[-2,1]#inv(Rtilt)[1,-2]*inv(R)[1,-1]
    #=
    Ftmp = LinearAlgebra.svd(TMP)
    P2 = Rtilt'*Ftmp.V[:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp.S[1:chimax]))))
    Pdagger2 = R'*Ftmp.U[:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp.S[1:chimax]))))
    =#
    Ftmp = tsvd(TEMP,chimax)
    P2 = Rtilt'*Ftmp[3][:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp[2][1:chimax]))))
    Pdagger2 = R'*Ftmp[1][:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp[2][1:chimax]))))

    P2 = reshape(P2,chi,Dp,chimax)
    Pdagger2 = reshape(Pdagger2,chi,Dp,chimax)
    
    return aU,aL,aD,aR,bU,bL,bD,bR,P2,Pdagger2
end


function ConstructProjectorIsometry(cornera::Array{Array{Float64}},cornerb::Array{Array{Float64}},edgea::Array{Array{Float64}},
                                edgeb::Array{Array{Float64}},c1::Array{Float64},c2::Array{Float64},c3::Array{Float64},c4::Array{Float64},
                                e1::Array{Float64},e2::Array{Float64},e3::Array{Float64},e4::Array{Float64},e5::Array{Float64},
                                e6::Array{Float64},e7::Array{Float64},e8::Array{Float64},a::Array{Float64},b::Array{Float64},
                                chimax::Int64,Dp::Int64)

    #println("I am here!")
    Dlinkau = size(a,2);Dlinkal = size(a,3);Dlinkad = size(a,4);Dlinkar = size(a,5);
    Dlinkbu = size(b,2);Dlinkbl = size(b,3);Dlinkbd = size(b,4);Dlinkbr = size(b,5);
    dphy = size(a,1);chi = size(c1,1)
    println("construct isometry")
    @time aU,aD,aL,aR = ConstructIsometries(a,cornera,edgea,Dp)
    bU,bD,bL,bR = ConstructIsometries(b,cornerb,edgeb,Dp)
    

    @tensor temp[:] := aU[1,2,-2]*bD[1,2,-1] 
    F = svd(temp)
    @tensor bD[:] := bD[-1,-2,1]*F.U[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    @tensor aU[:] := aU[-1,-2,1]*F.V[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]

    @tensor temp[:] := bU[1,2,-2]*aD[1,2,-1]
    F = svd(temp)
    @tensor aD[:] := aD[-1,-2,1]*F.U[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    @tensor bU[:] := bU[-1,-2,1]*F.V[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    #
    @tensor UpperPlane[:] := c1[3,6]*c2[23,20]*e2[-3,1,2,3]*e3[6,4,5,36]*e4[36,21,22,23]*e5[20,19,18,-1]*
                a[11,7,15,13,8]*a[11,9,14,12,10]*b[28,24,32,29,25]*b[28,26,31,30,27]*bD[4,5,17]*aU[9,7,17]*
                aL[1,2,16]*aL[14,15,16]*aR[10,8,35]*aR[31,32,35]*aD[12,13,-4]*aD[21,22,33]*bU[26,24,33]*bD[30,29,-2]*
                bR[27,25,34]*bR[19,18,34]
    @tensor LowerPlane[:] := c3[20,23]*c4[6,3]*e1[3,1,2,-3]*e6[-1,18,19,20]*e7[23,21,22,36]*e8[36,4,5,6]*
                b[11,14,7,8,12]*b[11,15,9,10,13]*a[28,24,32,30,25]*a[28,26,31,29,27]*bL[1,2,16]*bL[9,7,16]*
                bU[15,14,-4]*bD[10,8,17]*aU[4,5,17]*bR[13,12,35]*bR[31,32,35]*aU[26,24,-2]*aR[27,25,34]*aR[18,19,34]*
                aD[29,30,33]*bU[21,22,33]
    UpperPlane = reshape(UpperPlane,chi*Dp,chi*Dp)
    LowerPlane = reshape(LowerPlane,chi*Dp,chi*Dp)
    
    #=
    @tensor temp[:] := aR[1,2,-1]*bL[1,2,-2]
    F = svd(temp)
    @tensor aR[:] := aR[-1,-2,1]*F.U[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    @tensor bL[:] := bL[-1,-2,1]*F.V[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    
    @tensor temp[:] := bR[1,2,-1]*aL[1,2,-2]
    F = svd(temp)
    @tensor bR[:] := bR[-1,-2,1]*F.U[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    @tensor aL[:] := aL[-1,-2,1]*F.V[1,2]*Matrix(Diagonal(sqrt.(F.S)))[2,-3]
    
    @tensor UpperPlane[:] := c1[3,6]*c2[23,20]*e2[-3,1,2,3]*e3[6,4,5,36]*e4[36,21,22,23]*e5[20,19,18,-1]*
                a[11,7,15,13,8]*a[11,9,14,12,10]*b[28,24,32,29,25]*b[28,26,31,30,27]*bD[4,5,17]*aU[9,7,17]*
                bR[1,2,16]*aL[14,15,16]*aR[10,8,35]*bL[31,32,35]*aD[12,13,-4]*aD[21,22,33]*bU[26,24,33]*bD[30,29,-2]*
                bR[27,25,34]*aL[19,18,34]
    @tensor LowerPlane[:] := c3[20,23]*c4[6,3]*e1[3,1,2,-3]*e6[-1,18,19,20]*e7[23,21,22,36]*e8[36,4,5,6]*
                b[11,14,7,8,12]*b[11,15,9,10,13]*a[28,24,32,30,25]*a[28,26,31,29,27]*aR[1,2,16]*bL[9,7,16]*
                bU[15,14,-4]*bD[10,8,17]*aU[4,5,17]*bR[13,12,35]*aL[31,32,35]*aU[26,24,-2]*aR[27,25,34]*bL[18,19,34]*
                aD[29,30,33]*bU[21,22,33]
    UpperPlane = reshape(UpperPlane,chi*Dp,chi*Dp)
    LowerPlane = reshape(LowerPlane,chi*Dp,chi*Dp)
    =#

    Fupper = LinearAlgebra.qr(UpperPlane/maximum(UpperPlane))
    Flower = LinearAlgebra.qr(LowerPlane/maximum(LowerPlane))
    R = Fupper.R
    Rtilt = Flower.R
    @tensor TEMP[-1,-2] := R[-1,1]*Rtilt[-2,1]#inv(Rtilt)[1,-2]*inv(R)[1,-1]
    #=
    @time Ftmp = LinearAlgebra.svd(TEMP)
    #
    P2 = Rtilt'*Ftmp.V[:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp.S[1:chimax]))))
    Pdagger2 = R'*Ftmp.U[:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp.S[1:chimax]))))
    =#
    
    Ftmp = tsvd(TEMP,chimax)
    P2 = Rtilt'*Ftmp[3][:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp[2][1:chimax]))))
    Pdagger2 = R'*Ftmp[1][:,1:chimax]*inv(Matrix(LinearAlgebra.Diagonal(sqrt.(Ftmp[2][1:chimax]))))
    P2 = reshape(P2,chi,Dp,chimax)
    Pdagger2 = reshape(Pdagger2,chi,Dp,chimax)
    
    return aU,aL,aD,aR,bU,bL,bD,bR,P2,Pdagger2

end


