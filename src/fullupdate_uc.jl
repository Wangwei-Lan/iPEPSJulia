#####################################################################################################
#
#
#
#
#                              Update Rewrite
#
#
#
#
######################################################################################################  
"""
    ConvergeCheck calculate the difference of ||Ψ_old - Ψ_new ||
"""
function ConvergeCheck(Env,bD,aU,bDtilt,aUtilt,g)
    @tensor begin
        daboo[] := Env[8,12,1,3]*bD[4,2,1]*conj(bD)[9,10,8]*conj(aU)[11,10,12]*aU[5,2,3]*g[6,7,4,5]*conj(g)[6,7,9,11]
        dabnn[] := Env[2,8,1,5]*bDtilt[3,4,1]*conj(bDtilt)[3,6,2]*aUtilt[7,4,5]*conj(aUtilt)[7,6,8]
        dabno[] := Env[6,10,1,3]*bD[4,2,1]*conj(bDtilt)[7,8,6]*aU[5,2,3]*conj(aUtilt)[9,8,10]*g[7,9,4,5]
        dabon[] := Env[6,10,1,3]*bDtilt[4,2,1]*conj(bD)[7,8,6]*aUtilt[5,2,3]*conj(aU)[9,8,10]*conj(g)[4,5,7,9]
    end
    return (daboo[1]+dabnn[1]-dabno[1]-dabon[1])/daboo[1],daboo,dabnn,dabno,dabon
end

function PermuteEnvironment(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
    EdgeEnvironment::Array{Array{Array{Float64}}},cellper::Array{Int64},envper::Array{Int64})
    
    UnitCelltemp = copy(UnitCell)  
    for i in 1:length(UnitCell)
        UnitCelltemp[i]=permutedims(UnitCell[i],cellper)
        permute!(CornerEnvironment[i],envper)
        permute!(EdgeEnvironment[i],envper)
    end

    return UnitCelltemp,CornerEnvironment,EdgeEnvironment
end





function Update(param::PEPSParam,g::Array{Float64},taoind::Int64,chiind::Int64,Dind::Int64)



    c1 = param.CornerEnvironment[1,1][1];c2 = param.CornerEnvironment[1,1][2];
    c3 = param.CornerEnvironment[1,2][3];c4 = param.CornerEnvironment[1,2][4]
    e1 = param.EdgeEnvironment[1,2][1]
    e2 = param.EdgeEnvironment[1,1][1]
    e3 = param.EdgeEnvironment[1,1][2]
    e5 = param.EdgeEnvironment[1,1][3]
    e6 = param.EdgeEnvironment[1,2][3]
    e8 = param.EdgeEnvironment[1,2][4]
    a = param.UnitCell[1,1]; b = param.UnitCell[1,2]
    
    aDlinku = size(a,2);aDlinkl = size(a,3);aDlinkd = size(a,4);aDlinkr = size(a,5);
    bDlinku = size(b,2);bDlinkl = size(b,3);bDlinkd = size(b,4);bDlinkr = size(b,5);
    dphy = size(a,1)
    # qr decomposition to get reduced tensor
        atmp = reshape(permutedims(a,[5,2,3,1,4]),aDlinku*aDlinkl*aDlinkr,aDlinkd*dphy)
        A = svd(atmp); X = copy(A.U');aU = copy((Matrix(Diagonal(A.S))*A.Vt)')
        aUbond = size(aU,2)
        X = reshape(X,aUbond,aDlinkr,aDlinku,aDlinkl);aU = reshape(aU,dphy,aDlinkd,aUbond)
    

        btmp = reshape(permutedims(b,[3,4,5,1,2]),bDlinkr*bDlinkd*bDlinkl,bDlinku*dphy)
        B = svd(btmp);Y = copy(B.U);bD = copy(Matrix(Diagonal(B.S))*B.Vt)
        bDbond = size(bD,1)
        Y = reshape(Y,bDlinkl,bDlinkd,bDlinkr,bDbond);bD = permutedims(reshape(bD,bDbond,dphy,bDlinku),[2,3,1])
         


        #---- Environment
        #@time begin 

            #=
            chitemp = param.chimax[chiind]#size(e2,3)^2
            P_up_left_1,P_up_left_2 = compute_projector(c1,e2,e1,c4,e8,c3,e6,chitemp)
            P_dn_left_1,P_dn_left_2 = compute_projector(permutedims(c4,[2,1]),permutedims(e1,[4,2,3,1]),permutedims(e2,[4,2,3,1]),
                            permutedims(c1,[2,1]),permutedims(e3,[4,2,3,1]),permutedims(c2,[2,1]),permutedims(e5,[4,2,3,1]),chitemp)
            P_up_right_1,P_up_right_2 = compute_projector(permutedims(c2,[2,1]),permutedims(e5,[4,2,3,1]),permutedims(e6,[4,2,3,1]),
                            permutedims(c3,[2,1]),permutedims(e8,[4,2,3,1]),permutedims(c4,[2,1]),permutedims(e1,[4,2,3,1]),chitemp)
            P_dn_right_1,P_dn_right_2 = compute_projector(c3,e6,e5,c2,e3,c1,e2,chitemp)


            @tensor C1t[:] := c1[1,2]*e2[-1,5,3,1]*P_up_left_1[2,3,4]*P_up_left_2[4,5,-2]
            @tensor E2t[:] := e3[1,8,3,4]*X[-3,5,3,2]*X[-2,11,8,7]*P_up_left_1[1,2,6]*P_up_left_2[6,7,-1]*P_up_right_1[4,5,10]*P_up_right_2[10,11,-4]
            @tensor C2t[:] :=  c2[2,1]*e5[1,5,3,-2]*P_up_right_1[2,3,4]*P_up_right_2[4,5,-1]

            @tensor C3t[:] := c3[3,1]*e6[-1,5,2,3]*P_dn_right_1[1,2,4]*P_dn_right_2[4,5,-2]
            @tensor E4t[:] := e8[4,8,2,1]*Y[3,2,5,-3]*Y[11,8,7,-2]*P_dn_right_1[4,5,6]*P_dn_right_2[6,7,-1]*
                                P_dn_left_1[1,3,10]*P_dn_left_2[10,11,-4]
            @tensor C4t[:] := c4[2,1]*e1[1,5,3,-2]*P_dn_left_1[2,3,4]*P_dn_left_2[4,5,-1]
            @tensor  Env[:] := C1t[5,1]*E2t[1,-2,-4,2]*C2t[2,6]*C3t[6,4]*C4t[3,5]*E4t[4,-1,-3,3]
            =#
            #
            @tensor begin
                Env[-1,-2,-3,-4] := c1[1,2]*c2[7,8]*c3[18,17]*c4[11,12]*e1[12,16,14,21]*e2[21,5,3,1]*e3[2,6,4,7]*e5[8,9,10,22]*
                            e6[22,20,19,18]*e8[17,15,13,11]*X[-4,10,4,3]*X[-2,9,6,5]*Y[14,13,19,-3]*Y[16,15,20,-1]
            end
            #
        #end
        Env = Env/maximum(Env)
        #print("\n")
        #---------------------------------------- gauge fixing
        #
        """
        Gauge Fixing is based on arxiv 1503.05345v2 Fast Full Update
        """
        @tensor Envtilt[-1,-2,-3,-4] := 0.5*(Env[-1,-2,-3,-4]+Env[-3,-4,-1,-2])
        #println("condition number for Env: ",cond(reshape(Env,prod(size(Env)[1:2]),prod(size(Env)[3:4]))))
        eige = eigen(reshape(Envtilt,size(Envtilt)[1]*size(Envtilt)[2],size(Envtilt)[1]*size(Envtilt)[2]))
        eigenvalue = eige.values/sign(eige.values[argmax(abs.(eige.values))])                           # make sure eigen values are positive
        eigenvalue = eigenvalue.*(eigenvalue .> 0)                                                      # set non positive eigen values to be zero
        Z =reshape(eige.vectors*Matrix(LinearAlgebra.Diagonal((sqrt.(eigenvalue)))),
                            size(Envtilt)[1],size(Envtilt)[2],size(Envtilt)[1]*size(Envtilt)[2])        # construct Z
        #
        #@tensor Env[-1,-2,-3,-4] = Z[-1,-2,1]*conj(Z)[-3,-4,1]
        #Env = Env/maximum(Env)
        #println("condition number for Env: ",cond(reshape(permutedims(Env,[1,3,2,4]),size(Env,1)*size(Env,3),size(Env,1)*size(Env,3))))
        (QL,R) = LinearAlgebra.qr(reshape(permutedims(Z,[1,3,2]),size(Z)[1]*size(Z)[3],size(Z)[2]))
        (L,QR) = LinearAlgebra.lq(reshape(permutedims(Z,[1,3,2]),size(Z)[1],size(Z)[2]*size(Z)[3]))
        @tensor begin
            Ztilt[-1,-2,-3] := inv(L)[-1,1]*Z[1,2,-3]*inv(R)[2,-2]
            bD[-1,-2,-3] := L[1,-3]*bD[-1,-2,1]
            aU[-1,-2,-3] := R[-3,1]*aU[-1,-2,1]
            #bDtilt[-1,-2,-3] := L[1,-3]*bDtilt[-1,-2,1]
            #aUtilt[-1,-2,-3] := R[-3,1]*aUtilt[-1,-2,1]
            X[-1,-2,-3,-4] := X[1,-2,-3,-4]*inv(R)[1,-1]
            Y[-1,-2,-3,-4] := Y[-1,-2,-3,1]*inv(L)[-4,1]
        end
        @tensor Env[-1,-2,-3,-4] = Ztilt[-1,-2,1]*conj(Ztilt)[-3,-4,1]
        Env = Env/maximum(Env)
        #println("condition number for Env: ",cond(reshape(permutedims(Env,[1,3,2,4]),size(Env,1)*size(Env,3),size(Env,1)*size(Env,3))))


        #! fix gauge first and then update 
                #
        #--- Update
        @tensor TMP[-1,-2,-3,-4] := aU[3,1,-1]*bD[2,1,-3]*g[2,3,-4,-2]
        F = LinearAlgebra.svd(reshape(TMP,dphy*aUbond,dphy*bDbond))
        U = copy(F.U);S = copy(F.S);V = copy(F.V);Vt = copy(F.Vt)
        #--- Initial of aUtilt and bDtilt
        aUtilt = permutedims(reshape(U[:,1:param.Dlink[Dind]]*Matrix(LinearAlgebra.Diagonal(sqrt.(S[1:param.Dlink[Dind]]))),aUbond,dphy,param.Dlink[Dind]),[2,3,1])
        bDtilt = permutedims(reshape(Matrix(LinearAlgebra.Diagonal(sqrt.(S[1:param.Dlink[Dind]])))*Vt[1:param.Dlink[Dind],:],param.Dlink[Dind],bDbond,dphy),[3,1,2])
        #


        """
            Gauge Fixing based on arxiv 1801.05390
        """
        #! Note yet proved to work, but worth trying
        #! No, not worth trying, because they are basically the same thing 

        """
            Gauge Fixing based on arxiv 1503.05345v2; but include aU and bD to do QR or LQ decomposition
        """
        #! Why try this ? Why not try this? 


        #
        #! check whether la and ra are correct!!!!!!!!! It seems we have bD and aU in the wrong place!!!!!!!
        costfunc = [[1.0]];numcvg = 0;k=0
        #Rlt = qr(reshape(permutedims(bDtilt,[1,3,2]),dphy*bDbond,param.Dlink[Dind]))
        #bDtilt = permutedims(reshape(Rlt.Q[:,1:param.Dlink[Dind]],dphy,bDbond,param.Dlink[Dind]),[1,3,2])
        #@tensor aUtilt[:] := aUtilt[-1,1,-3]*Rlt.R[-2,1]
        for i in 1:param.MaxLoop
            #------------------------- Iterative Update atilt and btilt
            #println("------------Update Step $i--------------------")
            #
            @tensor begin
                la[-1,-2,-3,-4,-5,-6] := Env[2,-3,1,-6]*bDtilt[3,-5,1]*conj(bDtilt)[3,-2,2]*Matrix(1.0I,dphy,dphy)[-1,-4]
                ra[-1,-2,-3] := Env[6,-3,1,3]*aU[5,2,3]*bD[4,2,1]*g[7,-1,4,5]*conj(bDtilt)[7,-2,6]
            end
            la = reshape(la,param.Dlink[Dind]*dphy*aUbond,param.Dlink[Dind]*dphy*aUbond)
            ra = reshape(ra,param.Dlink[Dind]*dphy*aUbond)
            aUtilt = reshape(\(la,ra),dphy,param.Dlink[Dind],aUbond)
            #aUtilt = reshape(pinv(la)*ra,dphy,param.Dlink[Dind],aUbond)
            #Rlt = qr(reshape(permutedims(aUtilt,[1,3,2]),dphy*aUbond,param.Dlink[Dind]))
            #aUtilt = permutedims(reshape(Rlt.Q[:,1:param.Dlink[Dind]],dphy,aUbond,param.Dlink[Dind]),[1,3,2])
            #@tensor bDtilt[:] := bDtilt[-1,1,-3]*Rlt.R[-2,1]

            @tensor begin
                lb[-1,-2,-3,-4,-5,-6] := Env[-3,1,-6,3]*aUtilt[2,-5,3]*conj(aUtilt)[2,-2,1]*Matrix(1.0I,dphy,dphy)[-1,-4]
                rb[-1,-2,-3] := Env[-3,6,2,1]*aU[5,3,1]*bD[4,3,2]*g[-1,7,4,5]*conj(aUtilt)[7,-2,6]
            end
            lb = reshape(lb,param.Dlink[Dind]*dphy*bDbond,param.Dlink[Dind]*dphy*bDbond)
            rb = reshape(rb,param.Dlink[Dind]*dphy*bDbond)
            bDtilt = reshape(\(lb,rb),dphy,param.Dlink[Dind],bDbond)
            #bDtilt = reshape(pinv(lb)*rb,dphy,param.Dlink[Dind],bDbond)
            #Rlt = qr(reshape(permutedims(bDtilt,[1,3,2]),dphy*bDbond,param.Dlink[Dind]))
            #bDtilt = permutedims(reshape(Rlt.Q[:,1:param.Dlink[Dind]],dphy,bDbond,param.Dlink[Dind]),[1,3,2])
            #@tensor aUtilt[:] := aUtilt[-1,1,-3]*Rlt.R[-2,1]
            #

            #=
            #----------------  Alternative way to update aUtilt and aDtilt
            @tensor begin
                la[:] := Env[2,-2,3,-4]*bDtilt[1,-1,2]*conj(bDtilt)[1,-3,3]
                ra[:] := Env[7,-3,4,5]*aU[3,1,5]*bD[2,1,4]*g[-1,6,3,2]*conj(bDtilt)[6,-2,7]
            end
            la = reshape(la,param.Dlink[Dind]*aUbond,param.Dlink[Dind]*aUbond)
            ra = reshape(ra,dphy,param.Dlink[Dind]*aUbond)
            aUtilt = permutedims(reshape(\(la,ra'),param.Dlink[Dind],aUbond,dphy),[3,1,2])
            #aUtilt = reshape(ra*pinv(la,5.0e-9),dphy,param.Dlink[Dind],aUbond)
            #
            #Rlt = qr(reshape(permutedims(aUtilt,[1,3,2]),dphy*aUbond,param.Dlink[Dind]))
            #aUtilt = permutedims(reshape(Rlt.Q[:,1:param.Dlink[Dind]],dphy,aUbond,param.Dlink[Dind]),[1,3,2])
            #@tensor bDtilt[:] := bDtilt[-1,1,-3]*Rlt.R[-2,1]
            #

            @tensor begin
                lb[:] := Env[-2,2,-4,3]*aUtilt[1,-1,2]*conj(aUtilt)[1,-3,3]
                rb[:] := Env[-3,6,4,5]*aU[2,1,5]*bD[3,1,4]*g[-1,7,3,2]*conj(aUtilt)[7,-2,6]
            end
            lb = reshape(lb,param.Dlink[Dind]*bDbond,param.Dlink[Dind]*bDbond)
            rb = reshape(rb,dphy,param.Dlink[Dind]*bDbond)
            bDtilt = permutedims(reshape(\(lb,rb'),param.Dlink[Dind],bDbond,dphy),[3,1,2])
            #bDtilt = reshape(rb*pinv(lb,5.0e-9),dphy,param.Dlink[Dind],bDbond)
            #Rlt = qr(reshape(permutedims(bDtilt,[1,3,2]),dphy*bDbond,param.Dlink[Dind]))
            #bDtilt = permutedims(reshape(Rlt.Q[:,1:param.Dlink[Dind]],dphy,bDbond,param.Dlink[Dind]),[1,3,2])
            #@tensor aUtilt[:] := aUtilt[-1,1,-3]*Rlt.R[-2,1]
            =#            
            #



            cost,daboo,dabnn,dabno,dabon = ConvergeCheck(Env,bD,aU,bDtilt,aUtilt,g)
            abs.((cost .- costfunc[end])[1]) < param.Tol ?  (numcvg+=1;k+=1) : (numcvg=0;k+=1)
            aUtilt = aUtilt/maximum(aUtilt)
            bDtilt = bDtilt/maximum(bDtilt)
            push!(costfunc,[cost])
            numcvg > 5 ? break : continue
        end
        #println("costfunc: ",costfunc[end])
        k == param.MaxLoop ? cvgchk = false : cvgchk = true
        
        
        
        #--- Next Few step seems not necessary! But I still include it here!
        # blance weight between a and b
        #=
        aUtmp = reshape(permutedims(aUtilt,[2,1,3]),size(aUtilt,2),dphy*aUbond)
        bDtmp = reshape(permutedims(bDtilt,[3,1,2]),bDbond*dphy,size(bDtilt,2))
        (Q1,R1) = qr(bDtmp);(L2,Q2) = lq(aUtmp)
        F = svd(R1*L2)
        bDtilt = Q1*F.U*Matrix(Diagonal(sqrt.(F.S)))
        aUtilt = Matrix(Diagonal(sqrt.(F.S)))*F.Vt*Q2
        bDtilt = permutedims(reshape(bDtilt,bDbond,dphy,size(bDtilt,2)),[2,3,1])
        aUtilt = permutedims(reshape(aUtilt,size(aUtilt,1),dphy,aUbond),[2,1,3])
        =#


        #! should multiply the gauge back 
        #@tensor aUtilt[:] := aUtilt[-1,-2,1]*inv(R)[-3,1]
        #@tensor bDtilt[:] := bDtilt[-1,-2,1]*inv(L)[1,-3]
        @tensor a[-1,-2,-3,-4,-5]:= X[1,-5,-2,-3]*aUtilt[-1,-4,1]
        @tensor b[-1,-2,-3,-4,-5]:= Y[-3,-4,-5,1]*bDtilt[-1,-2,1]
        
        param.UnitCell[1,1] = a/maximum(a)
        param.UnitCell[1,2] = b/maximum(b)
        param.UnitCell[2,1] = b/maximum(b)
        param.UnitCell[2,2] = a/maximum(a)
        a = a/maximum(a)
        b = b/maximum(b)

        #! The update cases should be correct
        #! Update the environment such that, the same tensor in the unit cell have the same environment 
        return a,b,cvgchk
end


"""
    update(param::PEPSParam,ind::Int64;direction=1::Int64)
    
    #! update the tensor for different directions.
    #!    direction == 4   :    Down Update
    #!    direction == 1   :    Right Update
    #!    direction == 2   :    Up Update
    #!    direction == 3   :    Left Update
"""
function update(param::PEPSParam,paramc::PEPSParam,taoind::Int64,chiind::Int64,Dind::Int64,Dnew::Int64;direction=1::Int64,gspecify=[0.0]::Array{Float64})
    x = size(param.UnitCell,1);y = size(param.UnitCell,2)
    @match direction begin
        #=
        4 => (permarg= [[1,2],[1:1:x,1:1:y],[1,2,3,4,5],[1,2,3,4],[1,2,3,4,5],[1,2,3,4]])
        1 => (permarg= [[2,1],[x:-1:1,1:1:y],[1,5,2,3,4],[2,3,4,1],[1,3,4,5,2],[4,1,2,3]])
        2 => (permarg= [[1,2],[x:-1:1,y:-1:1],[1,4,5,2,3],[3,4,1,2],[1,4,5,2,3],[3,4,1,2]])
        3 => (permarg= [[2,1],[1:1:x,y:-1:1],[1,3,4,5,2],[4,1,2,3],[1,5,2,3,4],[2,3,4,1]])
        =#
        4 => (permarg= [[1,2],[1:1:x,1:1:y],[1,2,3,4,5],[1,2,3,4],[1,2,3,4,5],[1,2,3,4]])
        1 => (permarg= [[2,1],[x:-1:1,1:1:y],[1,5,2,3,4],[2,3,4,1],[1,3,4,5,2],[4,1,2,3]])
        2 => (permarg= [[1,2],[x:-1:1,y:-1:1],[1,4,5,2,3],[3,4,1,2],[1,4,5,2,3],[3,4,1,2]])
        3 => (permarg= [[2,1],[1:1:x,y:-1:1],[1,3,4,5,2],[4,1,2,3],[1,5,2,3,4],[2,3,4,1]])

    end

    #=
    permutedims!(paramc.UnitCell,param.UnitCell,permarg[1])[permarg[2]...]
    permutedims!(paramc.CornerEnvironment,param.CornerEnvironment,permarg[1])[permarg[2]...]
    permutedims!(paramc.EdgeEnvironment,param.EdgeEnvironment,permarg[1])[permarg[2]...]

    paramc.UnitCell,paramc.CornerEnvironment,paramc.EdgeEnvironment = PermuteEnvironment(paramc.UnitCell,paramc.CornerEnvironment,
                                                    paramc.EdgeEnvironment,permarg[3],permarg[4])
    paramg = param.g[taoind]        
    if gspecify == [0.0]                                            
        direction == 4 ? (@tensor g[-1,-2,-3,-4] := paramg[-1,-2,1,2]*paramg[1,2,-3,-4]) : (g = param.g[taoind])
    else
        g = gspecify
    end
    Update(paramc,g,taoind,chiind,Dind)


    paramc.UnitCell,param.CornerEnvironment,param.EdgeEnvironment = PermuteEnvironment(paramc.UnitCell,paramc.CornerEnvironment,
                                                    paramc.EdgeEnvironment,permarg[5],permarg[6])
    permutedims!(param.UnitCell,paramc.UnitCell[permarg[2]...],permarg[1])
    permutedims!(param.CornerEnvironment,paramc.CornerEnvironment[permarg[2]...],permarg[1])
    permutedims!(param.EdgeEnvironment,paramc.EdgeEnvironment[permarg[2]...],permarg[1]) 
    =#

    UnitCell = deepcopy(param.UnitCell)
    paramc.UnitCell[:]              = deepcopy(permutedims(param.UnitCell,permarg[1])[permarg[2]...])
    paramc.CornerEnvironment[:]     = deepcopy(permutedims(param.CornerEnvironment,permarg[1])[permarg[2]...])
    paramc.EdgeEnvironment[:]       = deepcopy(permutedims(param.EdgeEnvironment,permarg[1])[permarg[2]...])
    paramc.UnitCell,paramc.CornerEnvironment,paramc.EdgeEnvironment = PermuteEnvironment(paramc.UnitCell,paramc.CornerEnvironment,
                                                    paramc.EdgeEnvironment,permarg[3],permarg[4])
    paramg = param.g[taoind]        
    if gspecify == [0.0]                                            
        direction == 4 ? (@tensor g[-1,-2,-3,-4] := paramg[-1,-2,1,2]*paramg[1,2,-3,-4]) : (g = param.g[taoind])
        #g = param.g[taoind]
    else
        g = gspecify
    end



    #!
    #!
    #!  update the bond
    #!
    #!
    Update(paramc,g,taoind,chiind,Dind)




    paramc.UnitCell,paramc.CornerEnvironment,paramc.EdgeEnvironment = PermuteEnvironment(paramc.UnitCell,paramc.CornerEnvironment,
                                                    paramc.EdgeEnvironment,permarg[5],permarg[6])
    #
    param.UnitCell[:] = deepcopy(permutedims(paramc.UnitCell[permarg[2]...],permarg[1]))
    #! Corner and Edge environment won't change in this step
    #param.CornerEnvironment[:] = permutedims(paramc.CornerEnvironment[permarg[2]...],permarg[1])
    #param.EdgeEnvironment[:] = permutedims(paramc.EdgeEnvironment[permarg[2]...],permarg[1])     



end




