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
    return (daboo[1]+dabnn[1]-dabno[1]-dabon[1]),daboo,dabnn,dabno,dabon
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


function Update(a::Array{Float64},b::Array{Float64},c1::Array{Float64},c2::Array{Float64},c3::Array{Float64},c4::Array{Float64},
                e1::Array{Float64},e2::Array{Float64},e3::Array{Float64},e5::Array{Float64},e6::Array{Float64},e8::Array{Float64},
                g::Array{Float64})
        
    Dlinku = size(a,2);;Dlinkl = size(a,3);Dlinkd = size(a,4);Dlinkr = size(a,5);
    dphy = size(a,1)
    # qr decomposition to get reduced tensor
        #=
        atmp = reshape(permutedims(a,[1,4,5,2,3]),Dlink*dphy,Dlink^3)
        A = lq(atmp);aU = copy(A.L);X = copy(A.Q)
        =#
        atmp = reshape(permutedims(a,[5,2,3,1,4]),Dlinku*Dlinkl*Dlinkr,Dlinkd*dphy)
        A = svd(atmp); X = copy(A.U');aU = copy((Matrix(Diagonal(A.S))*A.Vt)')
    
        aUbond = size(aU,2)
        X = reshape(X,aUbond,Dlinkr,Dlinku,Dlinkl);aU = reshape(aU,dphy,Dlinkd,aUbond)
    
        #=
        btmp = reshape(permutedims(b,[3,4,5,1,2]),Dlink^3,Dlink*dphy)
        B = LinearAlgebra.qr(btmp);bD = copy(B.R);Y= copy(B.Q)
        =#
        btmp = reshape(permutedims(b,[3,4,5,1,2]),Dlinkr*Dlinku*Dlinkl,Dlinkd*dphy)
        B = svd(btmp);Y = copy(B.U);bD = copy(Matrix(Diagonal(B.S))*B.Vt)
    
        bDbond = size(bD,1)
        Y = reshape(Y,Dlinkl,Dlinku,Dlinkr,bDbond);bD = permutedims(reshape(bD,bDbond,dphy,Dlinkd),[2,3,1])
    
        #--- Update
        @tensor TMP[-1,-2,-3,-4] := aU[3,1,-1]*bD[2,1,-3]*g[2,3,-4,-2]
        F = LinearAlgebra.svd(reshape(TMP,dphy*aUbond,dphy*bDbond))
        U = copy(F.U);S = copy(F.S);V = copy(F.V);Vt = copy(F.Vt)
    
        #--- Initial of aUtilt and bDtilt
        aUtilt = permutedims(reshape(U[:,1:param.Dlink]*Matrix(LinearAlgebra.Diagonal(sqrt.(S[1:param.Dlink]))),aUbond,dphy,param.Dlink),[2,3,1])
        #bDtilt = permutedims(reshape(V[:,1:Dlink]*Matrix(LinearAlgebra.Diagonal(sqrt.(S[1:Dlink]))),bDbond,dphy,Dlink),[2,3,1])
        bDtilt = permutedims(reshape(Matrix(LinearAlgebra.Diagonal(sqrt.(S[1:param.Dlink])))*Vt[1:param.Dlink,:],param.Dlink,bDbond,dphy),[3,1,2])
    
        #---- Environment
        @tensor begin
            Env[-1,-2,-3,-4] := c1[1,2]*c2[7,8]*c3[18,17]*c4[11,12]*e1[12,16,14,21]*e2[21,5,3,1]*e3[2,6,4,7]*e5[8,9,10,22]*
                            e6[22,20,19,18]*e8[17,15,13,11]*X[-4,10,4,3]*X[-2,9,6,5]*Y[14,13,19,-3]*Y[16,15,20,-1]
        end
        Env = Env/maximum(Env)
        #---------------------------------------- gauge fixing
        # REVIEW 
        """
            Gauge Fixing is based on arxiv 1503.05345v2 Fast Full Update
        """
        @tensor Envtilt[-1,-2,-3,-4] := 0.5*(Env[-1,-2,-3,-4]+Env[-3,-4,-1,-2])
        eige = eigen(reshape(Envtilt,size(Envtilt)[1]*size(Envtilt)[2],size(Envtilt)[1]*size(Envtilt)[2]))
        eigenvalue = eige.values/sign(eige.values[argmax(abs.(eige.values))])
        eigenvalue = eigenvalue.*(eigenvalue .> 0)
        Z =reshape(eige.vectors*Matrix(LinearAlgebra.Diagonal((sqrt.(eigenvalue)))),
                            size(Envtilt)[1],size(Envtilt)[2],size(Envtilt)[1]*size(Envtilt)[2])
        (QL,R) = LinearAlgebra.qr(reshape(permutedims(Z,[1,3,2]),size(Z)[1]*size(Z)[3],size(Z)[2]))
        #(QR,L) = LinearAlgebra.qr(reshape(permutedims(Z,[1,3,2]),size(Z)[2]*size(Z)[3],size(Z)[1]))
        (L,QR) = LinearAlgebra.lq(reshape(permutedims(Z,[1,3,2]),size(Z)[2],size(Z)[1]*size(Z)[3]))
        @tensor begin
            #=
            Ztilt[-1,-2,-3] := inv(L)[2,-2]*Z[1,2,-3]*inv(R)[1,-1]
            bD[-1,-2,-3] := R[-3,1]*bD[-1,-2,1]
            aU[-1,-2,-3] := L[-3,1]*aU[-1,-2,1]
            bDtilt[-1,-2,-3] := R[-3,1]*bDtilt[-1,-2,1]
            aUtilt[-1,-2,-3] := L[-3,1]*aUtilt[-1,-2,1]
            Xtilt[-1,-2,-3,-4] := X[1,-2,-3,-4]*inv(L)[1,-1]
            Ytilt[-1,-2,-3,-4] := Y[-1,-2,-3,1]*inv(R)[1,-4]
            =#
            Ztilt[-1,-2,-3] := inv(L)[-1,1]*Z[1,2,-3]*inv(R)[2,-2]
            bD[-1,-2,-3] := L[1,-3]*bD[-1,-2,1]
            aU[-1,-2,-3] := R[-3,1]*aU[-1,-2,1]
            bDtilt[-1,-2,-3] := L[1,-3]*bDtilt[-1,-2,1]
            aUtilt[-1,-2,-3] := R[-3,1]*aUtilt[-1,-2,1]
            X[-1,-2,-3,-4] := X[1,-2,-3,-4]*inv(R)[1,-1]
            Y[-1,-2,-3,-4] := Y[-1,-2,-3,1]*inv(L)[-4,1]
        end
        @tensor Env[-1,-2,-3,-4] = Ztilt[-1,-2,1]*conj(Ztilt)[-3,-4,1]
        Env = Env/maximum(Env)EdgeEnvironment
        #
        costfunc = [[1.0]];numcvg = 0;k=0
        for i in 1:param.MaxLoop
            #println(" Update Loop $i ")
            #------------------------- Update
            #println("------------Update Step $i--------------------")
            @tensor begin
                la[-1,-2,-3,-4,-5,-6] := Env[2,-3,1,-6]*bDtilt[3,-5,1]*conj(bDtilt)[3,-2,2]*Matrix(1.0I,dphy,dphy)[-1,-4]
                ra[-1,-2,-3] := Env[6,-3,1,3]*aU[5,2,3]*bD[4,2,1]*g[7,-1,4,5]*conj(bDtilt)[7,-2,6]
            end
            la = reshape(la,param.Dlink*dphy*aUbond,param.Dlink*dphy*aUbond)
            ra = reshape(ra,param.Dlink*dphy*aUbond)
            aUtilt = reshape(\(la,ra),dphy,param.Dlink,aUbond)
            @tensor begin
                lb[-1,-2,-3,-4,-5,-6] := Env[-3,1,-6,3]*aUtilt[2,-5,3]*conj(aUtilt)[2,-2,1]*Matrix(1.0I,dphy,dphy)[-1,-4]
                rb[-1,-2,-3] := Env[-3,6,2,1]*aU[5,3,1]*bD[4,3,2]*g[-1,7,4,5]*conj(aUtilt)[7,-2,6]
            end
            lb = reshape(lb,param.Dlink*dphy*bDbond,param.Dlink*dphy*bDbond)
            rb = reshape(rb,param.Dlink*dphy*bDbond)
            bDtilt = reshape(\(lb,rb),dphy,param.Dlink,bDbond)
            cost,daboo,dabnn,dabno,dabon = ConvergeCheck(Env,bD,aU,bDtilt,aUtilt,g)
            abs.((cost .- costfunc[end])[1]) < param.Tol ?  (numcvg+=1;k+=1) : (numcvg=0;k+=1)
            aUtilt = aUtilt/maximum(aUtilt)
            bDtilt = bDtilt/maximum(bDtilt)
            push!(costfunc,[cost])
            numcvg > 2 ? break : continue
        end
        k == param.MaxLoop ? cvgchk = false : cvgchk = true
        #--- Next Few step seems not necessary! But I still include it here!
        #=
        aUtmp = reshape(permutedims(aUtilt,[2,1,3]),Dlink,dphy*aUbond)
        bDtmp = reshape(permutedims(bDtilt,[3,1,2]),bDbond*dphy,Dlink)
        (Q1,R) = qr(bDtmp);(L,Q2) = lq(aUtmp)
        F = svd(R*L)
        bDtilt = Q1*F.U*Matrix(Diagonal(sqrt.(F.S)))
        aUtilt = Matrix(Diagonal(sqrt.(F.S)))*F.Vt*Q2
        bDtilt = permutedims(reshape(bDtilt,bDbond,dphy,Dlink),[2,3,1])
        aUtilt = permutedims(reshape(aUtilt,Dlink,dphy,aUbond),[2,1,3])
        =#
        @tensor a[-1,-2,-3,-4,-5]:= X[1,-5,-2,-3]*aUtilt[-1,-4,1]
        @tensor b[-1,-2,-3,-4,-5]:= Y[-3,-4,-5,1]*bDtilt[-1,-2,1]
        
        param.UnitCell[1,1][:] = a/maximum(a)
        param.UnitCell[1,2][:] = b/maximum(b)
        param.UnitCell[2,1][:] = b/maximum(b)
        param.UnitCell[2,2][:] = a/maximum(a)
        a = a/maximum(a)
        b = b/maximum(b)
        return a,b,cvgchk
end


"""
    update(param::PEPSParam,ind::Int64;direction=1::Int64)

update the tensor for different directions.
    direction == 1   :    Left Update
    direction == 2   :    Up Update
    direction == 3   :    Right Update
    direction == 4   :    Down Update
"""
function update(param::PEPSParam,paramc::PEPSParam,ind::Int64;direction=1::Int64)
    x = size(param.UnitCell,1);y = size(param.UnitCell,2)
    @match direction begin
        1 => (permarg= [[1,2],[1:1:x,1:1:y],[1,2,3,4,5],[1,2,3,4],[1,2,3,4,5],[1,2,3,4]])
        2 => (permarg= [[2,1],[x:-1:1,1:1:y],[1,5,2,3,4],[2,3,4,1],[1,3,4,5,2],[4,1,2,3]])
        3 => (permarg= [[1,2],[x:-1:1,y:-1:1],[1,4,5,2,3],[3,4,1,2],[1,4,5,2,3],[3,4,1,2]])
        4 => (permarg= [[2,1],[1:1:x,y:-1:1],[1,3,4,5,2],[4,1,2,3],[1,5,2,3,4],[2,3,4,1]])
    end
    paramc.UnitCell[:] = permutedims(param.UnitCell,permarg[1])[permarg[2]...]
    paramc.CornerEnvironment[:] = permutedims(param.CornerEnvironment,permarg[1])[permarg[2]...]
    paramc.EdgeEnvironment[:] = permutedims(param.EdgeEnvironment,permarg[1])[permarg[2]...]

    paramc.UnitCell,paramc.CornerEnvironment,paramc.EdgeEnvironment = PermuteEnvironment(paramc.UnitCell,paramc.CornerEnvironment,
                                                    paramc.EdgeEnvironment,permarg[3],permarg[4])
    direction == 4 ? (@tensor g[-1,-2,-3,-4] := param.g[ind][-1,-2,1,2]*param.g[ind][1,2,-3,-4]) : (g = param.g[ind])
    Update(paramc,g)
    paramc.UnitCell,param.CornerEnvironment,param.EdgeEnvironment = PermuteEnvironment(paramc.UnitCell,paramc.CornerEnvironment,
                                                    paramc.EdgeEnvironment,permarg[5],permarg[6])
    param.UnitCell[:] = permutedims(paramc.UnitCell[permarg[2]...],permarg[1])
    param.CornerEnvironment[:] = permutedims(paramc.CornerEnvironment[permarg[2]...],permarg[1])
    param.EdgeEnvironment[:] = permutedims(paramc.EdgeEnvironment[permarg[2]...],permarg[1]) 

end

