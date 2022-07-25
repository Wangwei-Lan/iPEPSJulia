"""
    LeftUpdate consistent with 2x2 unit cell
"""
function LeftUpdate(a,b,g,c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,updateloop,updatetol,chimax)
    Dlink = size(a,2)
    dphy = size(a,1)

    c3,c2,e5,e6 = RightMover(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax)
    # qr decomposition to get reduced tensor
    atmp = reshape(permutedims(a,[1,4,5,2,3]),Dlink*dphy,Dlink^3)
    A = lq(atmp);aU = copy(A.L);X = copy(A.Q)
    aUbond = size(aU,2)
    X = reshape(X[1:aUbond,:],aUbond,Dlink,Dlink,Dlink);aU = reshape(aU,dphy,Dlink,aUbond)

    btmp = reshape(permutedims(b,[3,4,5,1,2]),Dlink^3,Dlink*dphy)
    B = LinearAlgebra.qr(btmp);bD = copy(B.R);Y= copy(B.Q)
    bDbond = size(bD,1)
    Y = reshape(Y[:,1:bDbond],Dlink,Dlink,Dlink,bDbond);bD = permutedims(reshape(bD,bDbond,dphy,Dlink),[2,3,1])

    #--- Update
    @tensor TMP[-1,-2,-3,-4] := aU[3,1,-1]*bD[2,1,-3]*g[2,3,-4,-2]
    F = LinearAlgebra.svd(reshape(TMP,dphy*aUbond,dphy*bDbond))
    U = copy(F.U);S = copy(F.S);V = copy(F.V)
    #--- Initial of aUtilt and bDtilt (Another easier way is to use bD and aU directly)
    aUtilt = permutedims(reshape(U[:,1:Dlink]*Matrix(LinearAlgebra.Diagonal(sqrt.(S[1:Dlink]))),aUbond,dphy,Dlink),[2,3,1])
    bDtilt = permutedims(reshape(V[:,1:Dlink]*Matrix(LinearAlgebra.Diagonal(sqrt.(S[1:Dlink]))),bDbond,dphy,Dlink),[2,3,1])
    #bDtilt = bD; aUtilt = aU
    @tensor begin
        Env[-1,-2,-3,-4] := c1[1,2]*c2[7,8]*c3[18,17]*c4[11,12]*e1[12,16,14,21]*e2[21,5,3,1]*e3[2,6,4,7]*e5[8,9,10,22]*
                        e6[22,20,19,18]*e8[17,15,13,11]*X[-4,10,4,3]*X[-2,9,6,5]*Y[14,13,19,-3]*Y[16,15,20,-1]
    end
    Env = Env/maximum(Env)
    #---------------------------------------- gauge fixing
    """
        Gauge Fixing is based on arxiv 1503.05345v2 Fast Full Update
    """
    @tensor Envtilt[-1,-2,-3,-4] := 0.5*(Env[-1,-2,-3,-4]+Env[-3,-4,-1,-2])
    eige = eigen(reshape(Envtilt,size(Envtilt)[1]*size(Envtilt)[2],size(Envtilt)[1]*size(Envtilt)[2]))
    #println("This is sign of eigenvalue ",sign(eige.values[argmax(abs.(eige.values))]))
    eigenvalue = eige.values/sign(eige.values[argmax(abs.(eige.values))])
    eigenvalue = eigenvalue.*(eigenvalue .> 0)
    Z =reshape(eige.vectors*Matrix(LinearAlgebra.Diagonal((sqrt.(eigenvalue))))*sign(eige.values[argmax(abs.(eige.values))]),
                        size(Envtilt)[1],size(Envtilt)[2],size(Envtilt)[1]*size(Envtilt)[2])
    #(QL,R) = qr(reshape(permutedims(Z,[2,3,1]),size(Z)[1]*size(Z)[3],size(Z)[2]))
    #(QR,L) = qr(reshape(permutedims(Z,[1,3,2]),size(Z)[2]*size(Z)[3],size(Z)[1]))
    (QL,R) = LinearAlgebra.qr(reshape(permutedims(Z,[1,3,2]),size(Z)[1]*size(Z)[3],size(Z)[2]))
    (QR,L) = LinearAlgebra.qr(reshape(permutedims(Z,[2,3,1]),size(Z)[2]*size(Z)[3],size(Z)[1]))

    @tensor begin
        Ztilt[-1,-2,-3] := inv(L)[1,-1]*Z[1,2,-3]*inv(R)[2,-2]
        bD[-1,-2,-3] := L[-3,1]*bD[-1,-2,1]
        aU[-1,-2,-3] := R[-3,1]*aU[-1,-2,1]
        bDtilt[-1,-2,-3] := L[-3,1]*bDtilt[-1,-2,1]
        aUtilt[-1,-2,-3] := R[-3,1]*aUtilt[-1,-2,1]
        Xtilt[-1,-2,-3,-4] := X[1,-2,-3,-4]*inv(R)[1,-1]
        Ytilt[-1,-2,-3,-4] := Y[-1,-2,-3,1]*inv(L)[1,-4]
    end

    @tensor Env[-1,-2,-3,-4] := Ztilt[-1,-2,1]*conj(Ztilt)[-3,-4,1]
    Env = Env/maximum(Env)

    costfunc = [[1.0]];numcvg = 0;k=0
    for i in 1:updateloop
        #println(" Update Loop $i ")
        #------------------------- Update
        #println("------------Update Step $i--------------------")
        @tensor begin
            la[-1,-2,-3,-4,-5,-6] := Env[2,-3,1,-6]*bDtilt[3,-5,1]*conj(bDtilt)[3,-2,2]*Matrix(1.0I,dphy,dphy)[-1,-4]
            ra[-1,-2,-3] := Env[6,-3,1,3]*aU[5,2,3]*bD[4,2,1]*g[7,-1,4,5]*conj(bDtilt)[7,-2,6]
        end
        la = reshape(la,Dlink*dphy*aUbond,Dlink*dphy*aUbond)
        ra = reshape(ra,Dlink*dphy*aUbond)
        aUtilt = reshape(\(la,ra),dphy,Dlink,aUbond)
        @tensor begin
            lb[-1,-2,-3,-4,-5,-6] := Env[-3,1,-6,3]*aUtilt[2,-5,3]*conj(aUtilt)[2,-2,1]*Matrix(1.0I,dphy,dphy)[-1,-4]
            rb[-1,-2,-3] := Env[-3,6,2,1]*aU[5,3,1]*bD[4,3,2]*g[ -1,7,4,5]*conj(aUtilt)[7,-2,6]
        end
        lb = reshape(lb,Dlink*dphy*bDbond,Dlink*dphy*bDbond)
        rb = reshape(rb,Dlink*dphy*bDbond)
        bDtilt = reshape(\(lb,rb),dphy,Dlink,bDbond)
        cost,daboo,dabnn,dabno,dabon = ConvergeCheck(Env,bD,aU,bDtilt,aUtilt,g)
        abs.((cost .- costfunc[end])[1]) < updatetol ?  (numcvg+=1;k+=1) : (numcvg=0;k+=1)

        aUtilt = aUtilt/maximum(aUtilt)
        bDtilt = bDtilt/maximum(bDtilt)
        push!(costfunc,[cost])
        numcvg > 2 ? break : continue
    end
    k == updateloop ? cvgchk = false : cvgchk = true



    @tensor a[-1,-2,-3,-4,-5]:= Xtilt[1,-5,-2,-3]*aUtilt[-1,-4,1]
    @tensor b[-1,-2,-3,-4,-5]:= Ytilt[-3,-4,-5,1]*bDtilt[-1,-2,1]
    a = a/maximum(a)
    b = b/maximum(b)
    return a,b,cvgchk

end

function RightUpdate(a,b,g,c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,updateloop,updatetol,chimax)
    a = permutedims(a,[1,4,5,2,3])
    b = permutedims(b,[1,4,5,2,3])
    a,b,cvgchk=LeftUpdate(a,b,g,c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,updateloop,updatetol,chimax)
    a = permutedims(a,[1,4,5,2,3])
    b = permutedims(b,[1,4,5,2,3])
    return a,b,cvgchk
end

function DownUpdate(b,a,g,c4,c1,c2,c3,e7,e8,e1,e2,e3,e4,e5,e6,updateloop,updatetol,chimax)
    b = permutedims(b,[1,3,4,5,2])
    a = permutedims(a,[1,3,4,5,2])
    b,a,cvgchk = LeftUpdate(b,a,g,c4,c1,c2,c3,e7,e8,e1,e2,e3,e4,e5,e6,updateloop,updatetol,chimax)
    b = permutedims(b,[1,5,2,3,4])
    a = permutedims(a,[1,5,2,3,4])
    return b,a,cvgchk
end

function UpUpdate(b,a,g,c2,c3,c4,c1,e3,e4,e5,e6,e7,e8,e1,e2,updateloop,updatetol,chimax)
    b = permutedims(b,[1,5,2,3,4])
    a = permutedims(a,[1,5,2,3,4])
    b,a,cvgchk = LeftUpdate(b,a,g,c2,c3,c4,c1,e3,e4,e5,e6,e7,e8,e1,e2,updateloop,updatetol,chimax)
    b = permutedims(b,[1,3,4,5,2])
    a = permutedims(a,[1,3,4,5,2])
    return b,a,cvgchk
end

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
