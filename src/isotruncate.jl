""" 
    ConstructIsometries(A::Array{Float64},chimax::Int64;direction=1::Int64)
Construct Isometries :
1 => Up
2 => Left
3 => Down 
4 => Right 
"""
function ConstructIsometries(A::Array{Float64},Dp::Int64;direction=1::Int64)
    #@tensor TEMP[:] := A[7,1,-2,3,2]*A[7,6,-1,4,5]*A[8,1,-4,3,2]*A[8,6,-3,4,5]
    @match direction begin 
        1 => (permarg = [1,2,3,4,5])
        2 => (permarg = [1,3,4,5,2])
        3 => (permarg = [1,4,5,2,3])
        4 => (permarg = [1,5,2,3,4])
    end
    Atemp = permutedims(A,permarg)
    @tensor TEMP[:] := Atemp[7,-2,1,2,3]*Atemp[7,-1,4,5,6]*Atemp[8,-4,1,2,3]*Atemp[8,-3,4,5,6]
    sizeTemp = size(TEMP)
    TEMP = reshape(TEMP,sizeTemp[1]^2,sizeTemp[1]^2)
    F = svd(TEMP)
    P = reshape(F.U,sizeTemp[1],sizeTemp[1],sizeTemp[1]^2)
    size(P,3) > Dp ? P = P[:,:,1:Dp] : ( )
    return P
end




#
""" 
    ConstructIsometries(A::Array{Float64},chimax::Int64;direction=1::Int64)
Construct Isometries :
1 => Up
2 => Left
3 => Down 
4 => Right 
"""
function ConstructIsometries(A::Array{Float64},B::Array{Float64},Dp::Int64;direction=1::Int64)
    #@tensor TEMP[:] := A[7,1,-2,3,2]*A[7,6,-1,4,5]*A[8,1,-4,3,2]*A[8,6,-3,4,5]
    @match direction begin 
        1 => (permarg = [1,2,3,4,5])
        2 => (permarg = [1,3,4,5,2])
        3 => (permarg = [1,4,5,2,3])
        4 => (permarg = [1,5,2,3,4])
    end
    Atemp = permutedims(A,permarg)
    Btemp = permutedims(B,permarg)
    @tensor TEMP[:] := Atemp[15,-2,11,10,12]*Atemp[15,-1,16,17,18]*Atemp[13,-4,11,9,12]*Atemp[13,-3,16,14,18]*
                    Btemp[8,10,1,2,3]*Btemp[8,17,4,5,6]*Btemp[7,9,1,2,3]*Btemp[7,14,4,5,6]
    #@tensor TEMP[:] := Atemp[7,-2,1,2,3]*Atemp[7,-1,4,5,6]*Atemp[8,-4,1,2,3]*Atemp[8,-3,4,5,6]
    sizeTemp = size(TEMP)
    TEMP = reshape(TEMP,sizeTemp[1]^2,sizeTemp[1]^2)
    F = svd(TEMP)
    P = reshape(F.U,sizeTemp[1],sizeTemp[1],sizeTemp[1]^2)
    size(P,3) > Dp ? P = P[:,:,1:Dp] : ( )
    return P
end
#


#
""" 
    ConstructIsometries(A::Array{Float64},chimax::Int64;direction=1::Int64)
Construct Isometries :
1 => Up
2 => Left
3 => Down 
4 => Right 
"""
function ConstructIsometries(A::Array{Float64},cornera::Array{Array{Float64}},edgea::Array{Array{Float64}},
                Dp::Int64)#;direction=1::Int64)
    #@tensor TEMP[:] := A[7,1,-2,3,2]*A[7,6,-1,4,5]*A[8,1,-4,3,2]*A[8,6,-3,4,5]
    #@match direction begin 
    #    1 => (permarg = [1,2,3,4,5])
    #    2 => (permarg = [1,3,4,5,2])
    #    3 => (permarg = [1,4,5,2,3])
    #    4 => (permarg = [1,5,2,3,4])
    #end
    #
    cornera1 = cornera[1]
    cornera4 = cornera[4]
    edgea1 = edgea[1]
    @tensor temp[:] := cornera1[2,5]*cornera4[6,1]*cornera1[4,5]*cornera4[6,3]*
                    edgea1[1,-1,-2,2]*edgea1[3,-3,-4,4]
    #@tensor temp[:] := edgea[1][1,-1,-2,2]*edgea[1][1,-3,-4,2]
    #@time @tensor temp[:] := cornera[2][8,9]*cornera[3][1,2]*cornera[2][19,20]*cornera[3][13,12]*edgea[2][24,10,11,8]*edgea[3][9,6,4,1]*
    #                edgea[4][2,5,3,23]*edgea[2][24,21,22,19]*edgea[3][20,16,14,13]*edgea[4][12,17,15,23]*A[7,11,-2,3,4]*A[7,10,-1,5,6]*
    #                A[18,22,-4,15,14]*A[18,21,-3,17,16]
    sizetemp = size(temp)
    F = svd(reshape(temp,size(temp,1)^2,size(temp,1)^2))
    FF = tsvd(reshape(temp,size(temp,1)^2,size(temp,1)^2),Dp)
    aL = reshape(F.U,sizetemp[1],sizetemp[2],sizetemp[1]*sizetemp[2])
    
    cornera1=cornera[1]
    cornera2=cornera[2]
    edgea2 = edgea[2]
    @tensor temp[:] := cornera1[5,1]*cornera2[2,6]*cornera1[5,3]*cornera2[4,6]*
            edgea2[1,-1,-2,2]*edgea2[3,-3,-4,4]
    #@tensor temp[:] := edgea[2][1,-1,-2,2]*edgea[2][1,-3,-4,2]
    #@tensor temp[:] := cornera[3][8,9]*cornera[4][1,2]*cornera[3][19,20]*cornera[4][12,13]*edgea[3][23,10,11,8]*edgea[4][9,6,4,1]*
    #                edgea[1][2,5,3,24]*edgea[1][13,16,14,24]*edgea[3][23,21,22,19]*edgea[4][20,17,15,12]*A[7,-2,3,4,11]*A[7,-1,5,6,10]*
    #                A[18,-4,14,15,22]*A[18,-3,16,17,21]
    F = svd(reshape(temp,size(temp,1)^2,size(temp,1)^2))
    aU = reshape(F.U,sizetemp[1],sizetemp[2],sizetemp[1]*sizetemp[2])

    cornera2 = cornera[2]
    cornera3 = cornera[3]
    edgea3 = edgea[3]
    @tensor temp[:] := cornera2[5,1]*cornera3[2,6]*cornera2[5,3]*cornera3[4,6]*
            edgea3[1,-1,-2,2]*edgea3[3,-3,-4,4]
    #@tensor temp[:] := edgea[3][1,-1,-2,2]*edgea[3][1,-3,-4,2]
    #@tensor temp[:] := cornera[1][1,2]*cornera[4][8,9]*cornera[1][13,12]*cornera[4][19,20]*edgea[1][9,5,3,1]*edgea[2][2,6,4,24]*edgea[4][23,10,11,8]*
    #                edgea[1][20,17,15,13]*edgea[2][12,16,14,24]*edgea[4][23,22,21,19]*A[7,4,3,11,-2]*A[7,6,5,10,-1]*
    #                A[18,14,15,21,-4]*A[18,16,17,22,-3]
    F = svd(reshape(temp,size(temp,1)^2,size(temp,1)^2))
    aR = reshape(F.U,sizetemp[1],sizetemp[2],sizetemp[1]*sizetemp[2])
    
    cornera3=cornera[3]
    cornera4 = cornera[4]
    edgea4 = edgea[4]
    @tensor temp[:] := cornera3[6,2]*cornera4[1,5]*cornera3[6,4]*cornera4[3,5]*
        edgea4[2,-1,-2,1]*edgea4[4,-3,-4,3]
    #@tensor temp[:] := edgea[4][1,-1,-2,2]*edgea[4][1,-3,-4,2]
    #@tensor temp[:] := cornera[1][2,1]*cornera[2][9,8]*cornera[1][13,12]*cornera[2][19,20]*A[7,4,3,-2,11]*A[7,6,5,-1,10]*
    #                    A[18,14,15,-4,22]*A[18,17,16,-3,21]*edgea[1][23,5,3,2]*edgea[1][23,16,15,13]*edgea[2][1,6,4,9]*edgea[2][12,17,14,20]*
    #                    edgea[3][8,10,11,24]*edgea[3][19,21,22,24]
    F = svd(reshape(temp,size(temp,1)^2,size(temp,1)^2))
    aD = reshape(F.U,sizetemp[1],sizetemp[2],sizetemp[1]*sizetemp[2])
    
    
    size(aU,3) > Dp ? aU = aU[:,:,1:Dp] : ( )
    size(aL,3) > Dp ? aL = aL[:,:,1:Dp] : ( )
    size(aR,3) > Dp ? aR = aR[:,:,1:Dp] : ( )
    size(aD,3) > Dp ? aD = aD[:,:,1:Dp] : ( )
    
    return aU,aD,aL,aR
end
#


function ConstructIsometriesBraLayer(A::Array{Float64},edgea::Array{Float64},chimax::Int64;direction=1::Int64)

    @match direction begin 
        1 => (permarg = [1,2,3,4,5])
        2 => (permarg = [1,3,4,5,2])
        3 => (permarg = [1,4,5,2,3])
        4 => (permarg = [1,5,2,3,4])
    end
    
    Atemp = permutedims(A,permarg)

    @tensor temp[:] := edgea[-1,4,6,5]*Atemp[2,1,6,-2,3]*Atemp[2,1,7,-4,3]*edgea[-3,4,7,5]
    F = svd(reshape(temp,size(temp,1)*size(temp,2),size(temp,1)*size(temp,2)))
    P = reshape(F.U,size(temp,1),size(temp,2),size(temp,1)*size(temp,2))
    size(temp,1)*size(temp,2) > chimax ? P = P[:,:,1:chimax] : ( )

    return P 
end



function ConstructIsometriesKetLayer(A::Array{Float64},edgea::Array{Float64},chimax::Int64;direction=1::Int64)

    @match direction begin 
        1 => (permarg = [1,2,3,4,5])
        2 => (permarg = [1,3,4,5,2])
        3 => (permarg = [1,4,5,2,3])
        4 => (permarg = [1,5,2,3,4])
    end

    Atemp = permutedims(A,permarg)
    #@tensor temp[:] := edgea[-1,5,6,2,1]*edgea[-3,7,8,2,1]*Atemp[6,3,5,-2,4]*Atemp[8,1,7,-4,4]
    F = svd(reshape(temp,size(temp,1)*size(temp,2),size(temp)*size(temp)))
    P = reshape(F.U,size(temp,1),size(temp,2),size(temp,1)*size(temp,2))

    size(temp,1)*size(temp,2) > chimax ? P = P[:,:,1:chimax] : ( )

    return P 

end