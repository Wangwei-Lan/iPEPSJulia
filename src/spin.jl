function Spin(spin::Float64)
    if spin == 1/2
        #return  1/2*[0.0 1; 1 0],1/2*[0.0 -im;im 0],1/2*[1.0 0; 0 -1]
        return  [0.0 1; 1 0],[0.0 -im;im 0],[1.0 0; 0 -1]

    elseif spin == 1.0

        return 1,1,1
    elseif spin == 2.0
        Sz = [2.0 0 0 0 0;
            0 1.0 0 0 0;
            0 0 0 0 0;
            0 0 0 -1.0 0;
            0 0 0 0 -2]
        Sx = 1/2*[0 2 0 0 0;
                2 0 sqrt(6) 0 0;
                0 sqrt(6) 0 sqrt(6) 0;
                0 0 sqrt(6) 0 2;
                0 0 0 2 0]
        Sy = -1*im/2*[0 2 0 0 0;
                -2 0 sqrt(6) 0 0;
                0 -1*sqrt(6) 0 sqrt(6) 0;
                0 0 -1*sqrt(6) 0 2;
                0 0 0 -2 0]
        return Sx,Sy,Sz
    end
end
