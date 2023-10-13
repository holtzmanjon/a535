
function rd,file
    dir='/home/holtz/raw/apo/dec06/UT061215/'
    ; return data array from input FITS file 
    return, mrdfits(dir+file,/un)
end

pro disp,im
    ; Display an input array in atv
    atv,im
end

function biassub,im
    ; subtract mean bias from input array
    bias=MEAN(im[1050:1070,10:1000])
    print,'bias level: ', bias
    return,im-bias
end

function norm,im
    ; normalize input array from mean of central region
    norm=MEAN(im[400:600,400:600])
    print,'normalization level: ', norm
    return,im/norm
end

function combine,cube
    ; Median combine input list of arrays
    return,median(cube,dim=3)
end
   
function mkflat,files
    ; Create flat field from input list of files
    for i=0,n_elements(files)-1 do begin
        im=norm(biassub(rd(files[i])))
        if i eq 0 then cube=im else cube=[[[cube]],[[im]]]
    endfor
    return,combine(cube)
end

function reduce,file,flat
    ; Reduce an input file given a flat field 
    im=rd(file)
    return,biassub(im)/flat
end

gflat=mkflat(['flat_g.0010.fits','flat_g.0011.fits','flat_g.0012.fits'])
sn17135_r=reduce('SN17135_r.0103.fits',gflat)
end
