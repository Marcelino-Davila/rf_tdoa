import sys, numpy as np, os
MAGIC = bytes([0x7F,0x80,0x7F,0x80,0x00,0xFF,0x00,0xFF,0x55,0xAA,0x55,0xAA])
def rate(path):
    pos=[]; chunk=16<<20; tail=b""; base=0
    with open(path,'rb') as f:
        while True:
            buf=f.read(chunk)
            if not buf: break
            hay=tail+buf; s=0
            while True:
                i=hay.find(MAGIC,s)
                if i==-1: break
                bp=base-len(tail)+i
                if bp>=0: pos.append(bp//2)
                s=i+1
            tail=hay[-(len(MAGIC)-1):]; base+=len(buf)
    if len(pos)<2: return None,len(pos),None
    g=np.diff(pos); g=g[(g>1e7)&(g<3e7)]
    if len(g)==0: return None,len(pos),None
    return float(np.median(g)), len(pos), int(g.max()-g.min())
for p in sys.argv[1:]:
    r,n,spread=rate(p)
    name=os.path.basename(p)
    if r is None: print(f"{name}: {n} markers, no clean 1s gaps")
    else: print(f"{name}: {r:,.0f} samp/s  ({(r-20e6)/20e6*1e6:+.0f} ppm)  spread {spread}  markers {n}")