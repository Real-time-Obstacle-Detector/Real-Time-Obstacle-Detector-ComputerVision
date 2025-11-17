
def conv2d_macs(h,w,cin,cout,kh,kw,groups=1):
    # Rough MACs (not counting bias/BN): H*W*Cin/G*Kh*Kw*Co
    return h*w*(cin//groups)*kh*kw*cout

def depthwise_macs(h,w,cin,kh,kw):
    return h*w*cin*kh*kw
