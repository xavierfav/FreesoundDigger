def parse_dict(o, kk=None):
    full_dict = {}
    if o>0:
        unique_values = set(list_dict[o].values())
        for u in unique_values:
            full_dict[u] = {}
        if kk==None:
            for k, v in list_dict[o].items():
                #print 'full_dict 0', full_dict
                full_dict[v][k] = parse_dict(o-1,k)   
            return full_dict
        else:
            for k, v in list_dict[o].items():
                if v == kk:
                    full_dict[v][k] = parse_dict(o-1,k) 
    elif o==0:
        l = []
        for i in list_dict[o].keys():
            if list_dict[o][i] == kk:
                l.append(i)
        return l
    
