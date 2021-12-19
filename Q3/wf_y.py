def levensthein_trace(D: np.ndarray, a: str, b: str) -> Tuple[List, str]:
    i, j = len(a), len(b)
    a_ = list(a)
    ops = []
    while i and j:
        if D[i,j] == D[i-1, j] + cost(a[i-1], None):
            ops += ["Supprésion de {} en position {}".format(a[i-1], i+1)]
            del a_[i-1]
            i -= 1
        elif D[i,j] == D[i, j-1] + cost(None, b[j-1]):
            ops += ["Ajout de {} en position {} ".format(b[j-1], i+1)]
            a_.insert(i-1, b[j-1])
            j -= 1
        else:
            if a[i-1] != b[j-1]:
                ops += ["Subsitution de {} par {} en position {}".format(
                    a[i-1], b[j-1], i
                )]
                a_[i-1] = b[j-1]
            i -= 1
            j -= 1
    return ops[::-1], ''.join(a_)
