import numpy as np
from scipy.special import binom


def partitions(n):
    n_partitions = 0
    if n % 2 == 0:
        for k in range(int(n/2)):
            n_partitions += binom(n,k)
        n_partitions += binom(n,n/2)/2
    else:
        for k in range(int(np.ceil(n/2))):
            n_partitions += binom(n,k)

    partition_list = []
    for i in range(int(n_partitions)):
        partition = (((i & (1 << np.arange(n)))) > 0).astype(int)
        partition[np.where(partition == 0)] = -1
        partition_list.append(partition)

    return partition_list
