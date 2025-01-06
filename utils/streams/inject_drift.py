import random
import numpy as np
import pandas as pd
import utils
import math

dri


def Driftpoints(X, min_point=0.7, max_point=0.9):
    driftpoint_perc = np.random.uniform(min_point, max_point, 1)
    driftpoint = int(driftpoint_perc * X.shape[0])
    num_ids = utils.num_cols(X)

    l = len(num_ids) / 2
    l = math.ceil(l)

    ids = random.sample(list(num_ids), l)

    dpoints = dict({"row": driftpoint, "cols": ids})

    return (dpoints)


def Swapcols(df, class_vec, ids, t_change):
    uclass = list(np.unique(class_vec))
    cclass = random.sample(uclass, 1)[0]

    is_class = [class_vec[i] == cclass for i in range(len(class_vec))]

    idx_class = np.argwhere(is_class)
    idx_class = [x[0] for x in idx_class]
    idx_class = np.array(idx_class)
    interval_ = idx_class[idx_class >= t_change]

    isnt_class = [not x for x in is_class]
    idx_class_not = np.argwhere(isnt_class)
    idx_class_not = [x[0] for x in idx_class_not]
    idx_class_not = np.array(idx_class_not)
    interval_not = idx_class_not[idx_class_not >= t_change]

    cnames = list(df.columns)

    df_before = df.iloc[range(t_change),].values

    ids = np.array(ids)
    ids_ord = np.sort(ids)
    all_ids = np.array(list(range(df.shape[1])))
    all_ids[ids_ord] = all_ids[ids]

    df_after = df.iloc[interval_, all_ids].values
    df_after_clean_not = df.iloc[interval_not,].values

    df_conc = np.concatenate((df_before, df_after, df_after_clean_not))
    df_conc = pd.DataFrame(df_conc)
    df_conc.columns = cnames

    before_ = list(range(t_change))
    l = [before_, interval_, interval_not]
    flat_list = [item for sublist in l for item in sublist]

    df_conc.index = flat_list
    df_conc = df_conc.sort_index(inplace=False)

    return (df_conc)
