"""
    The functions in this file is used to get the common and unique samples between any 2 views.

    Author: Shihao Ma at WangLab, Dec. 14, 2020
"""

import numpy as np
from collections import defaultdict

def data_indexing(matrices):
    """
    Performs data indexing on input expression matrices

    Parameters
    ----------
    matrices : (M, N) array_like
        Input expression matrices, with gene/feature in columns and sample in row.

    Returns
    -------
    matrices_pure: Expression matrices without the first column and first row
    dict_commonSample: dictionaries that give you the common samples between 2 views
    dict_uniqueSample: dictionaries that give you the unique samples between 2 views
    original_order: the original order of samples for each view
    """

    if len(matrices) < 1:
        print("Input nothing, return nothing")
        return None

    print("Start indexing input expression matrices!")

    original_order = [0] * (len(matrices))
    dict_original_order = {}
    dict_commonSample = {}
    dict_uniqueSample = {}
    dict_commonSampleIndex = {}
    dict_sampleToIndexs = defaultdict(list)

    for i in range(0, len(matrices)):
        original_order[i] = list(matrices[i].index)
        dict_original_order[i] = original_order[i]
        for sample in original_order[i]:
            dict_sampleToIndexs[sample].append(
                (i, np.argwhere(matrices[i].index == sample).squeeze().tolist())
            )

    for i in range(0, len(original_order)):
        for j in range(i + 1, len(original_order)):
            commonList = list(set(original_order[i]).intersection(original_order[j]))
            print("Common sample between view{} and view{}: {}".format(i, j, len(commonList)))
            dict_commonSample.update(
                dict_commonSample.fromkeys([(i, j), (j, i)], commonList)
            )
            dict_commonSampleIndex[(i, j)] = [
                np.argwhere(matrices[i].index == x).squeeze().tolist()
                for x in commonList
            ]
            dict_commonSampleIndex[(j, i)] = [
                np.argwhere(matrices[j].index == x).squeeze().tolist()
                for x in commonList
            ]

            dict_uniqueSample[(i, j)] = list(
                set(original_order[i]).symmetric_difference(commonList)
            )
            dict_uniqueSample[(j, i)] = list(
                set(original_order[j]).symmetric_difference(commonList)
            )

    return (
        dict_commonSample,
        dict_commonSampleIndex,
        dict_sampleToIndexs,
        dict_uniqueSample,
        original_order,
        dict_original_order,
    )