#!/usr/bin/miniconda3/bin/ python
import os
import re
import math
import pandas as pd
import numpy as np

# feature engineering: composition(C), transition(T), distribution(D), conjoint_triad(CT)

## composition


def Count(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum


def CTDC(seq, **kw):
    group1 = {
        "hydrophobicity_PRAM900101": "RKEDQN",
        "hydrophobicity_ARGP820101": "QSTNGDE",
        "hydrophobicity_ZIMJ680101": "QNGSWTDERA",
        "hydrophobicity_PONP930101": "KPDESNQT",
        "hydrophobicity_CASG920101": "KDEQPSRNTG",
        "hydrophobicity_ENGD860101": "RDKENQHYP",
        "hydrophobicity_FASG890101": "KERSQD",
        "normwaalsvolume": "GASTPDC",
        "polarity": "LIFWCMVY",
        "polarizability": "GASDT",
        "charge": "KR",
        "secondarystruct": "EALMQKRH",
        "solventaccess": "ALFCGIVW",
    }
    group2 = {
        "hydrophobicity_PRAM900101": "GASTPHY",
        "hydrophobicity_ARGP820101": "RAHCKMV",
        "hydrophobicity_ZIMJ680101": "HMCKV",
        "hydrophobicity_PONP930101": "GRHA",
        "hydrophobicity_CASG920101": "AHYMLV",
        "hydrophobicity_ENGD860101": "SGTAW",
        "hydrophobicity_FASG890101": "NTPG",
        "normwaalsvolume": "NVEQIL",
        "polarity": "PATGS",
        "polarizability": "CPNVEQIL",
        "charge": "ANCQGHILMFPSTWYV",
        "secondarystruct": "VIYCWFT",
        "solventaccess": "RKQEND",
    }
    group3 = {
        "hydrophobicity_PRAM900101": "CLVIMFW",
        "hydrophobicity_ARGP820101": "LYPFIW",
        "hydrophobicity_ZIMJ680101": "LPFYI",
        "hydrophobicity_PONP930101": "YMFWLCVI",
        "hydrophobicity_CASG920101": "FIWC",
        "hydrophobicity_ENGD860101": "CVLIMF",
        "hydrophobicity_FASG890101": "AYHWVMFLIC",
        "normwaalsvolume": "MHKFRYW",
        "polarity": "HQRKNED",
        "polarizability": "KMHFRYW",
        "charge": "DE",
        "secondarystruct": "GNPSD",
        "solventaccess": "MSPTHY",
    }

    groups = [group1, group2, group3]
    property = (
        "hydrophobicity_PRAM900101",
        "hydrophobicity_ARGP820101",
        "hydrophobicity_ZIMJ680101",
        "hydrophobicity_PONP930101",
        "hydrophobicity_CASG920101",
        "hydrophobicity_ENGD860101",
        "hydrophobicity_FASG890101",
        "normwaalsvolume",
        "polarity",
        "polarizability",
        "charge",
        "secondarystruct",
        "solventaccess",
    )

    encodings = []
    header = ["#"]
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append(p + ".G" + str(g))
    # encodings.append(header)
    for i in range(len(seq)):
        sequence = seq[i]
        code = []
        for p in property:
            c1 = Count(group1[p], sequence) / len(sequence)
            c2 = Count(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
        encodings.append(code)
    return encodings  # (x, 39)


## transition


def CTDT(seq, **kw):
    group1 = {
        "hydrophobicity_PRAM900101": "RKEDQN",
        "hydrophobicity_ARGP820101": "QSTNGDE",
        "hydrophobicity_ZIMJ680101": "QNGSWTDERA",
        "hydrophobicity_PONP930101": "KPDESNQT",
        "hydrophobicity_CASG920101": "KDEQPSRNTG",
        "hydrophobicity_ENGD860101": "RDKENQHYP",
        "hydrophobicity_FASG890101": "KERSQD",
        "normwaalsvolume": "GASTPDC",
        "polarity": "LIFWCMVY",
        "polarizability": "GASDT",
        "charge": "KR",
        "secondarystruct": "EALMQKRH",
        "solventaccess": "ALFCGIVW",
    }
    group2 = {
        "hydrophobicity_PRAM900101": "GASTPHY",
        "hydrophobicity_ARGP820101": "RAHCKMV",
        "hydrophobicity_ZIMJ680101": "HMCKV",
        "hydrophobicity_PONP930101": "GRHA",
        "hydrophobicity_CASG920101": "AHYMLV",
        "hydrophobicity_ENGD860101": "SGTAW",
        "hydrophobicity_FASG890101": "NTPG",
        "normwaalsvolume": "NVEQIL",
        "polarity": "PATGS",
        "polarizability": "CPNVEQIL",
        "charge": "ANCQGHILMFPSTWYV",
        "secondarystruct": "VIYCWFT",
        "solventaccess": "RKQEND",
    }
    group3 = {
        "hydrophobicity_PRAM900101": "CLVIMFW",
        "hydrophobicity_ARGP820101": "LYPFIW",
        "hydrophobicity_ZIMJ680101": "LPFYI",
        "hydrophobicity_PONP930101": "YMFWLCVI",
        "hydrophobicity_CASG920101": "FIWC",
        "hydrophobicity_ENGD860101": "CVLIMF",
        "hydrophobicity_FASG890101": "AYHWVMFLIC",
        "normwaalsvolume": "MHKFRYW",
        "polarity": "HQRKNED",
        "polarizability": "KMHFRYW",
        "charge": "DE",
        "secondarystruct": "GNPSD",
        "solventaccess": "MSPTHY",
    }

    groups = [group1, group2, group3]
    property = (
        "hydrophobicity_PRAM900101",
        "hydrophobicity_ARGP820101",
        "hydrophobicity_ZIMJ680101",
        "hydrophobicity_PONP930101",
        "hydrophobicity_CASG920101",
        "hydrophobicity_ENGD860101",
        "hydrophobicity_FASG890101",
        "normwaalsvolume",
        "polarity",
        "polarizability",
        "charge",
        "secondarystruct",
        "solventaccess",
    )

    encodings = []
    header = ["#"]
    for p in property:
        for tr in ("Tr1221", "Tr1331", "Tr2332"):
            header.append(p + "." + tr)
    # encodings.append(header)

    for i in range(len(seq)):
        sequence = seq[i]
        code = []
        aaPair = [sequence[j : j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (
                    pair[0] in group2[p] and pair[1] in group1[p]
                ):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (
                    pair[0] in group3[p] and pair[1] in group1[p]
                ):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (
                    pair[0] in group3[p] and pair[1] in group2[p]
                ):
                    c2332 = c2332 + 1
            code = code + [
                c1221 / len(aaPair),
                c1331 / len(aaPair),
                c2332 / len(aaPair),
            ]
        encodings.append(code)
    return encodings  # (x, 39)


## distribution


def count(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [
        1,
        math.floor(0.25 * number),
        math.floor(0.50 * number),
        math.floor(0.75 * number),
        number,
    ]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code


def CTDD(seq, **kw):
    group1 = {
        "hydrophobicity_PRAM900101": "RKEDQN",
        "hydrophobicity_ARGP820101": "QSTNGDE",
        "hydrophobicity_ZIMJ680101": "QNGSWTDERA",
        "hydrophobicity_PONP930101": "KPDESNQT",
        "hydrophobicity_CASG920101": "KDEQPSRNTG",
        "hydrophobicity_ENGD860101": "RDKENQHYP",
        "hydrophobicity_FASG890101": "KERSQD",
        "normwaalsvolume": "GASTPDC",
        "polarity": "LIFWCMVY",
        "polarizability": "GASDT",
        "charge": "KR",
        "secondarystruct": "EALMQKRH",
        "solventaccess": "ALFCGIVW",
    }
    group2 = {
        "hydrophobicity_PRAM900101": "GASTPHY",
        "hydrophobicity_ARGP820101": "RAHCKMV",
        "hydrophobicity_ZIMJ680101": "HMCKV",
        "hydrophobicity_PONP930101": "GRHA",
        "hydrophobicity_CASG920101": "AHYMLV",
        "hydrophobicity_ENGD860101": "SGTAW",
        "hydrophobicity_FASG890101": "NTPG",
        "normwaalsvolume": "NVEQIL",
        "polarity": "PATGS",
        "polarizability": "CPNVEQIL",
        "charge": "ANCQGHILMFPSTWYV",
        "secondarystruct": "VIYCWFT",
        "solventaccess": "RKQEND",
    }
    group3 = {
        "hydrophobicity_PRAM900101": "CLVIMFW",
        "hydrophobicity_ARGP820101": "LYPFIW",
        "hydrophobicity_ZIMJ680101": "LPFYI",
        "hydrophobicity_PONP930101": "YMFWLCVI",
        "hydrophobicity_CASG920101": "FIWC",
        "hydrophobicity_ENGD860101": "CVLIMF",
        "hydrophobicity_FASG890101": "AYHWVMFLIC",
        "normwaalsvolume": "MHKFRYW",
        "polarity": "HQRKNED",
        "polarizability": "KMHFRYW",
        "charge": "DE",
        "secondarystruct": "GNPSD",
        "solventaccess": "MSPTHY",
    }

    groups = [group1, group2, group3]
    property = (
        "hydrophobicity_PRAM900101",
        "hydrophobicity_ARGP820101",
        "hydrophobicity_ZIMJ680101",
        "hydrophobicity_PONP930101",
        "hydrophobicity_CASG920101",
        "hydrophobicity_ENGD860101",
        "hydrophobicity_FASG890101",
        "normwaalsvolume",
        "polarity",
        "polarizability",
        "charge",
        "secondarystruct",
        "solventaccess",
    )

    encodings = []
    header = ["#"]
    for p in property:
        for g in ("1", "2", "3"):
            for d in ["0", "25", "50", "75", "100"]:
                header.append(p + "." + g + ".residue" + d)
    # encodings.append(header)

    for i in range(len(seq)):
        sequence = seq[i]
        code = []
        for p in property:
            code = (
                code
                + count(group1[p], sequence)
                + count(group2[p], sequence)
                + count(group3[p], sequence)
            )
        encodings.append(code)

    return encodings  # (x, 195)


## Conjoint-triad

AALetter = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

_repmat = {
    1: ["A", "G", "V"],
    2: ["I", "L", "F", "P"],
    3: ["Y", "M", "T", "S"],
    4: ["H", "N", "Q", "W"],
    5: ["R", "K"],
    6: ["D", "E"],
    7: ["C"],
}


def _Str2Num(proteinsequence):
    """
    translate the amino acid letter into the corresponding class based on the
    given form.
    """
    repmat = {}
    for i in _repmat:
        for j in _repmat[i]:
            repmat[j] = i

    res = proteinsequence
    for i in repmat:
        res = res.replace(i, str(repmat[i]))
    return res


def CalculateConjointTriad(proteinsequence):
    """
    Calculate the conjoint triad features from protein sequence.
    Useage:
    res = CalculateConjointTriad(protein)
    Input: protein is a pure protein sequence.
    Output is a dict form containing all 343 conjoint triad features.
    """
    res = {}
    proteinnum = _Str2Num(proteinsequence)
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                temp = str(i) + str(j) + str(k)
                res[temp] = proteinnum.count(temp)
    return res


def CT_processing(sequences):
    code = []
    for i in sequences:
        DPC = CalculateConjointTriad(i)
        ct = list(DPC.values())
        code.append(ct)

    return code


# feature extraction

positive = pd.read_csv("~/RBProtein/new_Data/Data/RBP.csv")
positive_sequence = positive["sequence"].values
positive_label = positive["label"]

decoys = pd.read_csv("~/RBProtein/new_Data/Data/non_RBP.csv")
decoys_sequence = decoys["sequence"]
decoys_label = decoys["label"]

sequence_list = np.append(positive_sequence, decoys_sequence)
labels = positive_label.append(decoys_label)
label_list = labels.values
print(label_list)


print(len(sequence_list))

composition = CTDC(sequence_list)
composition_CTD = pd.DataFrame(composition)
ctdc = np.array(composition_CTD)
print(ctdc.shape)

transition = CTDT(sequence_list)
transition_CTD = pd.DataFrame(transition)
ctdt = np.array(transition_CTD)
print(ctdt.shape)


distribution = CTDD(sequence_list)
distribution_CTD = pd.DataFrame(distribution)
ctdd = np.array(distribution_CTD)
print(ctdd.shape)


CT = CT_processing(sequence_list)
conjoint_triad = np.array(CT)
print(conjoint_triad.shape)  # (, 343)

targets = pd.DataFrame(label_list)
labels = np.array(targets)

# feature concatenation
features = np.concatenate((ctdc, ctdt, ctdd, conjoint_triad), axis=1)
print(features.shape)
dir_input = "features/"
os.makedirs(dir_input, exist_ok=True)

np.save(dir_input + "composition", ctdc)  # 39
np.save(dir_input + "transition", ctdt)  # 39
np.save(dir_input + "distribution", ctdd)  # 195
np.save(dir_input + "conjoint_triad", conjoint_triad)  # 343
np.save(dir_input + "features", features)  # 616
np.save(dir_input + "labels", labels)  # 14725


print("The preprocess of dataset has finished!")

