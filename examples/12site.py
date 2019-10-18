 ---------------------------------------------------------
                       Hubbard model
 ---------------------------------------------------------
 nsite:    12
 beta :  1.000000
 U    :  5.000000
 Clusters:
IDX000:DIM0256:000|001|002|003|
IDX001:DIM0256:004|005|006|007|
IDX002:DIM0256:008|009|010|011|
 Add 1-body terms
 0_____ 1_____ 2_____
 Δa Δb: Δa Δb: Δa Δb:|    0,    1,    2,
  0  0:  0  0:  0  0:|   Aa, ____, ____,
  0  0:  0  0:  0  0:|   Bb, ____, ____,
  0  0:  0  0:  0  0:| ____,   Aa, ____,
  0  0:  0  0:  0  0:| ____,   Bb, ____,
  0  0:  0  0:  0  0:| ____, ____,   Aa,
  0  0:  0  0:  0  0:| ____, ____,   Bb,
  1  0: -1  0:  0  0:|    A,    a, ____,
  0  1:  0 -1:  0  0:|    B,    b, ____,
 -1  0:  1  0:  0  0:|    a,    A, ____,
  0 -1:  0  1:  0  0:|    b,    B, ____,
  0  0:  1  0: -1  0:| ____,    A,    a,
  0  0:  0  1:  0 -1:| ____,    B,    b,
  0  0: -1  0:  1  0:| ____,    a,    A,
  0  0:  0 -1:  0  1:| ____,    b,    B,
 Add 2-body terms
 Build cluster basis
 Extract local operator for cluster 0


 Form basis by diagonalize local Hamiltonian for cluster:  0
 Do CI for each particle number block
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 0    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 0    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 0    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.23606798
 State:    1      -1.00000000
 State:    2       0.00000000
 State:    3       0.00000000
 State:    4       1.00000000
 State:    5       2.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 0    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 0    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 1    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 1    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.56635019
 State:    1      -2.23606798
 State:    2      -1.36257857
 State:    3      -1.00000000
 State:    4      -0.49213091
 State:    5      -0.21175677
 State:    6       0.00000000
 State:    7       0.00000000
 State:    8       0.64139001
 State:    9       1.00000000
 State:   10       1.84739427
 State:   11       2.23606798
 State:   12       5.00000000
 State:   13       5.22497070
 State:   14       5.72118856
 State:   15       6.19787290
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 1    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.47829280
 State:    1      -1.93358473
 State:    2      -1.61803399
 State:    3      -1.27828284
 State:    4      -0.85570859
 State:    5      -0.61803399
 State:    6      -0.10537777
 State:    7       0.37068792
 State:    8       0.61803399
 State:    9       0.69709818
 State:   10       1.29529769
 State:   11       1.61803399
 State:   12       3.38196601
 State:   13       3.70470231
 State:   14       4.30290182
 State:   15       4.38196601
 State:   16       4.62931208
 State:   17       5.10537777
 State:   18       5.61803399
 State:   19       5.85570859
 State:   20       6.27828284
 State:   21       6.61803399
 State:   22       6.93358473
 State:   23       7.47829280
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 1    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.19787290
 State:    1      -0.72118856
 State:    2      -0.22497070
 State:    3       0.00000000
 State:    4       2.76393202
 State:    5       3.15260573
 State:    6       4.00000000
 State:    7       4.35860999
 State:    8       5.00000000
 State:    9       5.00000000
 State:   10       5.21175677
 State:   11       5.49213091
 State:   12       6.00000000
 State:   13       6.36257857
 State:   14       7.23606798
 State:   15       7.56635019
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 1    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       3.38196601
 State:    1       4.38196601
 State:    2       5.61803399
 State:    3       6.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 2    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.23606798
 State:    1      -1.00000000
 State:    2       0.00000000
 State:    3       0.00000000
 State:    4       1.00000000
 State:    5       2.23606798
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 2    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.47829280
 State:    1      -1.93358473
 State:    2      -1.61803399
 State:    3      -1.27828284
 State:    4      -0.85570859
 State:    5      -0.61803399
 State:    6      -0.10537777
 State:    7       0.37068792
 State:    8       0.61803399
 State:    9       0.69709818
 State:   10       1.29529769
 State:   11       1.61803399
 State:   12       3.38196601
 State:   13       3.70470231
 State:   14       4.30290182
 State:   15       4.38196601
 State:   16       4.62931208
 State:   17       5.10537777
 State:   18       5.61803399
 State:   19       5.85570859
 State:   20       6.27828284
 State:   21       6.61803399
 State:   22       6.93358473
 State:   23       7.47829280
 CI solver:: Dim: 36       NOrb: 4    NAlpha: 2    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.65740584
 State:    1      -1.19787290
 State:    2      -0.72118856
 State:    3      -0.48682350
 State:    4      -0.22497070
 State:    5       0.00000000
 State:    6       2.43364981
 State:    7       2.76393202
 State:    8       2.87698250
 State:    9       3.15260573
 State:   10       3.63742143
 State:   11       4.00000000
 State:   12       4.00000000
 State:   13       4.35860999
 State:   14       4.50786909
 State:   15       4.74210056
 State:   16       4.78824323
 State:   17       5.00000000
 State:   18       5.00000000
 State:   19       5.21175677
 State:   20       5.25789944
 State:   21       5.49213091
 State:   22       5.64139001
 State:   23       6.00000000
 State:   24       6.00000000
 State:   25       6.36257857
 State:   26       6.84739427
 State:   27       7.12301750
 State:   28       7.23606798
 State:   29       7.56635019
 State:   30      10.00000000
 State:   31      10.22497070
 State:   32      10.48682350
 State:   33      10.72118856
 State:   34      11.19787290
 State:   35      11.65740584
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 2    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       2.52170720
 State:    1       3.06641527
 State:    2       3.38196601
 State:    3       3.72171716
 State:    4       4.14429141
 State:    5       4.38196601
 State:    6       4.89462223
 State:    7       5.37068792
 State:    8       5.61803399
 State:    9       5.69709818
 State:   10       6.29529769
 State:   11       6.61803399
 State:   12       8.38196601
 State:   13       8.70470231
 State:   14       9.30290182
 State:   15       9.38196601
 State:   16       9.62931208
 State:   17      10.10537777
 State:   18      10.61803399
 State:   19      10.85570859
 State:   20      11.27828284
 State:   21      11.61803399
 State:   22      11.93358473
 State:   23      12.47829280
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 2    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.76393202
 State:    1       9.00000000
 State:    2      10.00000000
 State:    3      10.00000000
 State:    4      11.00000000
 State:    5      12.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 3    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 3    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.19787290
 State:    1      -0.72118856
 State:    2      -0.22497070
 State:    3       0.00000000
 State:    4       2.76393202
 State:    5       3.15260573
 State:    6       4.00000000
 State:    7       4.35860999
 State:    8       5.00000000
 State:    9       5.00000000
 State:   10       5.21175677
 State:   11       5.49213091
 State:   12       6.00000000
 State:   13       6.36257857
 State:   14       7.23606798
 State:   15       7.56635019
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 3    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       2.52170720
 State:    1       3.06641527
 State:    2       3.38196601
 State:    3       3.72171716
 State:    4       4.14429141
 State:    5       4.38196601
 State:    6       4.89462223
 State:    7       5.37068792
 State:    8       5.61803399
 State:    9       5.69709818
 State:   10       6.29529769
 State:   11       6.61803399
 State:   12       8.38196601
 State:   13       8.70470231
 State:   14       9.30290182
 State:   15       9.38196601
 State:   16       9.62931208
 State:   17      10.10537777
 State:   18      10.61803399
 State:   19      10.85570859
 State:   20      11.27828284
 State:   21      11.61803399
 State:   22      11.93358473
 State:   23      12.47829280
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 3    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.43364981
 State:    1       7.76393202
 State:    2       8.63742143
 State:    3       9.00000000
 State:    4       9.50786909
 State:    5       9.78824323
 State:    6      10.00000000
 State:    7      10.00000000
 State:    8      10.64139001
 State:    9      11.00000000
 State:   10      11.84739427
 State:   11      12.23606798
 State:   12      15.00000000
 State:   13      15.22497070
 State:   14      15.72118856
 State:   15      16.19787290
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 3    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      13.38196601
 State:    1      14.38196601
 State:    2      15.61803399
 State:    3      16.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 4    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 4    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       3.38196601
 State:    1       4.38196601
 State:    2       5.61803399
 State:    3       6.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 4    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.76393202
 State:    1       9.00000000
 State:    2      10.00000000
 State:    3      10.00000000
 State:    4      11.00000000
 State:    5      12.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 4    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      13.38196601
 State:    1      14.38196601
 State:    2      15.61803399
 State:    3      16.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 4    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      20.00000000
 Extract local operator for cluster 1


 Form basis by diagonalize local Hamiltonian for cluster:  1
 Do CI for each particle number block
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 0    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 0    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 0    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.23606798
 State:    1      -1.00000000
 State:    2       0.00000000
 State:    3       0.00000000
 State:    4       1.00000000
 State:    5       2.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 0    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 0    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 1    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 1    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.56635019
 State:    1      -2.23606798
 State:    2      -1.36257857
 State:    3      -1.00000000
 State:    4      -0.49213091
 State:    5      -0.21175677
 State:    6       0.00000000
 State:    7       0.00000000
 State:    8       0.64139001
 State:    9       1.00000000
 State:   10       1.84739427
 State:   11       2.23606798
 State:   12       5.00000000
 State:   13       5.22497070
 State:   14       5.72118856
 State:   15       6.19787290
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 1    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.47829280
 State:    1      -1.93358473
 State:    2      -1.61803399
 State:    3      -1.27828284
 State:    4      -0.85570859
 State:    5      -0.61803399
 State:    6      -0.10537777
 State:    7       0.37068792
 State:    8       0.61803399
 State:    9       0.69709818
 State:   10       1.29529769
 State:   11       1.61803399
 State:   12       3.38196601
 State:   13       3.70470231
 State:   14       4.30290182
 State:   15       4.38196601
 State:   16       4.62931208
 State:   17       5.10537777
 State:   18       5.61803399
 State:   19       5.85570859
 State:   20       6.27828284
 State:   21       6.61803399
 State:   22       6.93358473
 State:   23       7.47829280
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 1    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.19787290
 State:    1      -0.72118856
 State:    2      -0.22497070
 State:    3       0.00000000
 State:    4       2.76393202
 State:    5       3.15260573
 State:    6       4.00000000
 State:    7       4.35860999
 State:    8       5.00000000
 State:    9       5.00000000
 State:   10       5.21175677
 State:   11       5.49213091
 State:   12       6.00000000
 State:   13       6.36257857
 State:   14       7.23606798
 State:   15       7.56635019
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 1    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       3.38196601
 State:    1       4.38196601
 State:    2       5.61803399
 State:    3       6.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 2    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.23606798
 State:    1      -1.00000000
 State:    2       0.00000000
 State:    3       0.00000000
 State:    4       1.00000000
 State:    5       2.23606798
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 2    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.47829280
 State:    1      -1.93358473
 State:    2      -1.61803399
 State:    3      -1.27828284
 State:    4      -0.85570859
 State:    5      -0.61803399
 State:    6      -0.10537777
 State:    7       0.37068792
 State:    8       0.61803399
 State:    9       0.69709818
 State:   10       1.29529769
 State:   11       1.61803399
 State:   12       3.38196601
 State:   13       3.70470231
 State:   14       4.30290182
 State:   15       4.38196601
 State:   16       4.62931208
 State:   17       5.10537777
 State:   18       5.61803399
 State:   19       5.85570859
 State:   20       6.27828284
 State:   21       6.61803399
 State:   22       6.93358473
 State:   23       7.47829280
 CI solver:: Dim: 36       NOrb: 4    NAlpha: 2    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.65740584
 State:    1      -1.19787290
 State:    2      -0.72118856
 State:    3      -0.48682350
 State:    4      -0.22497070
 State:    5       0.00000000
 State:    6       2.43364981
 State:    7       2.76393202
 State:    8       2.87698250
 State:    9       3.15260573
 State:   10       3.63742143
 State:   11       4.00000000
 State:   12       4.00000000
 State:   13       4.35860999
 State:   14       4.50786909
 State:   15       4.74210056
 State:   16       4.78824323
 State:   17       5.00000000
 State:   18       5.00000000
 State:   19       5.21175677
 State:   20       5.25789944
 State:   21       5.49213091
 State:   22       5.64139001
 State:   23       6.00000000
 State:   24       6.00000000
 State:   25       6.36257857
 State:   26       6.84739427
 State:   27       7.12301750
 State:   28       7.23606798
 State:   29       7.56635019
 State:   30      10.00000000
 State:   31      10.22497070
 State:   32      10.48682350
 State:   33      10.72118856
 State:   34      11.19787290
 State:   35      11.65740584
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 2    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       2.52170720
 State:    1       3.06641527
 State:    2       3.38196601
 State:    3       3.72171716
 State:    4       4.14429141
 State:    5       4.38196601
 State:    6       4.89462223
 State:    7       5.37068792
 State:    8       5.61803399
 State:    9       5.69709818
 State:   10       6.29529769
 State:   11       6.61803399
 State:   12       8.38196601
 State:   13       8.70470231
 State:   14       9.30290182
 State:   15       9.38196601
 State:   16       9.62931208
 State:   17      10.10537777
 State:   18      10.61803399
 State:   19      10.85570859
 State:   20      11.27828284
 State:   21      11.61803399
 State:   22      11.93358473
 State:   23      12.47829280
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 2    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.76393202
 State:    1       9.00000000
 State:    2      10.00000000
 State:    3      10.00000000
 State:    4      11.00000000
 State:    5      12.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 3    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 3    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.19787290
 State:    1      -0.72118856
 State:    2      -0.22497070
 State:    3       0.00000000
 State:    4       2.76393202
 State:    5       3.15260573
 State:    6       4.00000000
 State:    7       4.35860999
 State:    8       5.00000000
 State:    9       5.00000000
 State:   10       5.21175677
 State:   11       5.49213091
 State:   12       6.00000000
 State:   13       6.36257857
 State:   14       7.23606798
 State:   15       7.56635019
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 3    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       2.52170720
 State:    1       3.06641527
 State:    2       3.38196601
 State:    3       3.72171716
 State:    4       4.14429141
 State:    5       4.38196601
 State:    6       4.89462223
 State:    7       5.37068792
 State:    8       5.61803399
 State:    9       5.69709818
 State:   10       6.29529769
 State:   11       6.61803399
 State:   12       8.38196601
 State:   13       8.70470231
 State:   14       9.30290182
 State:   15       9.38196601
 State:   16       9.62931208
 State:   17      10.10537777
 State:   18      10.61803399
 State:   19      10.85570859
 State:   20      11.27828284
 State:   21      11.61803399
 State:   22      11.93358473
 State:   23      12.47829280
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 3    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.43364981
 State:    1       7.76393202
 State:    2       8.63742143
 State:    3       9.00000000
 State:    4       9.50786909
 State:    5       9.78824323
 State:    6      10.00000000
 State:    7      10.00000000
 State:    8      10.64139001
 State:    9      11.00000000
 State:   10      11.84739427
 State:   11      12.23606798
 State:   12      15.00000000
 State:   13      15.22497070
 State:   14      15.72118856
 State:   15      16.19787290
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 3    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      13.38196601
 State:    1      14.38196601
 State:    2      15.61803399
 State:    3      16.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 4    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 4    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       3.38196601
 State:    1       4.38196601
 State:    2       5.61803399
 State:    3       6.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 4    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.76393202
 State:    1       9.00000000
 State:    2      10.00000000
 State:    3      10.00000000
 State:    4      11.00000000
 State:    5      12.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 4    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      13.38196601
 State:    1      14.38196601
 State:    2      15.61803399
 State:    3      16.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 4    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      20.00000000
 Extract local operator for cluster 2


 Form basis by diagonalize local Hamiltonian for cluster:  2
 Do CI for each particle number block
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 0    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 0    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 0    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.23606798
 State:    1      -1.00000000
 State:    2       0.00000000
 State:    3       0.00000000
 State:    4       1.00000000
 State:    5       2.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 0    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 0    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 1    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 1    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.56635019
 State:    1      -2.23606798
 State:    2      -1.36257857
 State:    3      -1.00000000
 State:    4      -0.49213091
 State:    5      -0.21175677
 State:    6       0.00000000
 State:    7       0.00000000
 State:    8       0.64139001
 State:    9       1.00000000
 State:   10       1.84739427
 State:   11       2.23606798
 State:   12       5.00000000
 State:   13       5.22497070
 State:   14       5.72118856
 State:   15       6.19787290
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 1    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.47829280
 State:    1      -1.93358473
 State:    2      -1.61803399
 State:    3      -1.27828284
 State:    4      -0.85570859
 State:    5      -0.61803399
 State:    6      -0.10537777
 State:    7       0.37068792
 State:    8       0.61803399
 State:    9       0.69709818
 State:   10       1.29529769
 State:   11       1.61803399
 State:   12       3.38196601
 State:   13       3.70470231
 State:   14       4.30290182
 State:   15       4.38196601
 State:   16       4.62931208
 State:   17       5.10537777
 State:   18       5.61803399
 State:   19       5.85570859
 State:   20       6.27828284
 State:   21       6.61803399
 State:   22       6.93358473
 State:   23       7.47829280
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 1    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.19787290
 State:    1      -0.72118856
 State:    2      -0.22497070
 State:    3       0.00000000
 State:    4       2.76393202
 State:    5       3.15260573
 State:    6       4.00000000
 State:    7       4.35860999
 State:    8       5.00000000
 State:    9       5.00000000
 State:   10       5.21175677
 State:   11       5.49213091
 State:   12       6.00000000
 State:   13       6.36257857
 State:   14       7.23606798
 State:   15       7.56635019
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 1    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       3.38196601
 State:    1       4.38196601
 State:    2       5.61803399
 State:    3       6.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 2    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.23606798
 State:    1      -1.00000000
 State:    2       0.00000000
 State:    3       0.00000000
 State:    4       1.00000000
 State:    5       2.23606798
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 2    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -2.47829280
 State:    1      -1.93358473
 State:    2      -1.61803399
 State:    3      -1.27828284
 State:    4      -0.85570859
 State:    5      -0.61803399
 State:    6      -0.10537777
 State:    7       0.37068792
 State:    8       0.61803399
 State:    9       0.69709818
 State:   10       1.29529769
 State:   11       1.61803399
 State:   12       3.38196601
 State:   13       3.70470231
 State:   14       4.30290182
 State:   15       4.38196601
 State:   16       4.62931208
 State:   17       5.10537777
 State:   18       5.61803399
 State:   19       5.85570859
 State:   20       6.27828284
 State:   21       6.61803399
 State:   22       6.93358473
 State:   23       7.47829280
 CI solver:: Dim: 36       NOrb: 4    NAlpha: 2    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.65740584
 State:    1      -1.19787290
 State:    2      -0.72118856
 State:    3      -0.48682350
 State:    4      -0.22497070
 State:    5       0.00000000
 State:    6       2.43364981
 State:    7       2.76393202
 State:    8       2.87698250
 State:    9       3.15260573
 State:   10       3.63742143
 State:   11       4.00000000
 State:   12       4.00000000
 State:   13       4.35860999
 State:   14       4.50786909
 State:   15       4.74210056
 State:   16       4.78824323
 State:   17       5.00000000
 State:   18       5.00000000
 State:   19       5.21175677
 State:   20       5.25789944
 State:   21       5.49213091
 State:   22       5.64139001
 State:   23       6.00000000
 State:   24       6.00000000
 State:   25       6.36257857
 State:   26       6.84739427
 State:   27       7.12301750
 State:   28       7.23606798
 State:   29       7.56635019
 State:   30      10.00000000
 State:   31      10.22497070
 State:   32      10.48682350
 State:   33      10.72118856
 State:   34      11.19787290
 State:   35      11.65740584
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 2    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       2.52170720
 State:    1       3.06641527
 State:    2       3.38196601
 State:    3       3.72171716
 State:    4       4.14429141
 State:    5       4.38196601
 State:    6       4.89462223
 State:    7       5.37068792
 State:    8       5.61803399
 State:    9       5.69709818
 State:   10       6.29529769
 State:   11       6.61803399
 State:   12       8.38196601
 State:   13       8.70470231
 State:   14       9.30290182
 State:   15       9.38196601
 State:   16       9.62931208
 State:   17      10.10537777
 State:   18      10.61803399
 State:   19      10.85570859
 State:   20      11.27828284
 State:   21      11.61803399
 State:   22      11.93358473
 State:   23      12.47829280
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 2    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.76393202
 State:    1       9.00000000
 State:    2      10.00000000
 State:    3      10.00000000
 State:    4      11.00000000
 State:    5      12.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 3    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.61803399
 State:    1      -0.61803399
 State:    2       0.61803399
 State:    3       1.61803399
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 3    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      -1.19787290
 State:    1      -0.72118856
 State:    2      -0.22497070
 State:    3       0.00000000
 State:    4       2.76393202
 State:    5       3.15260573
 State:    6       4.00000000
 State:    7       4.35860999
 State:    8       5.00000000
 State:    9       5.00000000
 State:   10       5.21175677
 State:   11       5.49213091
 State:   12       6.00000000
 State:   13       6.36257857
 State:   14       7.23606798
 State:   15       7.56635019
 CI solver:: Dim: 24       NOrb: 4    NAlpha: 3    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       2.52170720
 State:    1       3.06641527
 State:    2       3.38196601
 State:    3       3.72171716
 State:    4       4.14429141
 State:    5       4.38196601
 State:    6       4.89462223
 State:    7       5.37068792
 State:    8       5.61803399
 State:    9       5.69709818
 State:   10       6.29529769
 State:   11       6.61803399
 State:   12       8.38196601
 State:   13       8.70470231
 State:   14       9.30290182
 State:   15       9.38196601
 State:   16       9.62931208
 State:   17      10.10537777
 State:   18      10.61803399
 State:   19      10.85570859
 State:   20      11.27828284
 State:   21      11.61803399
 State:   22      11.93358473
 State:   23      12.47829280
 CI solver:: Dim: 16       NOrb: 4    NAlpha: 3    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.43364981
 State:    1       7.76393202
 State:    2       8.63742143
 State:    3       9.00000000
 State:    4       9.50786909
 State:    5       9.78824323
 State:    6      10.00000000
 State:    7      10.00000000
 State:    8      10.64139001
 State:    9      11.00000000
 State:   10      11.84739427
 State:   11      12.23606798
 State:   12      15.00000000
 State:   13      15.22497070
 State:   14      15.72118856
 State:   15      16.19787290
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 3    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      13.38196601
 State:    1      14.38196601
 State:    2      15.61803399
 State:    3      16.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 4    NBeta: 0     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       0.00000000
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 4    NBeta: 1     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       3.38196601
 State:    1       4.38196601
 State:    2       5.61803399
 State:    3       6.61803399
 CI solver:: Dim: 6        NOrb: 4    NAlpha: 4    NBeta: 2     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0       7.76393202
 State:    1       9.00000000
 State:    2      10.00000000
 State:    3      10.00000000
 State:    4      11.00000000
 State:    5      12.23606798
 CI solver:: Dim: 4        NOrb: 4    NAlpha: 4    NBeta: 3     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      13.38196601
 State:    1      14.38196601
 State:    2      15.61803399
 State:    3      16.61803399
 CI solver:: Dim: 1        NOrb: 4    NAlpha: 4    NBeta: 4     Status: initialized
 Diagonalize Matrix for 1000 roots
 Eigenvalues of CI matrix:
 State:    0      20.00000000
 Build these local operators
 Build mats for cluster  0
 Build mats for cluster  1
 Build mats for cluster  2

 ===================================================================
     Selected CI Iteration:    0 epsilon:   0.00010000
 ===================================================================
 Build full Hamiltonian
 Diagonalize Hamiltonian Matrix:
 Ground state of CI:                  -4.97221752  CI Dim:    1 
 Clip CI Vector: thresh =  0.0001
 Old CI Dim:  1
 New CI Dim:  1
 Compute Matrix Vector Product:
 Variance:   2.00000000
 Remove CI space from pt_vector vector
 Norm of CI vector =   1.00000000
 Dimension of CI space:  1
 Dimension of PT space:  4638
 Compute Denominator
 Now compute tasks:
 done.
 PT2 Energy Correction =  -0.17615686
 PT2 Energy Total      =  -5.14837438
 Choose which states to add to CI space
 Next iteration CI space dimension 21

 ===================================================================
     Selected CI Iteration:    1 epsilon:   0.00010000
 ===================================================================
 Build full Hamiltonian
 Diagonalize Hamiltonian Matrix:
 Ground state of CI:                  -5.00657368  CI Dim:   21 
 Clip CI Vector: thresh =  0.0001
 Old CI Dim:  21
 New CI Dim:  21
 Compute Matrix Vector Product:
 Variance:   2.09001562
 Remove CI space from pt_vector vector
 Norm of CI vector =   1.00000000
 Dimension of CI space:  21
 Dimension of PT space:  33655
 Compute Denominator
 Now compute tasks:
 done.
 PT2 Energy Correction =  -0.17892610
 PT2 Energy Total      =  -5.18549978
 Choose which states to add to CI space
 Next iteration CI space dimension 53

 ===================================================================
     Selected CI Iteration:    2 epsilon:   0.00010000
 ===================================================================
 Build full Hamiltonian
 Diagonalize Hamiltonian Matrix:
 Ground state of CI:                  -5.04939394  CI Dim:   53 
 Clip CI Vector: thresh =  0.0001
 Old CI Dim:  53
 New CI Dim:  53
 Compute Matrix Vector Product:
 Variance:   2.12767052
 Remove CI space from pt_vector vector
 Norm of CI vector =   1.00000000
 Dimension of CI space:  53
 Dimension of PT space:  61647
 Compute Denominator
 Now compute tasks:
 done.
 PT2 Energy Correction =  -0.17812000
 PT2 Energy Total      =  -5.22751394
 Choose which states to add to CI space
 Next iteration CI space dimension 61

 ===================================================================
     Selected CI Iteration:    3 epsilon:   0.00010000
 ===================================================================
 Build full Hamiltonian
 Diagonalize Hamiltonian Matrix:
 Ground state of CI:                  -5.05131124  CI Dim:   61 
 Clip CI Vector: thresh =  0.0001
 Old CI Dim:  61
 New CI Dim:  61
 Compute Matrix Vector Product:
 Variance:   2.23418504
 Remove CI space from pt_vector vector
 Norm of CI vector =   1.00000000
 Dimension of CI space:  61
 Dimension of PT space:  71319
 Compute Denominator
 Now compute tasks:
 done.
 PT2 Energy Correction =  -0.18513233
 PT2 Energy Total      =  -5.23644357
 Choose which states to add to CI space
 Next iteration CI space dimension 65

 ===================================================================
     Selected CI Iteration:    4 epsilon:   0.00010000
 ===================================================================
 Build full Hamiltonian
 Diagonalize Hamiltonian Matrix:
 Ground state of CI:                  -5.05545037  CI Dim:   65 
 Clip CI Vector: thresh =  0.0001
 Old CI Dim:  65
 New CI Dim:  65
 Compute Matrix Vector Product:
 Variance:   2.24265395
 Remove CI space from pt_vector vector
 Norm of CI vector =   1.00000000
 Dimension of CI space:  65
 Dimension of PT space:  74315
 Compute Denominator
 Now compute tasks:
