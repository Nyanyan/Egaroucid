import matplotlib.pyplot as plt

s0 = '''59	723 mae 0.00280718 mse 0.0121957 n_data 1165938 n_param 528065 n_used_params 38448
58	658 mae 1.76168 mse 6.23564 n_data 1169517 n_param 528065 n_used_params 88890
57	636 mae 2.51387 mse 11.5428 n_data 1173224 n_param 528065 n_used_params 140888
56	642 mae 3.01714 mse 16.1961 n_data 1176515 n_param 528065 n_used_params 183010
55	638 mae 3.38831 mse 20.081 n_data 1179408 n_param 528065 n_used_params 213372
54	662 mae 3.63898 mse 23.022 n_data 1181643 n_param 528065 n_used_params 231890
53	665 mae 3.84542 mse 25.5073 n_data 1183383 n_param 528065 n_used_params 243251
52	655 mae 3.97165 mse 27.1472 n_data 1184759 n_param 528065 n_used_params 249778
51	652 mae 4.06673 mse 28.3911 n_data 1185784 n_param 528065 n_used_params 254414
50	567 mae 4.23687 mse 30.6119 n_data 1286683 n_param 528065 n_used_params 291676
49	566 mae 4.27945 mse 31.1817 n_data 1287980 n_param 528065 n_used_params 293713
48	555 mae 4.30133 mse 31.5362 n_data 1289061 n_param 528065 n_used_params 295259
47	562 mae 4.30082 mse 31.4899 n_data 1290304 n_param 528065 n_used_params 295553
46	554 mae 4.29059 mse 31.3535 n_data 1291275 n_param 528065 n_used_params 295733
45	611 mae 4.19188 mse 29.958 n_data 1227594 n_param 528065 n_used_params 276827
44	634 mae 4.14092 mse 29.2246 n_data 1193611 n_param 528065 n_used_params 252857
43	634 mae 4.11487 mse 28.8957 n_data 1195093 n_param 528065 n_used_params 250365
42	637 mae 4.09044 mse 28.6219 n_data 1196235 n_param 528065 n_used_params 247929
41	637 mae 4.06505 mse 28.3506 n_data 1197384 n_param 528065 n_used_params 245136
40	636 mae 4.05566 mse 28.3069 n_data 1198325 n_param 528065 n_used_params 242219
39	636 mae 4.03707 mse 28.1885 n_data 1199109 n_param 528065 n_used_params 239013
38	641 mae 4.05767 mse 28.5632 n_data 1199715 n_param 528065 n_used_params 235933
37	641 mae 4.07249 mse 28.905 n_data 1200216 n_param 528065 n_used_params 232203
36	646 mae 4.14632 mse 30.014 n_data 1200593 n_param 528065 n_used_params 228684
35	654 mae 4.3371 mse 32.7795 n_data 1200918 n_param 528065 n_used_params 225121
34	669 mae 4.46507 mse 34.7003 n_data 1201170 n_param 528065 n_used_params 222030
33	654 mae 4.58389 mse 36.5515 n_data 1201349 n_param 528065 n_used_params 218459
32	660 mae 4.68968 mse 38.2466 n_data 1201500 n_param 528065 n_used_params 215317
31	658 mae 4.79251 mse 39.9427 n_data 1201600 n_param 528065 n_used_params 211447
30	662 mae 4.90133 mse 41.7653 n_data 1201640 n_param 528065 n_used_params 208240
29	666 mae 5.03682 mse 44.1279 n_data 1201671 n_param 528065 n_used_params 204271
28	674 mae 5.15463 mse 46.3163 n_data 1201677 n_param 528065 n_used_params 200808
27	680 mae 5.27746 mse 48.6288 n_data 1201679 n_param 528065 n_used_params 196448
26	684 mae 5.38456 mse 50.639 n_data 1201680 n_param 528065 n_used_params 192585
25	696 mae 5.49118 mse 52.7774 n_data 1201684 n_param 528065 n_used_params 187451
24	703 mae 5.59842 mse 54.9408 n_data 1201689 n_param 528065 n_used_params 183154
23	714 mae 5.69938 mse 57.0793 n_data 1201694 n_param 528065 n_used_params 176970
22	724 mae 5.79257 mse 59.1477 n_data 1201700 n_param 528065 n_used_params 171089
21	739 mae 5.9037 mse 61.4987 n_data 1201704 n_param 528065 n_used_params 163712
20	747 mae 6.06416 mse 65.2218 n_data 1201759 n_param 528065 n_used_params 157078
19	2395 mae 6.3869 mse 74.0933 n_data 533337 n_param 528065 n_used_params 122006
18	2457 mae 6.9096 mse 90.1371 n_data 533350 n_param 528065 n_used_params 112940
17	2470 mae 7.73612 mse 115.762 n_data 533351 n_param 528065 n_used_params 102080
16	2531 mae 8.47811 mse 140.94 n_data 533371 n_param 528065 n_used_params 91650
15	2575 mae 9.47774 mse 175.018 n_data 533381 n_param 528065 n_used_params 80291
14	2618 mae 10.3431 mse 206.313 n_data 533399 n_param 528065 n_used_params 69582
13	2634 mae 11.4229 mse 243.926 n_data 533414 n_param 528065 n_used_params 58528
12	2706 mae 12.3594 mse 279.493 n_data 533433 n_param 528065 n_used_params 48297
11	2788 mae 13.4115 mse 318.265 n_data 533438 n_param 528065 n_used_params 38128
10	2851 mae 14.3471 mse 354.528 n_data 533457 n_param 528065 n_used_params 28462
9	2782 mae 12.8077 mse 294.265 n_data 586611 n_param 528065 n_used_params 17930
8	2828 mae 13.2913 mse 312.779 n_data 586612 n_param 528065 n_used_params 10967
7	2930 mae 13.7883 mse 333.511 n_data 586612 n_param 528065 n_used_params 5405'''

x0 = []
y0 = []
for line in s0.splitlines():
    line_split = line.split()
    if int(line_split[0]) > 20:
        x0.append(int(line_split[0]))
        y0.append(float(line_split[3]))

s1 = '''29	314 600 345.584 1.34994
28	274 600 771.565 3.01393
27	254 600 950.331 3.71223
26	246 600 1041.44 4.06813
25	226 601 1105.64 4.31892
24	205 600 1128.64 4.40876
23	202 600 1123.77 4.38973
22	229 600 1068.41 4.17347
21	234 600 1042.92 4.07392
20	232 601 1025.86 4.00728
19	231 600 1015.66 3.96744
18	233 602 1023.72 3.99889
17	233 601 1086.98 4.24601
16	237 601 1143.84 4.46811
15	238 601 1196.85 4.6752
14	242 602 1269.22 4.95789
13	245 600 1338.92 5.23016
12	251 601 1402.53 5.47864
11	258 601 1465.15 5.72325
10	268 602 1532.73 5.98722'''

x1 = []
y1 = []
for line in s1.splitlines():
    line_split = line.split()
    x1.append(int(line_split[0]) * 2)
    y1.append(float(line_split[4]))


plt.plot(x0, y0, label='data_0')
plt.plot(x1, y1, label='data_1')
plt.show()