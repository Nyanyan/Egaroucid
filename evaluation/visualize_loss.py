import matplotlib.pyplot as plt

s0 = '''59	1499 mae 0.00148893 mse 0.0104308 n_data 1165938 n_param 528065 n_used_params 38448
58	1489 mae 1.74769 mse 6.12936 n_data 1169517 n_param 528065 n_used_params 88890
57	1414 mae 2.49957 mse 11.4209 n_data 1173224 n_param 528065 n_used_params 140888
56	1267 mae 3.0038 mse 16.0624 n_data 1176515 n_param 528065 n_used_params 183010
55	1311 mae 3.37525 mse 19.9302 n_data 1179408 n_param 528065 n_used_params 213372
54	1358 mae 3.62279 mse 22.8484 n_data 1181643 n_param 528065 n_used_params 231890
53	1321 mae 3.83121 mse 25.3558 n_data 1183383 n_param 528065 n_used_params 243251
52	1305 mae 3.95796 mse 26.9745 n_data 1184759 n_param 528065 n_used_params 249778
51	1296 mae 4.05387 mse 28.2427 n_data 1185784 n_param 528065 n_used_params 254414
50	1137 mae 4.21922 mse 30.3838 n_data 1286683 n_param 528065 n_used_params 291676
49	1129 mae 4.26223 mse 30.964 n_data 1287980 n_param 528065 n_used_params 293713
48	1125 mae 4.28233 mse 31.2933 n_data 1289061 n_param 528065 n_used_params 295259
47	1122 mae 4.28321 mse 31.2712 n_data 1290304 n_param 528065 n_used_params 295553
46	1111 mae 4.27129 mse 31.1107 n_data 1291275 n_param 528065 n_used_params 295733
45	1212 mae 4.17399 mse 29.7443 n_data 1227594 n_param 528065 n_used_params 276827
44	1278 mae 4.12466 mse 29.0365 n_data 1193611 n_param 528065 n_used_params 252857
43	1273 mae 4.09732 mse 28.6996 n_data 1195093 n_param 528065 n_used_params 250365
42	1276 mae 4.07323 mse 28.4229 n_data 1196235 n_param 528065 n_used_params 247929
41	1279 mae 4.04796 mse 28.1452 n_data 1197384 n_param 528065 n_used_params 245136
40	1274 mae 4.03633 mse 28.0665 n_data 1198325 n_param 528065 n_used_params 242219
39	1277 mae 4.01666 mse 27.9385 n_data 1199109 n_param 528065 n_used_params 239013
38	1278 mae 4.03499 mse 28.2824 n_data 1199715 n_param 528065 n_used_params 235933
37	1281 mae 4.04967 mse 28.6217 n_data 1200216 n_param 528065 n_used_params 232203
36	1276 mae 4.12215 mse 29.6958 n_data 1200593 n_param 528065 n_used_params 228684
35	1284 mae 4.31401 mse 32.4666 n_data 1200918 n_param 528065 n_used_params 225121
34	1282 mae 4.44129 mse 34.3792 n_data 1201170 n_param 528065 n_used_params 222030
33	1315 mae 4.55635 mse 36.1724 n_data 1201349 n_param 528065 n_used_params 218459
32	1314 mae 4.66165 mse 37.854 n_data 1201500 n_param 528065 n_used_params 215317
31	1323 mae 4.7632 mse 39.5371 n_data 1201600 n_param 528065 n_used_params 211447
30	1332 mae 4.87188 mse 41.333 n_data 1201640 n_param 528065 n_used_params 208240
29	1352 mae 5.00759 mse 43.6879 n_data 1201671 n_param 528065 n_used_params 204271
28	1359 mae 5.12564 mse 45.8652 n_data 1201677 n_param 528065 n_used_params 200808
27	1368 mae 5.24886 mse 48.1831 n_data 1201679 n_param 528065 n_used_params 196448
26	1390 mae 5.35502 mse 50.1699 n_data 1201680 n_param 528065 n_used_params 192585
25	1404 mae 5.4632 mse 52.3293 n_data 1201684 n_param 528065 n_used_params 187451
24	1417 mae 5.57097 mse 54.4869 n_data 1201689 n_param 528065 n_used_params 183154
23	1435 mae 5.67186 mse 56.6296 n_data 1201694 n_param 528065 n_used_params 176970
22	1457 mae 5.76503 mse 58.6885 n_data 1201700 n_param 528065 n_used_params 171089
21	1481 mae 5.87537 mse 61.04 n_data 1201704 n_param 528065 n_used_params 163712
20	1503 mae 6.0361 mse 64.7211 n_data 1201759 n_param 528065 n_used_params 157078
19	4850 mae 6.31846 mse 72.3195 n_data 533337 n_param 528065 n_used_params 122006
18	4893 mae 6.85128 mse 88.3493 n_data 533350 n_param 528065 n_used_params 112940
17	4972 mae 7.67854 mse 113.779 n_data 533351 n_param 528065 n_used_params 102080
16	5054 mae 8.42606 mse 138.918 n_data 533371 n_param 528065 n_used_params 91650
15	5140 mae 9.42969 mse 172.986 n_data 533381 n_param 528065 n_used_params 80291
14	5268 mae 10.3015 mse 204.385 n_data 533399 n_param 528065 n_used_params 69582
13	5383 mae 11.3833 mse 242.069 n_data 533414 n_param 528065 n_used_params 58528
12	5444 mae 12.3269 mse 277.86 n_data 533433 n_param 528065 n_used_params 48297
11	5638 mae 13.3853 mse 316.836 n_data 533438 n_param 528065 n_used_params 38128
10	5711 mae 14.3282 mse 353.472 n_data 533457 n_param 528065 n_used_params 28462
9	5573 mae 12.7953 mse 293.531 n_data 586611 n_param 528065 n_used_params 17930
8	5719 mae 13.2825 mse 312.279 n_data 586612 n_param 528065 n_used_params 10967
7	5849 mae 13.7817 mse 333.239 n_data 586612 n_param 528065 n_used_params 5405
6	6002 mae 14.1297 mse 347.712 n_data 586612 n_param 528065 n_used_params 2556
5	6128 mae 14.4192 mse 360.674 n_data 586612 n_param 528065 n_used_params 1059
4	6194 mae 14.6635 mse 369.981 n_data 586612 n_param 528065 n_used_params 451
3	6393 mae 14.7849 mse 375.039 n_data 586612 n_param 528065 n_used_params 176
2	6521 mae 14.8826 mse 379.352 n_data 586612 n_param 528065 n_used_params 78
1	6616 mae 14.9114 mse 380.177 n_data 586612 n_param 528065 n_used_params 40
0	6651 mae 14.9114 mse 380.14 n_data 586612 n_param 528065 n_used_params 28'''

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