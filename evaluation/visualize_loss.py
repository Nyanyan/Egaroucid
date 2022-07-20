import matplotlib.pyplot as plt

s0 = '''59	733 mae 0.00148207 mse 0.00747858 n_data 1165938 n_param 528065 n_used_params 37671
58	670 mae 1.72615 mse 6.06521 n_data 1169517 n_param 528065 n_used_params 88370
57	653 mae 2.46903 mse 11.2146 n_data 1173224 n_param 528065 n_used_params 140766
56	654 mae 2.97058 mse 15.7668 n_data 1176515 n_param 528065 n_used_params 183294
55	1342 mae 3.31676 mse 19.3186 n_data 1179408 n_param 528065 n_used_params 213017
54	1326 mae 3.5662 mse 22.1804 n_data 1181643 n_param 528065 n_used_params 231032
53	1299 mae 3.77258 mse 24.634 n_data 1183383 n_param 528065 n_used_params 241441
52	1285 mae 3.89915 mse 26.2325 n_data 1184759 n_param 528065 n_used_params 247185
51	1269 mae 3.99693 mse 27.4833 n_data 1185784 n_param 528065 n_used_params 250782
50	1105 mae 4.16247 mse 29.5757 n_data 1286683 n_param 528065 n_used_params 287935
49	1097 mae 4.20246 mse 30.1241 n_data 1287980 n_param 528065 n_used_params 289377
48	1100 mae 4.22395 mse 30.4767 n_data 1289061 n_param 528065 n_used_params 290352
47	1095 mae 4.22747 mse 30.4613 n_data 1290304 n_param 528065 n_used_params 290265
46	1094 mae 4.21452 mse 30.288 n_data 1291275 n_param 528065 n_used_params 290330
45	1198 mae 4.11841 mse 28.9664 n_data 1227594 n_param 528065 n_used_params 271288
44	1244 mae 4.06879 mse 28.307 n_data 1193611 n_param 528065 n_used_params 246921
43	1240 mae 4.04297 mse 27.9701 n_data 1195093 n_param 528065 n_used_params 245026
42	1234 mae 4.02323 mse 27.7374 n_data 1196235 n_param 528065 n_used_params 243300
41	1238 mae 3.99501 mse 27.437 n_data 1197384 n_param 528065 n_used_params 241383
40	1241 mae 3.98701 mse 27.3964 n_data 1198325 n_param 528065 n_used_params 239618
39	1240 mae 3.96744 mse 27.2609 n_data 1199109 n_param 528065 n_used_params 237650
38	1250 mae 3.98501 mse 27.6067 n_data 1199715 n_param 528065 n_used_params 235696
37	1252 mae 4.00047 mse 27.9472 n_data 1200216 n_param 528065 n_used_params 233115
36	571 mae 4.11399 mse 29.5397 n_data 1200593 n_param 528065 n_used_params 230682
35	588 mae 4.30888 mse 32.3274 n_data 1200918 n_param 528065 n_used_params 228331
34	589 mae 4.43721 mse 34.2522 n_data 1201170 n_param 528065 n_used_params 226294
33	586 mae 4.56017 mse 36.1269 n_data 1201349 n_param 528065 n_used_params 223857
32	590 mae 4.66557 mse 37.8001 n_data 1201500 n_param 528065 n_used_params 221612
31	634 mae 4.76428 mse 39.4352 n_data 1201600 n_param 528065 n_used_params 218624
30	645 mae 4.87246 mse 41.2529 n_data 1201640 n_param 528065 n_used_params 216133
29	646 mae 5.00876 mse 43.6035 n_data 1201671 n_param 528065 n_used_params 212875
28	656 mae 5.12748 mse 45.7825 n_data 1201677 n_param 528065 n_used_params 209744
27	657 mae 5.25139 mse 48.1207 n_data 1201679 n_param 528065 n_used_params 205564
26	670 mae 5.35998 mse 50.1345 n_data 1201680 n_param 528065 n_used_params 201941
25	676 mae 5.46805 mse 52.2879 n_data 1201684 n_param 528065 n_used_params 196749
24	684 mae 5.57723 mse 54.5016 n_data 1201689 n_param 528065 n_used_params 192171
23	700 mae 5.68088 mse 56.641 n_data 1201694 n_param 528065 n_used_params 185618
22	712 mae 5.77615 mse 58.7431 n_data 1201700 n_param 528065 n_used_params 179056
21	727 mae 5.889 mse 61.1443 n_data 1201704 n_param 528065 n_used_params 170939
20	751 mae 6.04731 mse 64.8435 n_data 1201759 n_param 528065 n_used_params 163472
19	2393 mae 6.36952 mse 73.6586 n_data 533337 n_param 528065 n_used_params 126414
18	2381 mae 6.89784 mse 89.7818 n_data 533350 n_param 528065 n_used_params 116919
17	2447 mae 7.71937 mse 115.229 n_data 533351 n_param 528065 n_used_params 105656
16	2492 mae 8.4651 mse 140.555 n_data 533371 n_param 528065 n_used_params 94777
15	2546 mae 9.46746 mse 174.525 n_data 533381 n_param 528065 n_used_params 83115
14	2574 mae 10.3309 mse 205.711 n_data 533399 n_param 528065 n_used_params 72059
13	2627 mae 11.4117 mse 243.333 n_data 533414 n_param 528065 n_used_params 60697
12	2673 mae 12.3503 mse 278.916 n_data 533433 n_param 528065 n_used_params 50074
11	2761 mae 13.4044 mse 317.776 n_data 533438 n_param 528065 n_used_params 39574
10	2817 mae 14.3434 mse 354.258 n_data 533457 n_param 528065 n_used_params 29422
9	2761 mae 12.8046 mse 294.112 n_data 586611 n_param 528065 n_used_params 18507
8	2763 mae 13.2899 mse 312.756 n_data 586612 n_param 528065 n_used_params 11235
7	2782 mae 13.7888 mse 333.544 n_data 586612 n_param 528065 n_used_params 5517
6	2848 mae 14.1357 mse 347.941 n_data 586612 n_param 528065 n_used_params 2599
5	2933 mae 14.421 mse 360.81 n_data 586612 n_param 528065 n_used_params 1070
4	2897 mae 14.6651 mse 370.023 n_data 586612 n_param 528065 n_used_params 454
3	3076 mae 14.7845 mse 375.038 n_data 586612 n_param 528065 n_used_params 176
2	3148 mae 14.8826 mse 379.352 n_data 586612 n_param 528065 n_used_params 78
1	3281 mae 14.9114 mse 380.156 n_data 586612 n_param 528065 n_used_params 40
0	3278 mae 14.9114 mse 380.146 n_data 586612 n_param 528065 n_used_params 28'''

x0 = []
y0 = []
for line in s0.splitlines():
    line_split = line.split()
    #if int(line_split[0]) > 20:
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
10	268 602 1532.73 5.98722
9  1 428 1694.84 6.62046
8  1 454 1786.29 6.97769
7  1 496 1864.17 7.28191
6  1 1092 1940.67 7.58073
5  1 1612 2224.65 8.69004    
4  1 1823 2324.23 9.07903
3  1 2291 2410.24 9.41499
2  1 2535 2468.24 9.64155
1  1 2793 2500.04 9.76579
0  1 385 2514.14 9.82084'''

x1 = []
y1 = []
for line in s1.splitlines():
    line_split = line.split()
    x1.append(int(line_split[0]) * 2)
    y1.append(float(line_split[4]))


plt.plot(x0, y0, label='data_0')
plt.plot(x1, y1, label='data_1')
plt.show()