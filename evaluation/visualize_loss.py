import matplotlib.pyplot as plt

s0 = '''59	733 mae 0.00148207 mse 0.00747858 n_data 1165938 n_param 528065 n_used_params 37671
58	670 mae 1.72615 mse 6.06521 n_data 1169517 n_param 528065 n_used_params 88370
57	653 mae 2.46903 mse 11.2146 n_data 1173224 n_param 528065 n_used_params 140766
56	654 mae 2.97058 mse 15.7668 n_data 1176515 n_param 528065 n_used_params 183294
55	636 mae 3.34599 mse 19.6235 n_data 1179408 n_param 528065 n_used_params 213017
54	634 mae 3.59767 mse 22.5228 n_data 1181643 n_param 528065 n_used_params 231032
53	651 mae 3.80313 mse 24.9832 n_data 1183383 n_param 528065 n_used_params 241441
52	650 mae 3.9298 mse 26.5923 n_data 1184759 n_param 528065 n_used_params 247185
51	615 mae 4.02845 mse 27.8628 n_data 1185784 n_param 528065 n_used_params 250782
50	525 mae 4.19917 mse 30.0624 n_data 1286683 n_param 528065 n_used_params 287935
49	524 mae 4.23916 mse 30.6137 n_data 1287980 n_param 528065 n_used_params 289377
48	514 mae 4.26203 mse 30.9882 n_data 1289061 n_param 528065 n_used_params 290352
47	491 mae 4.26701 mse 30.9948 n_data 1290304 n_param 528065 n_used_params 290265
46	508 mae 4.25308 mse 30.8069 n_data 1291275 n_param 528065 n_used_params 290330
45	550 mae 4.15662 mse 29.455 n_data 1227594 n_param 528065 n_used_params 271288
44	562 mae 4.1044 mse 28.7446 n_data 1193611 n_param 528065 n_used_params 246921
43	561 mae 4.07883 mse 28.4155 n_data 1195093 n_param 528065 n_used_params 245026'''

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