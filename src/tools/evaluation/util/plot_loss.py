import matplotlib.pyplot as plt

s = '''
phase 0 time 1003 ms data 2 n_loop 6400 MSE 0 MAE 0 (with int) alpha 80
phase 1 time 2003 ms data 17 n_loop 6320 MSE 2.10802e-08 MAE 0.000141664 (with int) alpha 500
phase 2 time 10003 ms data 382 n_loop 23927 MSE 0.0123221 MAE 0.0570949 (with int) alpha 10
phase 3 time 10005 ms data 12396 n_loop 9164 MSE 5.97299 MAE 1.80079 (with int) alpha 10
phase 4 time 30018 ms data 497766 n_loop 2670 MSE 13.0621 MAE 2.61378 (with int) alpha 50
phase 5 time 120470 ms data 22925522 n_loop 321 MSE 53.4699 MAE 5.11743 (with int) alpha 300
phase 6 time 180868 ms data 40016162 n_loop 259 MSE 50.8293 MAE 4.81473 (with int) alpha 500
phase 7 time 180721 ms data 40014163 n_loop 298 MSE 48.6191 MAE 4.73621 (with int) alpha 500
phase 8 time 181070 ms data 40013548 n_loop 264 MSE 47.1835 MAE 4.68887 (with int) alpha 500
phase 9 time 301458 ms data 52369441 n_loop 326 MSE 43.9737 MAE 4.31342 (with int) alpha 500
phase 10 time 601781 ms data 71073525 n_loop 482 MSE 40.2548 MAE 4.01197 (with int) alpha 500
phase 11 time 601784 ms data 75860208 n_loop 460 MSE 36.4857 MAE 3.64723 (with int) alpha 500
phase 12 time 602379 ms data 75859677 n_loop 444 MSE 34.379 MAE 3.49053 (with int) alpha 500
phase 13 time 601802 ms data 75859309 n_loop 392 MSE 32.7223 MAE 3.35741 (with int) alpha 500
phase 14 time 601767 ms data 75858790 n_loop 458 MSE 30.8756 MAE 3.22762 (with int) alpha 500
phase 15 time 1205161 ms data 85373029 n_loop 272 MSE 30.2567 MAE 3.01806 (with int) alpha 500
phase 16 time 1204571 ms data 85370751 n_loop 267 MSE 26.8007 MAE 2.81563 (with int) alpha 500
phase 17 time 1206257 ms data 85366241 n_loop 265 MSE 23.9563 MAE 2.64769 (with int) alpha 500
phase 18 time 1207590 ms data 85358027 n_loop 266 MSE 22.87 MAE 2.58643 (with int) alpha 500
phase 19 time 1207757 ms data 85346577 n_loop 263 MSE 22.8055 MAE 2.59113 (with int) alpha 500
phase 20 time 1207873 ms data 85325646 n_loop 252 MSE 23.3023 MAE 2.62912 (with int) alpha 500
phase 21 time 1207314 ms data 85298720 n_loop 259 MSE 24.357 MAE 2.69844 (with int) alpha 500
phase 22 time 1205244 ms data 85264647 n_loop 250 MSE 25.5923 MAE 2.77862 (with int) alpha 500
phase 23 time 1207589 ms data 85224657 n_loop 247 MSE 26.9763 MAE 2.86771 (with int) alpha 500
phase 24 time 1210118 ms data 85177405 n_loop 229 MSE 27.2284 MAE 2.88707 (with int) alpha 500
phase 25 time 1208144 ms data 85115085 n_loop 222 MSE 26.3644 MAE 2.83468 (with int) alpha 500
phase 26 time 1207462 ms data 85033489 n_loop 227 MSE 24.3644 MAE 2.71084 (with int) alpha 500
phase 27 time 1206101 ms data 84890853 n_loop 237 MSE 20.8526 MAE 2.49108 (with int) alpha 500
phase 28 time 1207216 ms data 84585957 n_loop 244 MSE 14.4012 MAE 2.1241 (with int) alpha 500
phase 29 time 304942 ms data 83111168 n_loop 69 MSE 4.44414 MAE 1.09993 (with int) alpha 400

'''


s = s.splitlines()

PHASE_IDX = 1
MSE_IDX = 10
MAE_IDX = 12

phase_arr = []
mse_arr = []
mae_arr = []

for ss in s:
    try:
        sss = ss.split()
        phase = int(sss[PHASE_IDX])
        mse = float(sss[MSE_IDX])
        mae = float(sss[MAE_IDX])
        phase_arr.append(phase)
        mse_arr.append(mse)
        mae_arr.append(mae)
    except:
        pass

fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1=ax1.plot(phase_arr, mae_arr, 'C0', marker='o', label='MAE')
ax2 = ax1.twinx()
ln2=ax2.plot(phase_arr, mse_arr, 'C1', marker='o', label='MSE')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper right')

ax1.set_xlabel('phase')
ax1.set_ylabel('MAE')
ax2.set_ylabel('MSE')
ax1.grid(True)

if input('save?: ') == 'y':
    fig.savefig('./../trained/loss.png')
plt.show()
