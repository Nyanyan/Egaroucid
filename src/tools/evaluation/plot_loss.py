import matplotlib.pyplot as plt

opt_log_file = 'trained/opt_log.txt'

with open(opt_log_file, 'r') as f:
    s = f.read()

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

ax1.set_ylim(-0.5, 7)
ax2.set_ylim(-5, 70)

ax1.set_xlabel('phase')
ax1.set_ylabel('MAE')
ax2.set_ylabel('MSE')
ax1.grid(True)

plt.show()

if input('save?: ') == 'y':
    fig.savefig('./trained/loss.png')
    print('saved')
