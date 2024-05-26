import matplotlib.pyplot as plt

'''
opt_log_file = 'trained/opt_log.txt'
test_file = 'trained/loss.txt'

with open(opt_log_file, 'r') as f:
    s = f.read()
s = s.splitlines()

with open(test_file, 'r') as f:
    t = f.read()
t = t.splitlines()

PHASE_IDX = 1
MSE_IDX = 10
MAE_IDX = 12

TEST_PHASE_IDX = 1
TEST_MSE_IDX = 5
TEST_MAE_IDX = 7

phase_arr = []
mse_arr = []
mae_arr = []
test_phase_arr = []
test_mse_arr = []
test_mae_arr = []

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

for tt in t:
    try:
        ttt = tt.split()
        phase = int(ttt[TEST_PHASE_IDX])
        mse = float(ttt[TEST_MSE_IDX])
        mae = float(ttt[TEST_MAE_IDX])
        test_phase_arr.append(phase)
        test_mse_arr.append(mse)
        test_mae_arr.append(mae)
    except:
        pass
'''

opt_log_file = 'trained/opt_log.txt'

with open(opt_log_file, 'r') as f:
    s = f.read()
s = s.splitlines()

PHASE_IDX = 1
MSE_IDX = 12
MAE_IDX = 14
TEST_MSE_IDX = 16
TEST_MAE_IDX = 18

phase_arr = []
mse_arr = []
mae_arr = []
test_phase_arr = []
test_mse_arr = []
test_mae_arr = []

for ss in s:
    try:
        sss = ss.split()
        phase = int(sss[PHASE_IDX])
        mse = float(sss[MSE_IDX])
        mae = float(sss[MAE_IDX])
        phase_arr.append(phase)
        mse_arr.append(mse)
        mae_arr.append(mae)
        test_mse = float(sss[TEST_MSE_IDX])
        test_mae = float(sss[TEST_MAE_IDX])
        test_phase_arr.append(phase)
        test_mse_arr.append(test_mse)
        test_mae_arr.append(test_mae)
    except:
        pass

'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1=ax1.plot(phase_arr, mae_arr, 'C0', marker='o', label='train_MAE')
ln1=ax1.plot(test_phase_arr, test_mae_arr, 'C2', marker='o', label='test_MAE')
ax2 = ax1.twinx()
ln2=ax2.plot(phase_arr, mse_arr, 'C1', marker='o', label='train_MSE')
ln2=ax2.plot(test_phase_arr, test_mse_arr, 'C3', marker='o', label='test_MSE')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper right')

ax1.set_ylim(-0.5, 9)
ax2.set_ylim(-5, 90)

ax1.set_xlabel('phase')
ax1.set_ylabel('MAE')
ax2.set_ylabel('MSE')
ax1.grid(True)
'''

plt.plot(phase_arr, mae_arr, 'C0', marker='o', label='train_MAE')
plt.plot(test_phase_arr, test_mae_arr, 'C2', marker='o', label='test_MAE')
plt.xlabel('phase')
plt.ylabel('MAE')
plt.ylim(-0.5, 5)
plt.grid(True)
plt.legend()
#plt.show()
if input('save?: ') == 'y':
    plt.savefig('./trained/loss_mae.png')
    print('saved')

plt.clf()
plt.plot(phase_arr, mse_arr, 'C1', marker='o', label='train_MSE')
plt.plot(test_phase_arr, test_mse_arr, 'C3', marker='o', label='test_MSE')
plt.xlabel('phase')
plt.ylabel('MSE')
plt.ylim(-5, 50)
plt.grid(True)
plt.legend()
#plt.show()
if input('save?: ') == 'y':
    plt.savefig('./trained/loss_mse.png')
    print('saved')
