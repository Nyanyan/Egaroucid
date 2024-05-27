import matplotlib.pyplot as plt

opt_log_file = 'trained/opt_log.txt'
test_file = 'trained/test.txt'

with open(opt_log_file, 'r') as f:
    s = f.read()
s = s.splitlines()

with open(test_file, 'r') as f:
    t = f.read()
t = t.splitlines()

PHASE_IDX = 1
MSE_IDX = 12
MAE_IDX = 14
VAL_MSE_IDX = 16
VAL_MAE_IDX = 18

TEST_PHASE_IDX = 1
TEST_MSE_IDX = 5
TEST_MAE_IDX = 7

phase_arr = []
mse_arr = []
mae_arr = []
val_phase_arr = []
val_mse_arr = []
val_mae_arr = []
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
        val_mse = float(sss[VAL_MSE_IDX])
        val_mae = float(sss[VAL_MAE_IDX])
        val_phase_arr.append(phase)
        val_mse_arr.append(val_mse)
        val_mae_arr.append(val_mae)
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

plt.plot(phase_arr, mae_arr, color='red', marker='o', label='train_MAE', linestyle="dotted", linewidth=5)
plt.plot(val_phase_arr, val_mae_arr, color='lightseagreen', marker='o', label='val_MAE')
plt.plot(test_phase_arr, test_mae_arr, color='blue', marker='o', label='test_MAE')
plt.xlabel('phase')
plt.ylabel('MAE')
plt.xlim(-1, 60)
plt.ylim(-0.5, 5)
plt.grid(True)
plt.legend()
#plt.show()
#if input('save?: ') == 'y':
plt.savefig('./trained/loss_mae.png')
print('saved')

plt.clf()
plt.plot(phase_arr, mse_arr, color='orange', marker='o', label='train_MSE', linestyle="dotted", linewidth=5)
plt.plot(val_phase_arr, val_mse_arr, color='limegreen', marker='o', label='val_MSE')
plt.plot(test_phase_arr, test_mse_arr, color='purple', marker='o', label='test_MSE')
plt.xlabel('phase')
plt.ylabel('MSE')
plt.xlim(-1, 60)
plt.ylim(-5, 50)
plt.grid(True)
plt.legend()
#plt.show()
#if input('save?: ') == 'y':
plt.savefig('./trained/loss_mse.png')
print('saved')
