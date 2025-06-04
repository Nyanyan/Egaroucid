import matplotlib.pyplot as plt

opt_log_file = 'trained/opt_log.txt'
test_files = ['trained/test_random.txt', 'trained/test_drawline.txt']
test_labels = ['test_random', 'test_drawline']
test_markers = ['^', '1']
test_colors = [
    ['blue', 'green'],
    ['purple', 'hotpink'],
]


with open(opt_log_file, 'r') as f:
    s = f.read()
s = s.splitlines()

tests = []

for test_file in test_files:
    with open(test_file, 'r') as f:
        tests.append(f.read().splitlines())

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
tests_phase_arr = []
tests_mse_arr = []
tests_mae_arr = []

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

for t in tests:
    tests_phase_arr.append([])
    tests_mse_arr.append([])
    tests_mae_arr.append([])
    for tt in t:
        try:
            ttt = tt.split()
            phase = int(ttt[TEST_PHASE_IDX])
            mse = float(ttt[TEST_MSE_IDX])
            mae = float(ttt[TEST_MAE_IDX])
            tests_phase_arr[-1].append(phase)
            tests_mse_arr[-1].append(mse)
            tests_mae_arr[-1].append(mae)
        except:
            pass

plt.plot(phase_arr, mae_arr, label='train_MAE', color='red', marker='s', linewidth=4)
plt.plot(val_phase_arr, val_mae_arr, label='val_MAE', color='lightseagreen', marker='o', linewidth=2)
for i in range(len(tests_phase_arr)):
    plt.plot(tests_phase_arr[i], tests_mae_arr[i], label=test_labels[i], color=test_colors[0][i], marker=test_markers[i], linestyle="dashed")
plt.xlabel('phase')
plt.ylabel('MAE')
plt.xlim(-1, 60)
plt.ylim(-0.5, 7)
plt.grid(True)
plt.legend()
#plt.show()
#if input('save?: ') == 'y':
plt.savefig('./trained/loss_mae.png')
print('saved')

plt.clf()
plt.plot(phase_arr, mse_arr, label='train_MSE', color='orange', marker='s', linewidth=4)
plt.plot(val_phase_arr, val_mse_arr, label='val_MSE', color='limegreen', marker='o', linewidth=2)
for i in range(len(tests_phase_arr)):
    plt.plot(tests_phase_arr[i], tests_mse_arr[i], label=test_labels[i], color=test_colors[1][i], marker=test_markers[i], linestyle="dashed")
plt.xlabel('phase')
plt.ylabel('MSE')
plt.xlim(-1, 60)
plt.ylim(-5, 90)
plt.grid(True)
plt.legend()
#plt.show()
#if input('save?: ') == 'y':
plt.savefig('./trained/loss_mse.png')
print('saved')
