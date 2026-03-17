import matplotlib.pyplot as plt
import math

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

# for before 202503
# PHASE_IDX = 1
# MSE_IDX = 12
# MAE_IDX = 14
# VAL_MSE_IDX = 16
# VAL_MAE_IDX = 18

PHASE_IDX = 1
N_TRAIN_DATA_IDX = 6
N_VAL_DATA_IDX = 8
SCORE_AVG_IDX = 10
N_LOOP_IDX = 12
MSE_IDX = 14
MAE_IDX = 16
VAL_MSE_IDX = 18
VAL_MAE_IDX = 20

TEST_PHASE_IDX = 1
TEST_MSE_IDX = 5
TEST_MAE_IDX = 7
N_LOOP_PLOT_MAX = 5000

phase_arr = []
n_train_data_arr = []
n_val_data_arr = []
score_avg_arr = []
n_loop_arr = []
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
        n_train_data = int(sss[N_TRAIN_DATA_IDX])
        n_val_data = int(sss[N_VAL_DATA_IDX])
        score_avg = float(sss[SCORE_AVG_IDX])
        n_loop = int(sss[N_LOOP_IDX])
        mse = float(sss[MSE_IDX])
        mae = float(sss[MAE_IDX])
        phase_arr.append(phase)
        n_train_data_arr.append(n_train_data)
        n_val_data_arr.append(n_val_data)
        score_avg_arr.append(score_avg)
        n_loop_arr.append(n_loop)
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

plt.clf()
plt.plot(phase_arr, n_train_data_arr, label='n_train_data', color='dodgerblue', marker='s', linewidth=2)
plt.plot(phase_arr, n_val_data_arr, label='n_val_data', color='coral', marker='o', linewidth=2)
plt.xlabel('phase')
plt.ylabel('data count')
plt.xlim(-1, 60)
plt.grid(True)
plt.legend()
plt.savefig('./trained/n_data.png')
print('saved')

plt.clf()
plt.plot(phase_arr, score_avg_arr, label='score_avg', color='slateblue', marker='d', linewidth=2)
plt.xlabel('phase')
plt.ylabel('score_avg')
plt.xlim(-1, 60)
plt.grid(True)
plt.legend()
plt.savefig('./trained/score_avg.png')
print('saved')

plt.clf()
n_loop_wave_y = N_LOOP_PLOT_MAX * 0.985
n_loop_wave_amp = N_LOOP_PLOT_MAX * 0.008
n_loop_over_y = N_LOOP_PLOT_MAX * 1.03
n_loop_ylim_top = N_LOOP_PLOT_MAX * 1.08

n_loop_plot_arr = []
for n_loop in n_loop_arr:
    if n_loop <= N_LOOP_PLOT_MAX:
        n_loop_plot_arr.append(n_loop)
    else:
        n_loop_plot_arr.append(n_loop_over_y)

wave_x_min = -1.0
wave_x_max = 60.0
wave_n = 320
wave_x = [wave_x_min + (wave_x_max - wave_x_min) * i / (wave_n - 1) for i in range(wave_n)]
wave_y = [
    n_loop_wave_y + n_loop_wave_amp * math.sin((x - wave_x_min) * 2.3)
    for x in wave_x
]

plt.plot(phase_arr, n_loop_plot_arr, label='n_loop', color='darkgoldenrod', marker='x', linewidth=2)
plt.plot(wave_x, wave_y, color='black', linewidth=1.5)
plt.xlabel('phase')
plt.ylabel('n_loop')
plt.xlim(-1, 60)
plt.ylim(0, n_loop_ylim_top)
plt.grid(True)
plt.legend()
plt.savefig('./trained/n_loop.png')
print('saved')


