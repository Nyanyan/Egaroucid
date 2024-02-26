import matplotlib.pyplot as plt

s = '''
phase 0 time 10002 ms data 1 n_loop 50018 MSE 0 MAE 0 (with int) alpha 100
phase 1 time 10002 ms data 1 n_loop 49565 MSE 0 MAE 0 (with int) alpha 100
phase 2 time 3029 ms data 3 n_loop 1691 MSE 1.82449e-12 MAE 9.61125e-07 (with int) alpha 1
phase 3 time 5002 ms data 14 n_loop 4925 MSE 9.58856e-15 MAE 3.32615e-08 (with int) alpha 1
phase 4 time 5012 ms data 60 n_loop 4491 MSE 4.64767e-05 MAE 0.00281484 (with int) alpha 1
phase 5 time 20008 ms data 322 n_loop 13472 MSE 1.49798 MAE 0.804629 (with int) alpha 5
phase 6 time 40003 ms data 1773 n_loop 43560 MSE 5.31641 MAE 1.71984 (with int) alpha 7
phase 7 time 60003 ms data 10623 n_loop 46049 MSE 11.0842 MAE 2.48181 (with int) alpha 7
phase 8 time 30010 ms data 67153 n_loop 7251 MSE 13.0721 MAE 2.65313 (with int) alpha 10
phase 9 time 50019 ms data 430613 n_loop 4814 MSE 16.6862 MAE 3.01143 (with int) alpha 20
phase 10 time 60102 ms data 2915537 n_loop 1155 MSE 19.5998 MAE 3.10401 (with int) alpha 60

phase 11 time 300344 ms data 20009985 n_loop 1020 MSE 63.5916 MAE 5.8545 (with int) alpha 300
phase 12 time 300325 ms data 20008232 n_loop 1112 MSE 57.3268 MAE 5.58479 (with int) alpha 300
phase 13 time 300345 ms data 20007930 n_loop 1078 MSE 56.2478 MAE 5.54393 (with int) alpha 300
phase 14 time 300316 ms data 20007126 n_loop 1045 MSE 54.5567 MAE 5.4686 (with int) alpha 300
phase 15 time 300497 ms data 20007037 n_loop 1009 MSE 54.2939 MAE 5.46873 (with int) alpha 300
phase 16 time 300403 ms data 20006931 n_loop 1101 MSE 53.2707 MAE 5.42178 (with int) alpha 300
phase 17 time 300336 ms data 20006617 n_loop 1189 MSE 52.7464 MAE 5.41275 (with int) alpha 300
phase 18 time 300278 ms data 20006578 n_loop 1180 MSE 51.377 MAE 5.34828 (with int) alpha 300

phase 19 time 300621 ms data 33710717 n_loop 682 MSE 52.2452 MAE 5.06652 (with int) alpha 500
phase 20 time 300619 ms data 33710558 n_loop 676 MSE 48.8278 MAE 4.94286 (with int) alpha 500

phase 21 time 300777 ms data 38497871 n_loop 591 MSE 48.2026 MAE 4.82303 (with int) alpha 500
phase 22 time 300870 ms data 38497672 n_loop 541 MSE 44.7556 MAE 4.65934 (with int) alpha 500
phase 23 time 300798 ms data 38497431 n_loop 532 MSE 43.3827 MAE 4.60047 (with int) alpha 500
phase 24 time 300783 ms data 38497356 n_loop 611 MSE 41.8386 MAE 4.53057 (with int) alpha 500
phase 25 time 300673 ms data 38497206 n_loop 621 MSE 40.8788 MAE 4.49037 (with int) alpha 500
phase 26 time 300493 ms data 38497153 n_loop 623 MSE 39.628 MAE 4.43428 (with int) alpha 500
phase 27 time 300817 ms data 38497034 n_loop 639 MSE 38.7503 MAE 4.39649 (with int) alpha 500
phase 28 time 300963 ms data 38496934 n_loop 621 MSE 37.6027 MAE 4.34471 (with int) alpha 500
phase 29 time 300968 ms data 38496732 n_loop 623 MSE 36.7293 MAE 4.3068 (with int) alpha 500

phase 30 time 300961 ms data 43254138 n_loop 523 MSE 38.0588 MAE 4.32704 (with int) alpha 500
phase 31 time 300928 ms data 43253752 n_loop 520 MSE 35.7722 MAE 4.21357 (with int) alpha 500
phase 32 time 300612 ms data 43253161 n_loop 523 MSE 34.2847 MAE 4.14569 (with int) alpha 500
phase 33 time 301035 ms data 43252362 n_loop 529 MSE 33.1135 MAE 4.09331 (with int) alpha 500
phase 34 time 300882 ms data 43251295 n_loop 560 MSE 31.7611 MAE 4.03709 (with int) alpha 500
phase 35 time 301148 ms data 43249598 n_loop 498 MSE 29.8584 MAE 3.9344 (with int) alpha 500
phase 36 time 301046 ms data 43247535 n_loop 576 MSE 29.2208 MAE 3.88505 (with int) alpha 500
phase 37 time 301046 ms data 43244973 n_loop 566 MSE 29.1958 MAE 3.88787 (with int) alpha 500
phase 38 time 300968 ms data 43242146 n_loop 542 MSE 28.9652 MAE 3.88557 (with int) alpha 500
phase 39 time 300570 ms data 43238662 n_loop 539 MSE 28.8805 MAE 3.89498 (with int) alpha 500
phase 40 time 301182 ms data 43232878 n_loop 502 MSE 29.0409 MAE 3.92118 (with int) alpha 500
phase 41 time 300603 ms data 43226687 n_loop 536 MSE 29.4826 MAE 3.95567 (with int) alpha 500
phase 42 time 301029 ms data 43220020 n_loop 466 MSE 29.9002 MAE 3.99144 (with int) alpha 500
phase 43 time 300962 ms data 43212227 n_loop 529 MSE 30.4994 MAE 4.0138 (with int) alpha 500
phase 44 time 301235 ms data 43203427 n_loop 460 MSE 31.0474 MAE 4.03238 (with int) alpha 500
phase 45 time 301118 ms data 43194315 n_loop 521 MSE 31.5949 MAE 4.04611 (with int) alpha 500
phase 46 time 300855 ms data 43184004 n_loop 455 MSE 32.2253 MAE 4.06653 (with int) alpha 500
phase 47 time 301090 ms data 43173401 n_loop 504 MSE 32.6596 MAE 4.0852 (with int) alpha 500
phase 48 time 301292 ms data 43161554 n_loop 450 MSE 32.5376 MAE 4.07894 (with int) alpha 500
phase 49 time 301175 ms data 43148217 n_loop 448 MSE 32.3565 MAE 4.06865 (with int) alpha 500
phase 50 time 301057 ms data 43132152 n_loop 497 MSE 31.597 MAE 4.03902 (with int) alpha 500
phase 51 time 301127 ms data 43114888 n_loop 514 MSE 30.9887 MAE 4.01637 (with int) alpha 500
phase 52 time 300931 ms data 43095248 n_loop 503 MSE 29.4019 MAE 3.91233 (with int) alpha 500
phase 53 time 301193 ms data 43069790 n_loop 480 MSE 27.9476 MAE 3.78812 (with int) alpha 500
phase 54 time 300743 ms data 43034578 n_loop 529 MSE 25.5662 MAE 3.58871 (with int) alpha 500
phase 55 time 300724 ms data 42986887 n_loop 545 MSE 22.8068 MAE 3.34003 (with int) alpha 500
phase 56 time 300805 ms data 42914134 n_loop 557 MSE 18.4154 MAE 2.90719 (with int) alpha 500
phase 57 time 300932 ms data 42800609 n_loop 580 MSE 13.9348 MAE 2.46603 (with int) alpha 500
phase 58 time 300882 ms data 42594281 n_loop 634 MSE 7.39802 MAE 1.75771 (with int) alpha 500
phase 59 time 300628 ms data 41632395 n_loop 805 MSE 0.00132541 MAE 0.01798 (with int) alpha 100

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
