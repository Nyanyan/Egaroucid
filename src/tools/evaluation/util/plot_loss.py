import matplotlib.pyplot as plt

s = '''
phase 0 time 10002 ms data 1 n_loop 52822 MSE 0 MAE 0 (with int) alpha 200
phase 1 time 10002 ms data 1 n_loop 54327 MSE 0 MAE 0 (with int) alpha 200
phase 2 time 10002 ms data 3 n_loop 28151 MSE 6.45564e-11 MAE 7.29164e-06 (with int) alpha 200
phase 3 time 10002 ms data 14 n_loop 26762 MSE 0.00605833 MAE 0.0742466 (with int) alpha 200
phase 4 time 10002 ms data 60 n_loop 25850 MSE 1.10696e-11 MAE 2.92206e-06 (with int) alpha 200
phase 5 time 10003 ms data 322 n_loop 26326 MSE 1.48619 MAE 0.810424 (with int) alpha 200
phase 6 time 10002 ms data 1773 n_loop 21759 MSE 5.3164 MAE 1.71936 (with int) alpha 200
phase 7 time 10003 ms data 10623 n_loop 12002 MSE 11.0842 MAE 2.48211 (with int) alpha 200
phase 8 time 60007 ms data 67153 n_loop 21141 MSE 13.0647 MAE 2.65214 (with int) alpha 200
phase 9 time 60017 ms data 430613 n_loop 6207 MSE 16.6894 MAE 3.01924 (with int) alpha 200
phase 10 time 60075 ms data 2915537 n_loop 1297 MSE 19.5783 MAE 3.10317 (with int) alpha 200

phase 11 time 600392 ms data 20009985 n_loop 1981 MSE 63.5789 MAE 5.85358 (with int) alpha 800
phase 12 time 600449 ms data 20008232 n_loop 2318 MSE 57.2999 MAE 5.58322 (with int) alpha 800
phase 13 time 600345 ms data 20007930 n_loop 2091 MSE 56.2149 MAE 5.54228 (with int) alpha 800
phase 14 time 600486 ms data 20007126 n_loop 2191 MSE 54.5258 MAE 5.46612 (with int) alpha 800
phase 15 time 600440 ms data 20007037 n_loop 2212 MSE 54.2555 MAE 5.46593 (with int) alpha 800
phase 16 time 600424 ms data 20006931 n_loop 2227 MSE 53.2318 MAE 5.41946 (with int) alpha 800
phase 17 time 600462 ms data 20006617 n_loop 2337 MSE 52.7032 MAE 5.41044 (with int) alpha 800
phase 18 time 600389 ms data 20006578 n_loop 2208 MSE 51.338 MAE 5.34595 (with int) alpha 800

phase 19 time 600641 ms data 33710717 n_loop 1335 MSE 52.2118 MAE 5.06445 (with int) alpha 800
phase 20 time 600776 ms data 33710558 n_loop 1385 MSE 48.793 MAE 4.94086 (with int) alpha 800

phase 21 time 300709 ms data 38497871 n_loop 556 MSE 48.1555 MAE 4.82081 (with int) alpha 700
phase 22 time 300686 ms data 38497672 n_loop 610 MSE 44.7042 MAE 4.65679 (with int) alpha 700
phase 23 time 300990 ms data 38497431 n_loop 549 MSE 43.3414 MAE 4.59821 (with int) alpha 700
phase 24 time 300967 ms data 38497356 n_loop 553 MSE 41.8011 MAE 4.52915 (with int) alpha 700
phase 25 time 300968 ms data 38497206 n_loop 595 MSE 40.8518 MAE 4.48902 (with int) alpha 700
phase 26 time 300902 ms data 38497153 n_loop 598 MSE 39.5977 MAE 4.43312 (with int) alpha 700
phase 27 time 300662 ms data 38497034 n_loop 543 MSE 38.7278 MAE 4.39546 (with int) alpha 700
phase 28 time 300817 ms data 38496934 n_loop 553 MSE 37.5781 MAE 4.34361 (with int) alpha 700
phase 29 time 300520 ms data 38496732 n_loop 634 MSE 36.7116 MAE 4.30609 (with int) alpha 700
phase 30 time 300633 ms data 43254138 n_loop 575 MSE 38.0475 MAE 4.32638 (with int) alpha 700
phase 31 time 300988 ms data 43253752 n_loop 590 MSE 35.7607 MAE 4.21314 (with int) alpha 700
phase 32 time 301032 ms data 43253161 n_loop 585 MSE 34.2759 MAE 4.14528 (with int) alpha 700
phase 33 time 301001 ms data 43252362 n_loop 560 MSE 33.1094 MAE 4.09322 (with int) alpha 700
phase 34 time 300954 ms data 43251295 n_loop 561 MSE 31.7559 MAE 4.03693 (with int) alpha 700
phase 35 time 301007 ms data 43249598 n_loop 567 MSE 29.8567 MAE 3.93449 (with int) alpha 700
phase 36 time 300794 ms data 43247535 n_loop 567 MSE 29.2181 MAE 3.88477 (with int) alpha 700
phase 37 time 300724 ms data 43244973 n_loop 567 MSE 29.1959 MAE 3.88789 (with int) alpha 700
phase 38 time 300949 ms data 43242146 n_loop 558 MSE 28.9657 MAE 3.88537 (with int) alpha 700
phase 39 time 300599 ms data 43238662 n_loop 553 MSE 28.8807 MAE 3.89495 (with int) alpha 700
phase 40 time 300777 ms data 43232878 n_loop 546 MSE 29.0375 MAE 3.92098 (with int) alpha 700
phase 41 time 300871 ms data 43226687 n_loop 511 MSE 29.4791 MAE 3.9554 (with int) alpha 700
phase 42 time 300954 ms data 43220020 n_loop 509 MSE 29.8972 MAE 3.99101 (with int) alpha 700
phase 43 time 300908 ms data 43212227 n_loop 500 MSE 30.4945 MAE 4.01365 (with int) alpha 700
phase 44 time 301180 ms data 43203427 n_loop 496 MSE 31.0431 MAE 4.0322 (with int) alpha 700
phase 45 time 301088 ms data 43194315 n_loop 493 MSE 31.588 MAE 4.04574 (with int) alpha 700
phase 46 time 301393 ms data 43184004 n_loop 431 MSE 32.221 MAE 4.06629 (with int) alpha 700
phase 47 time 300674 ms data 43173401 n_loop 501 MSE 32.6546 MAE 4.08507 (with int) alpha 700
phase 48 time 300941 ms data 43161554 n_loop 466 MSE 32.5382 MAE 4.07839 (with int) alpha 700
phase 49 time 301118 ms data 43148217 n_loop 467 MSE 32.3529 MAE 4.06854 (with int) alpha 700
phase 50 time 300869 ms data 43132152 n_loop 483 MSE 31.5918 MAE 4.03885 (with int) alpha 700
phase 51 time 301129 ms data 43114888 n_loop 485 MSE 30.9852 MAE 4.01626 (with int) alpha 700
phase 52 time 300990 ms data 43095248 n_loop 437 MSE 29.4007 MAE 3.91208 (with int) alpha 700
phase 53 time 300667 ms data 43069790 n_loop 491 MSE 27.9464 MAE 3.7881 (with int) alpha 700
phase 54 time 301225 ms data 43034578 n_loop 490 MSE 25.5655 MAE 3.58864 (with int) alpha 700
phase 55 time 300696 ms data 42986887 n_loop 515 MSE 22.805 MAE 3.34023 (with int) alpha 700
phase 56 time 300793 ms data 42914134 n_loop 533 MSE 18.4154 MAE 2.9072 (with int) alpha 700
phase 57 time 300740 ms data 42800609 n_loop 561 MSE 13.9323 MAE 2.46604 (with int) alpha 700

phase 58 time 600635 ms data 33242114 n_loop 1676 MSE 7.56473 MAE 1.85022 (with int) alpha 800
phase 59 time 600316 ms data 32549899 n_loop 2141 MSE 1.54371e-06 MAE 0.000328547 (with int) alpha 800
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
ln1=ax1.plot(phase_arr, mae_arr, 'C0', label='MAE')
ax2 = ax1.twinx()
ln2=ax2.plot(phase_arr, mse_arr, 'C1', label='MSE')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper right')

ax1.set_xlabel('phase')
ax1.set_ylabel('MAE')
ax2.set_ylabel('MSE')
ax1.grid(True)

plt.show()
