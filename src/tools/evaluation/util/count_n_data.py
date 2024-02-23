s = '''
phase 0 tl 10 data 1 n 45681 mse 0 mae 0 beta 0.5 alr 0.000317479
phase 1 tl 10 data 1 n 45980 mse 0 mae 0 beta 0.5 alr 0.000317479
phase 2 tl 10 data 3 n 42305 mse 0.00476074 mae 0.0677083 beta 0.5 alr 1
phase 3 tl 30 data 14 n 128053 mse 0.00459943 mae 0.0641741 beta 0.5 alr 1
phase 4 tl 60 data 60 n 225649 mse 0.00401815 mae 0.0440104 beta 0.5 alr 1
phase 5 tl 60 data 322 n 159633 mse 1.91536 mae 0.970934 beta 0.5 alr 7.53393e-05
phase 6 tl 60 data 1773 n 89009 mse 5.90277 mae 1.86468 beta 0.25 alr 7.53393e-05
phase 7 tl 60 data 10623 n 67953 mse 12.3004 mae 2.6605 beta 0.25 alr 7.53393e-05
phase 8 tl 60 data 67153 n 13137 mse 15.5379 mae 2.88002 beta 0.1 alr 7.53393e-05
phase 9 tl 120 data 430613 n 7400 mse 19.1481 mae 3.23547 beta 0.1 alr 0.00178381
phase 10 tl 120 data 2915537 n 882 mse 22.087 mae 3.31272 beta 0.1 alr 1

phase 11 tl 400 data 20009985 n 362 mse 68.4878 mae 6.17461 beta 0.25 alr 1
phase 12 tl 400 data 20008232 n 366 mse 62.0303 mae 5.89571 beta 0.25 alr 1
phase 13 tl 400 data 20007930 n 289 mse 60.9492 mae 5.85607 beta 0.25 alr 1
phase 14 tl 400 data 20007126 n 403 mse 59.1553 mae 5.77827 beta 0.25 alr 1
phase 15 tl 400 data 20007037 n 410 mse 58.8766 mae 5.77701 beta 0.25 alr 1
phase 16 tl 400 data 20006931 n 391 mse 57.9776 mae 5.73861 beta 0.25 alr 1
phase 17 tl 400 data 20006617 n 403 mse 57.2667 mae 5.71728 beta 0.25 alr 1
phase 18 tl 400 data 20006578 n 412 mse 56.0968 mae 5.66773 beta 0.25 alr 0.75

phase 19 tl 400 data 33710717 n 235 mse 60.1712 mae 5.8233 beta 0.25 alr 1
phase 20 tl 400 data 33710558 n 234 mse 56.2772 mae 5.67388 beta 0.25 alr 1
phase 21 tl 400 data 33710184 n 195 mse 54.5179 mae 5.59264 beta 0.25 alr 1
phase 22 tl 400 data 33710090 n 238 mse 51.5824 mae 5.44277 beta 0.25 alr 1
phase 23 tl 400 data 33709865 n 241 mse 50.4127 mae 5.38984 beta 0.25 alr 1
phase 24 tl 400 data 33709809 n 246 mse 48.4767 mae 5.28811 beta 0.25 alr 1
phase 25 tl 400 data 33709669 n 215 mse 47.7012 mae 5.25373 beta 0.25 alr 1
phase 26 tl 400 data 33709622 n 256 mse 45.975 mae 5.16175 beta 0.25 alr 1
phase 27 tl 400 data 33709521 n 266 mse 44.8003 mae 5.10065 beta 0.25 alr 1
phase 28 tl 400 data 33709427 n 235 mse 43.4564 mae 5.02588 beta 0.25 alr 1
phase 29 tl 400 data 33709239 n 267 mse 42.5359 mae 4.97978 beta 0.25 alr 1
phase 30 tl 400 data 33709047 n 267 mse 40.7396 mae 4.87336 beta 0.25 alr 1
phase 31 tl 400 data 33708702 n 262 mse 39.7506 mae 4.81867 beta 0.25 alr 1
phase 32 tl 400 data 33708123 n 269 mse 38.2342 mae 4.72625 beta 0.25 alr 1
phase 33 tl 400 data 33707348 n 263 mse 37.1418 mae 4.66195 beta 0.25 alr 1
phase 34 tl 400 data 33706290 n 288 mse 35.4414 mae 4.55336 beta 0.25 alr 1
phase 35 tl 400 data 33704633 n 286 mse 33.5747 mae 4.42934 beta 0.25 alr 1
phase 36 tl 400 data 33702612 n 277 mse 32.699 mae 4.3663 beta 0.25 alr 1
phase 37 tl 400 data 33700091 n 282 mse 32.7056 mae 4.36974 beta 0.25 alr 1
phase 38 tl 400 data 33697316 n 270 mse 32.4786 mae 4.36385 beta 0.25 alr 1
phase 39 tl 400 data 33693901 n 266 mse 32.4002 mae 4.37244 beta 0.25 alr 1
phase 40 tl 400 data 33688257 n 274 mse 32.5734 mae 4.39501 beta 0.25 alr 1
phase 41 tl 400 data 33682313 n 253 mse 33.1175 mae 4.43333 beta 0.25 alr 1
phase 42 tl 400 data 33675966 n 270 mse 33.6493 mae 4.47171 beta 0.25 alr 1
phase 43 tl 400 data 33668676 n 254 mse 34.3246 mae 4.51444 beta 0.25 alr 1
phase 44 tl 400 data 33660622 n 253 mse 34.9371 mae 4.55541 beta 0.25 alr 1
phase 45 tl 400 data 33652408 n 243 mse 35.529 mae 4.58732 beta 0.25 alr 1
phase 46 tl 400 data 33643309 n 251 mse 36.2983 mae 4.63371 beta 0.25 alr 1
phase 47 tl 400 data 33634247 n 241 mse 36.8515 mae 4.66822 beta 0.25 alr 1
phase 48 tl 400 data 33624260 n 249 mse 36.6555 mae 4.65372 beta 0.25 alr 1
phase 49 tl 400 data 33613403 n 237 mse 36.5031 mae 4.63989 beta 0.25 alr 1
phase 50 tl 400 data 33600874 n 241 mse 35.7197 mae 4.58757 beta 0.25 alr 1
phase 51 tl 400 data 33587908 n 239 mse 34.9241 mae 4.5312 beta 0.25 alr 1
phase 52 tl 400 data 33573297 n 243 mse 33.5424 mae 4.43627 beta 0.25 alr 1
phase 53 tl 400 data 33554882 n 242 mse 31.9104 mae 4.31761 beta 0.25 alr 1
phase 54 tl 400 data 33529768 n 245 mse 29.6183 mae 4.15213 beta 0.25 alr 1
phase 55 tl 400 data 33497013 n 236 mse 26.7281 mae 3.93192 beta 0.25 alr 1
phase 56 tl 400 data 33448458 n 236 mse 21.9142 mae 3.53415 beta 0.25 alr 1
phase 57 tl 400 data 33375060 n 234 mse 16.8224 mae 3.06568 beta 0.25 alr 1
phase 58 tl 400 data 33242114 n 238 mse 9.73061 mae 2.27288 beta 0.25 alr 1
phase 59 tl 400 data 32549899 n 252 mse 1.30766 mae 0.857198 beta 0.25 alr 0.5625

'''

s = s.splitlines()

n_data = 0
n_count_phases = 0

for line in s:
    try:
        n = int(line.split()[5])
        n_data += n
        n_count_phases += 1
    except:
        continue

print(n_data, 'data used for optimization of', n_count_phases, 'phases')