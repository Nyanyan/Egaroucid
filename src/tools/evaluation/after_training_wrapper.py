import subprocess

subprocess.run('output_egev.out 60 3275')
subprocess.run('output_egev2.out 60 3275')
subprocess.run('python test_loss_wrapper.py')
subprocess.run('python plot_loss.py')