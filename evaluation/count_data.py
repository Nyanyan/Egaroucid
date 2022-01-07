from tqdm import tqdm
import glob

files = glob.glob('third_party/records3/*')
ans = 0
for file in tqdm(files):
    with open(file, 'r') as f:
        ans += len(f.read().splitlines())
print(ans)