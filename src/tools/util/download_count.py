import requests
from datetime import datetime, timezone
import matplotlib.pyplot as plt

r = requests.get('https://api.github.com/repos/Nyanyan/Egaroucid/releases')

# GUI, Console
name_arr = [[], []]
published_arr = [[], []]
subtotal_arr = [[], []]

GUI_IDX = 0
CONSOLE_IDX = 1
OTHERS_IDX = 2
labels = ['GUI', 'Console', 'Others']

sum_download_counts = [0, 0, 0]
for item in reversed(r.json()):
    print("tag_name: ", item["tag_name"])
    print("name: ", item["name"])
    print("published: ", item["published_at"])
    subtotal = 0
    for i in range(len(item["assets"])):
        print("download count:", item["assets"][i]["download_count"], " name:", item["assets"][i]["name"])
        subtotal += int(item["assets"][i]["download_count"])
    print('subtotal:', subtotal)
    print("")
    dt = datetime.strptime(item["published_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    item_idx = -1
    if item["tag_name"][:9] == 'console_v':
        item_idx = CONSOLE_IDX
    elif item["tag_name"][0] == 'v':
        item_idx = GUI_IDX
    else:
        item_idx = OTHERS_IDX
    if item_idx != OTHERS_IDX:
        name_arr[item_idx].append(item["name"])
        published_arr[item_idx].append(int(dt.timestamp()))
        subtotal_arr[item_idx].append(subtotal)
    sum_download_counts[item_idx] += subtotal

for i in range(len(sum_download_counts)):
    print(labels[i] + ':', sum_download_counts[i])

total_download_count = sum(sum_download_counts)
print('')
print('total:', total_download_count)

# print(name_arr)
# print(published_arr)
# print(subtotal_arr)

duration_arr = [[], []]
now_ts = int(datetime.now(timezone.utc).timestamp())
for ii in range(len(duration_arr)):
    for i in range(len(published_arr[ii])):
        if i < len(published_arr[ii]) - 1:
            diff = published_arr[ii][i + 1] - published_arr[ii][i]
        else:
            diff = now_ts - published_arr[ii][i]
        duration_arr[ii].append(diff / 86400)
# print(duration_arr)

n_download_per_day_arr = [[], []]
for ii in range(len(n_download_per_day_arr)):
    for i in range(len(duration_arr[ii])):
        dur = duration_arr[ii][i]
        downloads = subtotal_arr[ii][i]
        if dur <= 0:
            rate = downloads
        else:
            rate = downloads / dur
        n_download_per_day_arr[ii].append(rate)
# print(n_download_per_day_arr)


for ii in range(len(name_arr)):
    x = name_arr[ii]
    y_rate = n_download_per_day_arr[ii]
    y_total = subtotal_arr[ii]

    fig_width = max(6, len(x) * 0.6)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 8), sharex=True)

    axes[0].bar(x, y_rate, color='#4c72b0')
    axes[0].set_ylabel('Downloads per day')
    axes[0].set_title(f'{labels[ii]} - Downloads per day')
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.4)

    axes[1].bar(x, y_total, color='#55a868')
    axes[1].set_ylabel('Total downloads')
    axes[1].set_title(f'{labels[ii]} - Total downloads')
    axes[1].set_xlabel('Release')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.4)

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()