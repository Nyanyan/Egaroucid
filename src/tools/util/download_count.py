import requests
from datetime import datetime, timezone
import matplotlib.pyplot as plt

r = requests.get('https://api.github.com/repos/Nyanyan/Egaroucid/releases')

# GUI, Console
name_arr = [[], []]
published_arr = [[], []]
subtotal_arr = [[], []]

total_download_count = 0
for item in reversed(r.json()):
    print("tag_name: ", item["tag_name"])
    print("name: ", item["name"])
    print("published: ", item["published_at"])
    subtotal = 0
    for i in range(len(item["assets"])):
        print("download count:", item["assets"][i]["download_count"], " name:", item["assets"][i]["name"])
        total_download_count += int(item["assets"][i]["download_count"])
        subtotal += int(item["assets"][i]["download_count"])
    print('subtotal:', subtotal)
    print("")
    dt = datetime.strptime(item["published_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    item_idx = -1
    if item["tag_name"][:9] == 'console_v':
        item_idx = 1
    elif item["tag_name"][0] == 'v':
        item_idx = 0
    if item_idx != -1:
        name_arr[item_idx].append(item["name"])
        published_arr[item_idx].append(int(dt.timestamp()))
        subtotal_arr[item_idx].append(subtotal)
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


labels = ['GUI', 'Console']
for ii in range(len(name_arr)):
    x = name_arr[ii]
    y = n_download_per_day_arr[ii]

    plt.figure(figsize=(max(6, len(x) * 0.6), 4))
    plt.bar(x, y, color='#4c72b0')
    plt.xlabel('Release')
    plt.ylabel('Downloads per day')
    plt.title(labels[ii])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()