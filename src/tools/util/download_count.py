import requests
from datetime import datetime, timezone
import re
from datetime import timedelta
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

jst = timezone(timedelta(hours=9))
now_jst = datetime.now(timezone.utc).astimezone(jst)

for ii in range(len(name_arr)):
    x = name_arr[ii]
    x = [m.group(0) if (m := re.search(r'\d+\.\d+\.\d+', s)) else s for s in x]
    y_total = subtotal_arr[ii]
    y_dur = duration_arr[ii]
    y_rate = n_download_per_day_arr[ii]

    fig_width = max(6, len(x) * 0.6)
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 10), sharex=True)

    # Top: total downloads
    bars0 = axes[0].bar(range(len(x)), y_total, color='#55a868')
    axes[0].set_ylabel('Total downloads')
    axes[0].set_title(f'{labels[ii]} - Total downloads')
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.4)
    # ensure space for labels above bars
    max0 = max(y_total) if y_total else 0
    axes[0].set_ylim(0, max0 * 1.08 if max0 > 0 else 1)
    for i, v in enumerate(y_total):
        axes[0].text(i, v + (max0 * 0.01 if max0 > 0 else 0.1), f'{int(v):,}', ha='center', va='bottom', fontsize=8)
    # show x tick labels below this subplot
    axes[0].xaxis.set_ticks_position('bottom')
    axes[0].set_xticks(range(len(x)))
    axes[0].set_xticklabels(x)
    axes[0].tick_params(axis='x', rotation=45, labelbottom=True)

    # Middle: duration (days)
    bars1 = axes[1].bar(range(len(x)), y_dur, color='#ffa07a')
    axes[1].set_ylabel('Duration (days)')
    axes[1].set_title(f'{labels[ii]} - Duration (days)')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.4)
    max1 = max(y_dur) if y_dur else 0
    axes[1].set_ylim(0, max1 * 1.08 if max1 > 0 else 1)
    for i, v in enumerate(y_dur):
        axes[1].text(i, v + (max1 * 0.01 if max1 > 0 else 0.1), f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    # show x tick labels below this subplot
    axes[1].xaxis.set_ticks_position('bottom')
    axes[1].set_xticks(range(len(x)))
    axes[1].set_xticklabels(x)
    axes[1].tick_params(axis='x', rotation=45, labelbottom=True)

    # Bottom: downloads per day
    bars2 = axes[2].bar(range(len(x)), y_rate, color='#4c72b0')
    axes[2].set_ylabel('Downloads per day')
    axes[2].set_title(f'{labels[ii]} - Downloads per day')
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.4)
    max2 = max(y_rate) if y_rate else 0
    axes[2].set_ylim(0, max2 * 1.08 if max2 > 0 else 1)
    for i, v in enumerate(y_rate):
        axes[2].text(i, v + (max2 * 0.01 if max2 > 0 else 0.01), f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    axes[2].set_xlabel('Release')

    # X ticks and labels on bottom only
    axes[2].set_xticks(range(len(x)))
    axes[2].set_xticklabels(x)
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    total_for_label = sum_download_counts[GUI_IDX] if ii == GUI_IDX else sum_download_counts[CONSOLE_IDX]
    fig.text(0.01, 0.98, f'Total Downloads: {total_for_label:,}', ha='left', fontsize=10)
    fig.text(0.99, 0.98, now_jst.strftime('%Y-%m-%d %H:%M:%S JST'), ha='right', fontsize=10)

    plt.tight_layout()
    plt.show()

# ----- Cumulative downloads vs Date (separate windows per series) -----
# For each series (GUI and Console), open a separate figure showing cumulative downloads over time
series_indices = [GUI_IDX, CONSOLE_IDX]
for idx in series_indices:
    dates_ts = published_arr[idx]
    totals = subtotal_arr[idx]
    if not dates_ts:
        # still create an empty figure with message
        plt.figure(figsize=(8, 4))
        plt.title(f'{labels[idx]} - No releases')
        plt.text(0.5, 0.5, 'No data', ha='center', va='center')
        plt.axis('off')
        plt.show()
        continue

    # sort by timestamp to ensure chronological order
    pairs = sorted(zip(dates_ts, totals), key=lambda x: x[0])
    # For cumulative plot, place each release's cumulative total at the timestamp of the next release.
    # For the last release, use current time (now_ts).
    ts_list = [p[0] for p in pairs]
    cum = []
    s = 0
    for _, v in pairs:
        s += v
        cum.append(s)
    # build x timestamps shifted to next release; last point uses now_ts
    x_ts = []
    for i in range(len(ts_list)):
        if i < len(ts_list) - 1:
            x_ts.append(ts_list[i + 1])
        else:
            x_ts.append(now_ts)
    dates_sorted = [datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(jst) for ts in x_ts]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dates_sorted, cum, marker='o', linestyle='-', color='#2a9d8f')
    ax.set_xlabel('Date')
    ax.set_ylabel('Downloads')
    ax.set_title(f'{labels[idx]} - Cumulative downloads')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_tick_params(rotation=45)

    # annotate with current total downloads for this series and current JST datetime
    total_for_label = sum_download_counts[idx]
    now_jst_str = now_jst.strftime('%Y-%m-%d %H:%M:%S JST')
    # place text in bottom right corner of the axes (axes coordinates)
    ax.text(0.98, 0.02, f'Total: {total_for_label:,}\n{now_jst_str}', ha='right', va='bottom', transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

    fig.tight_layout()
    plt.show()