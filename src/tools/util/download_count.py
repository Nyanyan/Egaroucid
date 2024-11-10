import requests

r = requests.get('https://api.github.com/repos/Nyanyan/Egaroucid/releases')

total_download_count = 0
for item in reversed(r.json()):
    print("tag_name: ",item["tag_name"])
    print("name: ", item["name"])
    subtotal = 0
    for i in range(len(item["assets"])):
        print("download count:", item["assets"][i]["download_count"], " name:", item["assets"][i]["name"])
        total_download_count += int(item["assets"][i]["download_count"])
        subtotal += int(item["assets"][i]["download_count"])
    print('subtotal:', subtotal)
    print("")

print('total:', total_download_count)