import requests

r = requests.get('https://api.github.com/repos/Nyanyan/Egaroucid/releases')

total_download_count = 0
for item in reversed(r.json()):
    print("tag_name: ",item["tag_name"])
    print("name: ", item["name"])
    print(item["assets"][0]["url"])
    print("download count: ", item["assets"][0]["download_count"])
    total_download_count += int(item["assets"][0]["download_count"])
    print("")

print('total:', total_download_count)