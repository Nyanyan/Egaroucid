import glob
import os
import shutil
import sys
import re

elements_dir = sys.argv[1]

if os.path.exists('generated/' + elements_dir):
    shutil.rmtree('generated/' + elements_dir)
os.mkdir('generated/' + elements_dir)

css_file = elements_dir + '/style.css'

with open('main_page_url.txt', 'r', encoding='utf-8') as f:
    main_page_url = f.readline()

with open(elements_dir + '/download.txt', 'r', encoding='utf-8') as f:
    raw_download_data = f.read().splitlines()
download_data = []
for datum in raw_download_data:
    link = datum.split()[-1]
    text = datum.replace(' ' + link, '')
    datum_html = '<div class="download_button"><a class="download_a" href="' + link + '">' + text + '</a></div>\n'
    download_data.append(datum_html)


with open(elements_dir + '/head.html', 'r', encoding='utf-8') as f:
    head = f.read()

menu = '<div class="menu_bar">\n'
menu += '<a class="menu_a" href="' + main_page_url + elements_dir + '"><img class="bar_icon" src="img/icon.png"></a>\n'
with open(elements_dir + '/menu_elements.txt', encoding='utf-8') as f:
    menu_elems = f.read().splitlines()
    menu_elems = [elem.split() for elem in menu_elems]
for text, link in menu_elems:
    if link[0] == '/':
        link = main_page_url + elements_dir + link
        menu += '<div class="menu_button"><a class="menu_a" href="' + link + '">' + text + '</a></div>'
    else:
        menu += '<div class="menu_button"><a class="menu_a" href="' + link + '" target="_blank" el=”noopener noreferrer”>' + text + '</div></a>\n'
menu += '</div>\n'

with open(elements_dir + '/tweet.html', 'r', encoding='utf-8') as f:
    tweet = f.read()

with open(elements_dir + '/foot.html', 'r', encoding='utf-8') as f:
    foot = f.read()

with open(elements_dir + '/sections.txt', 'r', encoding='utf-8') as f:
    sections = f.read().splitlines()

section_head1 = '<div>\n<h2>'
section_head2 = '</h2>\n'
section_foot = '</div>\n'

centering_head = '<div style="text-align: center">\n'
centering_foot = '</div>'

link1 = '<a href="'
link2 = '" target="_blank" el=”noopener noreferrer”>'
link3 = '</a>'

def create_html(dr):
    with open(dr + '/index.md', 'r', encoding='utf-8') as f:
        md = f.read()
    md_split = md.splitlines()
    for i, elem in enumerate(md_split):
        # special replace
        ## download
        if elem == 'DOWNLOAD_BUTTON_REPLACE':
            elem = ''
            for download_elem in download_data:
                elem += download_elem
        # section tags
        if elem[:2] == '# ':
            elem = '<h1>' + elem[2:] + '</h1>'
        if elem[:3] == '## ':
            elem = '<h2>' + elem[3:] + '</h2>'
        if elem[:4] == '### ':
            elem = '<h3>' + elem[3:] + '</h3>'
        # links
        links = re.findall('\[.+?\]\(.+?\)', elem)
        for link in links:
            text, url = link[1:-1].split('](')
            html_link = link1 + url + link2 + text + link3
            elem = elem.replace(link, html_link)
        # bold
        bolds = re.findall('\*\*.+?\*\*', elem)
        for bold in bolds:
            html_bold = '<b>' + bold[2:-2] + '</b>'
            elem = elem.replace(bold, html_bold)
        # modify data
        md_split[i] = elem
    html = '<div class="box">\n'
    last_empty = False
    for line in md_split:
        if last_empty == False and line == '':
            last_empty = True
        else:
            if line:
                html += line
            else:
                html += line + '<br>\n'
            last_empty = False
    html += '</div>\n'
    out_dr = 'generated/' + dr
    if not os.path.exists(out_dr):
        os.mkdir(out_dr)
    with open(out_dr + '/index.html', 'w', encoding='utf-8') as f:
        f.write(head + menu + tweet + html + foot)
    shutil.copy(css_file, out_dr + '/style.css')
    try:
        shutil.copytree(dr + '/img', out_dr + '/img')
    except:
        pass
    tasks = []
    try:
        with open(dr + '/tasks.txt', 'r', encoding='utf-8') as f:
            tasks = f.read().splitlines()
    except:
        pass
    if tasks:
        print(dr, 'new tasks', tasks)
        for task in tasks:
            create_html(dr + '/' + task)

create_html(elements_dir)