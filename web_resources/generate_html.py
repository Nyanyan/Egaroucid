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

langs = []
with open('lang.txt', 'r', encoding='utf-8') as f:
    for line in f.read().splitlines():
        dr = line.split()[-1]
        text = line.replace(' ' + dr, '')
        langs.append([dr, text])

with open('main_page_url.txt', 'r', encoding='utf-8') as f:
    main_page_url = f.readline()

with open(elements_dir + '/head.html', 'r', encoding='utf-8') as f:
    head = f.read()

with open(elements_dir + '/head2.html', 'r', encoding='utf-8') as f:
    head2 = f.read()

with open(elements_dir + '/download.txt', 'r', encoding='utf-8') as f:
    raw_download_data = f.read().splitlines()
download_data = []
for datum in raw_download_data:
    link = datum.split()[-1]
    text = datum.replace(' ' + link, '')
    datum_html = '<div class="download_button"><a class="download_a" href="' + link + '">' + text + '</a></div>\n'
    download_data.append(datum_html)

menu = '<div class="menu_bar">\n'
#menu += '<a class="menu_a" href="' + main_page_url + elements_dir + '"><img class="bar_icon" src="https://raw.githubusercontent.com/Nyanyan/Nyanyan.github.io/master/img/favicon.jpg"></a>\n'
with open(elements_dir + '/menu_elements.txt', encoding='utf-8') as f:
    menu_elems = f.read().splitlines()
    n_menu_elems = []
    for elem in menu_elems:
        link = elem.split()[-1]
        text = elem.replace(' ' + link, '')
        n_menu_elems.append([text, link])
    menu_elems = [elem for elem in n_menu_elems]
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

with open(elements_dir + '/main_page_title.txt', 'r', encoding='utf-8') as f:
    main_page_title = f.read()

#with open(elements_dir + '/main_page_description.txt', 'r', encoding='utf-8') as f:
#    main_page_description = f.read()

with open(elements_dir + '/meta_description.txt', 'r', encoding='utf-8') as f:
    meta_description = f.readline()

section_head1 = '<div>\n<h2>'
section_head2 = '</h2>\n'
section_foot = '</div>\n'

centering_head = '<div style="text-align: center">\n'
centering_foot = '</div>'

link1 = '<a href="'
link2 = '" target="_blank" el=”noopener noreferrer”>'
link3 = '</a>'

link21 = '<a href="'
link22 = '">'
link23 = '</a>'

def create_html(dr):
    alternate = ''
    for lang in langs:
        if lang[0] == elements_dir:
            continue
        if dr[3:]:
            alternate += '<link rel="alternate" hreflang="' + lang[0] + '" href="' + main_page_url + lang[0] + '/' + dr[3:] + '/"/>\n'
        else:
            alternate += '<link rel="alternate" hreflang="' + lang[0] + '" href="' + main_page_url + lang[0] + '/"/>\n'
    if dr[3:]:
        alternate += '<link rel="alternate" hreflang="x-default" href="' + main_page_url + 'en/' + dr[3:] + '/">\n'
    else:
        alternate += '<link rel="alternate"  hreflang="x-default" href="' + main_page_url + 'en/"/>\n'
    with open(dr + '/title.txt', 'r', encoding='utf-8') as f:
        page_title = f.readline()
    with open(dr + '/index.md', 'r', encoding='utf-8') as f:
        md = f.read()
    #page_title = ''
    md_split = md.splitlines()
    raw_html = 0
    for i, elem in enumerate(md_split):
        html_elems = re.findall('\<.+?\>', elem)
        for html_elem in html_elems:
            if html_elem[:2] == '</':
                raw_html -= 1
            else:
                raw_html += 1
        # section tags
        if elem[:2] == '# ':
            elem = '<h1>' + elem[2:] + '</h1>'
        elif elem[:3] == '## ':
            elem = '<h2>' + elem[3:] + '</h2>'
        elif elem[:4] == '### ':
            elem = '<h3>' + elem[3:] + '</h3>'
        elif elem[:5] == '#### ':
            elem = '<h4>' + elem[3:] + '</h4>'
        # links
        links = re.findall('\[.+?\]\(.+?\)', elem)
        for link in links:
            text, url = link[1:-1].split('](')
            if url[0] != '.':
                html_link = link1 + url + link2 + text + link3
            else:
                html_link = link21 + url + link22 + text + link23
            elem = elem.replace(link, html_link)
        # bold
        bolds = re.findall('\*\*.+?\*\*', elem)
        for bold in bolds:
            html_bold = '<b>' + bold[2:-2] + '</b>'
            elem = elem.replace(bold, html_bold)
        # paragraph
        if raw_html == 0:
            elem = '<p>' + elem + '</p>'
        # modify data
        md_split[i] = elem
    html = ''
    html += '<div class="box">\n'
    this_page_url = main_page_url + dr
    head_title = '<title>' + page_title + '</title>\n'
    og_image = '<meta property="og:image" content="' + this_page_url + '/img/eyecatch.png" />\n'
    html += '<p></p>\n'
    html += tweet.replace('DATA_URL', this_page_url).replace('DATA_TEXT', page_title) + ' \n'
    for lang_dr, lang_name in langs:
        original_lang = dr.split('/')[0]
        if lang_dr == original_lang:
            continue
        modified_dr = dr[len(original_lang) + 1:]
        lang_link = main_page_url + lang_dr + '/' + modified_dr
        html += link21 + lang_link + link22 + lang_name + link23 + ' \n'
    additional_head = '<meta property="og:url" content="' + this_page_url + '/" />\n'
    additional_head += '<meta property="og:title" content="' + page_title + '" />\n'
    #additional_head += '<meta property="og:description" content="' + main_page_description + '" />\n'
    additional_head += '<meta property="og:description" content="' + meta_description + '" />\n'
    additional_head += '<link rel="canonical" href="' + this_page_url + '/">\n'
    additional_head += '<meta name="description" content="' + meta_description + '"/>\n'
    try:
        with open(dr + '/additional_head.html', 'r', encoding='utf-8') as f:
            additional_head += f.read()
    except:
        pass
    last_empty = False
    for line in md_split:
        if (not last_empty) and line == '':
            last_empty = True
        else:
            if line:
                html += line
            #else:
            #    html += line + '<br>\n'
            last_empty = False
    html += '<p></p>\n'
    html += '</div>\n'
    out_dr = 'generated/' + dr
    if not os.path.exists(out_dr):
        os.mkdir(out_dr)
    with open(out_dr + '/index.html', 'w', encoding='utf-8') as f:
        f.write(head + alternate + og_image + additional_head + head_title + head2 + menu + html + foot)
    shutil.copy(css_file, out_dr + '/style.css')
    try:
        shutil.copytree(dr + '/img', out_dr + '/img')
    except:
        pass
    try:
        with open(dr + '/additional_files.txt', 'r', encoding='utf-8') as f:
            additional_files = f.read().splitlines()
        for additional_file in additional_files:
            shutil.copy(dr + '/' + additional_file, out_dr + '/' + additional_file)
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
