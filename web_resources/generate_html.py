import glob
import os
import shutil
import sys
import re
from PIL import Image


GUI_VERSION_DOT = '7.1.1'
GUI_DATE_STR = '2024/06/13'

CONSOLE_VERSION_DOT = '7.1.0'
CONSOLE_DATE_STR = '2024/06/06'

GUI_VERSION_UNDERBAR = GUI_VERSION_DOT.replace('.', '_')
CONSOLE_VERSION_UNDERBAR = CONSOLE_VERSION_DOT.replace('.', '_')

GUI_RELEASE_IDENTIFIER = 'GUI_DOWNLOAD_TABLE_HERE'
CONSOLE_RELEASE_IDENTIFIER = 'CONSOLE_DOWNLOAD_TABLE_HERE'
CONSOLE_SOURCE_RELEASE_IDENTIFIER = 'CONSOLE_SOURCE_TABLE_HERE'

DOWNLOAD_BUTTON_URL = 'https://github.com/Nyanyan/Egaroucid/releases/download/v' + GUI_VERSION_DOT + '/Egaroucid_' + GUI_VERSION_UNDERBAR + '_Installer.exe'


MAX_IMG_SIZE = 400

def convert_img(file):
    img = Image.open(file)
    shorter_side = min(img.width, img.height)
    ratio = min(1, MAX_IMG_SIZE / shorter_side)
    width = int(img.width * ratio)
    height = int(img.height * ratio)
    img_resized = img.resize((width, height))
    return img_resized

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

with open(elements_dir + '/release_gui.html', 'r', encoding='utf-8') as f:
    release_gui_html = f.read().replace('DATE', GUI_DATE_STR).replace('VERSION_DOT', GUI_VERSION_DOT).replace('VERSION_UNDERBAR', GUI_VERSION_UNDERBAR)

with open(elements_dir + '/release_console_zip.html', 'r', encoding='utf-8') as f:
    release_console_zip_html = f.read().replace('DATE', CONSOLE_DATE_STR).replace('VERSION_DOT', CONSOLE_VERSION_DOT).replace('VERSION_UNDERBAR', CONSOLE_VERSION_UNDERBAR)

with open(elements_dir + '/release_console_source.html', 'r', encoding='utf-8') as f:
    release_console_source_html = f.read().replace('DATE', CONSOLE_DATE_STR).replace('VERSION_DOT', CONSOLE_VERSION_DOT).replace('VERSION_UNDERBAR', CONSOLE_VERSION_UNDERBAR)

with open(elements_dir + '/download_button.html', 'r', encoding='utf-8') as f:
    download_button = f.read().replace('REPLACE_DOWNLOAD_BUTTON_URL', DOWNLOAD_BUTTON_URL).replace('REPLACE_DOWNLOAD_BUTTON_VERSION', GUI_VERSION_DOT)


section_head1 = '<div>\n<h2>'
section_head2 = '</h2>\n'
section_foot = '</div>\n'

centering_head = '<div style="text-align: center">\n'
centering_foot = '</div>'

link1 = '<a href="'
link2 = '" target="_blank" el="noopener noreferrer">'
link3 = '</a>'

link21 = '<a font-size="1.5em" href="'
link22 = '">'
link23 = '</a>'

tex_js = '''
<script type="text/javascript" async>
    window.MathJax = {
        chtml: {
        matchFontHeight: false
        },
        tex: {
        inlineMath: [['$', '$']]
        },
        svg: {
        fontCache: 'global'
        }
    };
    (function () {
        const script = document.createElement('script');
        if (navigator.userAgent.includes("Chrome") || navigator.userAgent.includes("Firefox"))
            script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js";
        else
            script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js";
        script.async = true;
        document.head.appendChild(script);
    })();
</script>
'''

def judge_raw_html(html_elem):
    html_tags = ['table', 'tr', 'td', 'th', 'a', 'div', 'ul', 'li', 'p', 'span', 'canvas', 'details', 'summary', 'code', 'label', 'script']
    count = 0
    for tag in html_tags:
        if '<' + tag in html_elem:
            count += 1
        elif '</' + tag in html_elem:
            count -= 1
    #print(html_elem)
    # ignore img, input, br
    return count

def create_html(dr):
    noenglish = False
    try:
        with open(dr + '/noenglish.txt', 'r', encoding='utf-8') as f:
            noenglish = True
    except:
        pass
    need_tex_js = False
    try:
        with open(dr + '/need_tex_js.txt', 'r', encoding='utf-8') as f:
            need_tex_js = True
    except:
        pass
    alternate = ''
    if not noenglish:
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
    need_table_of_contents = md.find('INSERT_TABLE_OF_CONTENTS_HERE') != -1
    table_of_contents = []
    md_split = md.splitlines()
    raw_html = 0
    last_h3_title = ''
    for i, elem in enumerate(md_split):
        while elem and (elem[0] == ' ' or elem[0] == '\t'):
            elem = elem[1:]
        html_elems = re.findall('\<.+?\>', elem)
        for html_elem in html_elems:
            raw_html += judge_raw_html(html_elem)
        # download button
        if 'REPLACE_DOWNLOAD_BUTTON_HERE' in elem:
            elem = elem.replace('REPLACE_DOWNLOAD_BUTTON_HERE', download_button)
        # download tables
        if GUI_RELEASE_IDENTIFIER in elem:
            elem = elem.replace(GUI_RELEASE_IDENTIFIER, release_gui_html)
        if CONSOLE_RELEASE_IDENTIFIER in elem:
            elem = elem.replace(CONSOLE_RELEASE_IDENTIFIER, release_console_zip_html)
        if CONSOLE_SOURCE_RELEASE_IDENTIFIER in elem:
            elem = elem.replace(CONSOLE_SOURCE_RELEASE_IDENTIFIER, release_console_source_html)
        # section tags
        if elem[:2] == '# ':
            elem = '<h1>' + elem[2:] + '</h1>'
        elif elem[:3] == '## ':
            if need_table_of_contents:
                table_of_contents.append([elem[3:], elem[3:], []])
            last_h3_title = elem[3:]
            elem = '<h2 id="' + elem[3:] + '">' + elem[3:] + '</h2>'
        elif elem[:4] == '### ':
            if need_table_of_contents:
                table_of_contents[-1][2].append([elem[4:], last_h3_title + '_' + elem[4:], []])
            elem = '<h3 id="' + last_h3_title + '_' + elem[4:] + '">' + elem[4:] + '</h3>'
        elif elem[:5] == '#### ':
            elem = '<h4>' + elem[5:] + '</h4>'
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
        # code
        codes = re.findall('```.+?```', elem)
        for code in codes:
            html_code = '<code>' + code[3:-3] + '</code>'
            elem = elem.replace(code, html_code)
        # paragraph
        if raw_html == 0 and len(elem):
            elem = '<p>' + elem + '</p>'
        # img
        if elem[:4] == '<img':
            img_file_name = ''
            for elem_elem in elem.split():
                if elem_elem[:4] == 'src=':
                    img_file_name = elem_elem[5:-1]
                    img_file_name = img_file_name.replace('"', '').replace('>', '')
            if img_file_name:
                img = Image.open(dr + '/' + img_file_name)
                shorter_side = min(img.width, img.height)
                ratio = min(1, MAX_IMG_SIZE / shorter_side)
                img_width = int(img.width * ratio)
                img_height = int(img.height * ratio)
                elem = '<img width="' + str(img_width) + '" height="' + str(img_height) + '"' + elem[4:]
        # modify data
        md_split[i] = elem
    # table of contents
    if need_table_of_contents:
        table_of_contents_html = '<details><summary>目次</summary><ol class="table_of_contents_ol">'
        for name1, id1, children1 in table_of_contents:
            table_of_contents_html += '<span class="table_of_contents_li"><li><a href="#' + id1 + '">' + name1 + '</a>'
            if children1:
                table_of_contents_html += '<ul>'
                for name2, id2, _ in children1:
                    table_of_contents_html += '<li><a href="#' + id2 + '">' + name2 + '</a></li>'
                table_of_contents_html += '</ul>'
            table_of_contents_html += '</li></span>'
        table_of_contents_html += '</ol></details>'
        for i in range(len(md_split)):
            if 'INSERT_TABLE_OF_CONTENTS_HERE' in md_split[i]:
                md_split[i] = md_split[i].replace('INSERT_TABLE_OF_CONTENTS_HERE', table_of_contents_html)
    html = ''
    html += '<div class="box">\n'
    this_page_url = main_page_url + dr
    head_title = '<title>' + page_title + '</title>\n'
    og_image = '<meta property="og:image" content="' + this_page_url + '/img/eyecatch.png" />\n'
    html += '<p></p>\n'
    html += tweet.replace('DATA_URL', this_page_url).replace('DATA_TEXT', page_title) + ' \n'
    if not noenglish:
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
    if need_tex_js:
        additional_head += tex_js
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
    if os.path.exists(dr + '/img'):
        img_files = glob.glob(dr + '/img/**')
        os.mkdir(out_dr + '/img')
        for file in img_files:
            resized_img = convert_img(file)
            file_name = file.split('\\')[-1]
            resized_img.save(out_dr + '/img/' + file_name)
    try:
        with open(dr + '/additional_files.txt', 'r', encoding='utf-8') as f:
            additional_files_file_str = f.read().splitlines()
        additional_files = []
        additional_drs = []
        for line in additional_files_file_str:
            if len(line.replace('\\', '/').split('/')) >= 2:
                additional_dr = line.replace('\\', '/').split('/')[0]
                additional_drs.append(additional_dr)
            additional_files.extend(glob.glob(dr + '/' + line))
        for additional_dr in additional_drs:
            os.mkdir(out_dr + '/' + additional_dr)
        for additional_file in additional_files:
            additional_filename = additional_file[len(dr + '/'):]
            shutil.copy(additional_file, out_dr + '/' + additional_filename)
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
