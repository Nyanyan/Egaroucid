import glob
import os
import shutil
import sys

shutil.rmtree('generated')
os.mkdir('generated')

elements_dir = sys.argv[1]

css_file = elements_dir + '/style.css'

generated = ''

with open(elements_dir + '/head.html', 'r', encoding='utf-8') as f:
    head = f.read()

with open(elements_dir + '/foot.html', 'r', encoding='utf-8') as f:
    foot = f.read()

with open(elements_dir + '/sections.txt', 'r', encoding='utf-8') as f:
    sections = f.read().splitlines()

with open(elements_dir + '/intro.html', 'r', encoding='utf-8') as f:
    generated += f.read()

section_head1 = '<div>\n<h2>'
section_head2 = '</h2>\n'
section_foot = '</div>\n'

centering_head = '<div style="text-align: center">\n'
centering_foot = '</div>'

for section in sections:
    print(section)
    section_name, section_file = section.split()
    
    if not os.path.exists('generated/' + section_file):
        os.mkdir('generated/' + section_file)
    
    generated += centering_head
    
    with open(elements_dir + '/' + section_file + '.html', 'r', encoding='utf-8') as f:
        generated += section_head1 + section_name + section_head2 + f.read() + section_foot
    
    with open(elements_dir + '/' + section_file + '/info.txt', 'r', encoding='utf-8') as f:
        info_names = f.read().splitlines()
    
    generated += '<table>\n<tr>\n'
    for name in info_names:
        generated += '<th>' + name + '</th>'
    generated += '</tr>\n'

    section_dirs = set(glob.glob(elements_dir + '/' + section_file + '/*')) - set(glob.glob(elements_dir + '/' + section_file + '/*.txt'))

    section_dirs = list(section_dirs)
    section_dirs = [[int(elem.split('\\')[-1].split('_')[0]), elem] for elem in section_dirs]
    section_dirs.sort(reverse=True)

    for _, dr in section_dirs:
        with open(dr + '/info.txt', 'r', encoding='utf-8') as f:
            info = f.read().splitlines()
        title = info[0]
        print(info)
        try:
            with open(dr + '/index.md', 'r', encoding='utf-8') as f:
                md = f.read()
            md_split = md.splitlines()
            for i, elem in enumerate(md_split):
                if elem[:2] == '# ':
                    md_split[i] = '<h1>' + elem[2:] + '</h1>'
                if elem[:3] == '## ':
                    md_split[i] = '<h2>' + elem[3:] + '</h2>'
                if elem[:4] == '### ':
                    md_split[i] = '<h3>' + elem[3:] + '</h3>'
            html = '<div class="box">\n'
            last_empty = False
            for line in md_split:
                if last_empty == False and line == '':
                    last_empty = True
                else:
                    if line and line[0] == '<':
                        html += line
                    else:
                        html += line + '<br>\n'
                    last_empty = False
            html += '</div>\n'
            drc = 'generated/' + section_file + '/' + title.replace(' ', '').replace('/', '_')
            if not os.path.exists(drc):
                os.mkdir(drc)
            with open(drc + '/index.html', 'w', encoding='utf-8') as f:
                f.write(head + html + foot)
            shutil.copy(css_file, drc + '/style.css')
            try:
                shutil.copytree(dr + '/img', 'generated/' + section_file + '/' + title.replace(' ', '').replace('/', '_') + '/img')
            except:
                pass
            generated += '<tr>\n'
            generated += '<td><a href=' + section_file + '/' + title.replace(' ', '').replace('/', '_') + '>' + title + '</a><br></td>\n'
            for datum in info[1:]:
                generated += '<td>' + datum + '</td>\n'
            generated += '</tr>\n'
        except:
            generated += '<tr>\n'
            for datum in info:
                generated += '<td>' + datum + '</td>\n'
            generated += '</tr>\n'
    generated += '</table>'
    generated += centering_foot

with open('generated/index.html', 'w', encoding='utf-8') as f:
    f.write(head + generated + foot)
shutil.copy(css_file, 'generated/style.css')
shutil.copytree(elements_dir + '/img', 'generated/img')