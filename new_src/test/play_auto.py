# -*- coding: utf-8 -*-

import subprocess
from othello_py import *
import tkinter

offset_y = 10
offset_x = 10
rect_size = 60
circle_offset = 3

ai_exe = subprocess.Popen('./a.exe'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#ai_exe.stdin.write((str(ai_player) + '\n').encode('utf-8'))
#ai_exe.stdin.flush()
record = ''
vals = []


o = None
legal_buttons = []

app = tkinter.Tk()
app.geometry('1000x700')
app.title('Egaroucid5')
canvas = tkinter.Canvas(app, width=1000, height = 700)
pixel_virtual = tkinter.PhotoImage(width=1, height=1)

def on_closing():
    global ai_exe
    ai_exe.kill()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)

for y in range(hw):
    for x in range(hw):
        canvas.create_rectangle(offset_x + rect_size * x, offset_y + rect_size * y, offset_x + rect_size * (x + 1), offset_y + rect_size * (y + 1), outline='black', width=2, fill='#16a085')

stone_str = tkinter.StringVar()
stone_str.set('*Black 2 - 2 White ')
stone_label = tkinter.Label(canvas, textvariable=stone_str, font=('', 30))
stone_label.place(x=250, y=600, anchor=tkinter.CENTER)

val_str = tkinter.StringVar()
val_str.set('value: 0')
val_label = tkinter.Label(canvas, textvariable=val_str, font=('', 20))
val_label.place(x=10, y=650)

def start():
    global o, record, vals, ai_player
    record = ''
    vals = [-1000 for _ in range(60)]
    o = othello()
    o.check_legal()
    show_grid()

start_button = tkinter.Button(canvas, text='Start', command=start)
start_button.place(x=600, y=10)

def end_game():
    result = o.n_stones[0] - o.n_stones[1]
    if result > 0:
        result += hw2 - sum(o.n_stones)
    elif result < 0:
        result -= hw2 - sum(o.n_stones)

def translate_coord(y, x):
    return chr(ord('a') + x) + str(y + 1)

def ai():
    global clicked, record
    ai_exe.stdin.write((str(o.player) + '\n').encode('utf-8'))
    ai_exe.stdin.flush()
    grid_str = ''
    for i in range(hw):
        for j in range(hw):
            grid_str += '0' if o.grid[i][j] == 0 else '1' if o.grid[i][j] == 1 else '.'
        grid_str += '\n'
    print(grid_str)
    ai_exe.stdin.write(grid_str.encode('utf-8'))
    ai_exe.stdin.flush()
    val, coord = ai_exe.stdout.readline().decode().split()
    val = float(val)
    y = int(coord[1]) - 1
    x = ord(coord[0]) - ord('a')
    vals[sum(o.n_stones) - 4] = val
    val_str.set('value: ' + str(val))
    record += translate_coord(y, x)
    print(y, x)
    clicked = True
    o.move(y, x)
    if not o.check_legal():
        o.player = 1 - o.player
        if not o.check_legal():
            o.print_info()
            o.player = -1
            end_game()
            print('end')
    s = ''
    if o.player == 0:
        s += '*'
    else:
        s += ' '
    s += 'Black '
    s += str(o.n_stones[0])
    s += ' - '
    s += str(o.n_stones[1])
    s += ' White'
    if o.player == 1:
        s += '*'
    else:
        s += ' '
    stone_str.set(s)
    #o.print_info()
    show_grid()

def show_grid():
    global clicked, legal_buttons
    for button in legal_buttons:
        button.place_forget()
    legal_buttons = []
    for y in range(hw):
        for x in range(hw):
            try:
                canvas.delete(str(y) + '_' + str(x))
            except:
                pass
            if o.grid[y][x] == vacant:
                continue
            color = ''
            if o.grid[y][x] == black:
                color = 'black'
            elif o.grid[y][x] == white:
                color = 'white'
            elif o.grid[y][x] == legal:
                continue
            canvas.create_oval(offset_x + rect_size * x + circle_offset, offset_y + rect_size * y + circle_offset, offset_x + rect_size * (x + 1) - circle_offset, offset_y + rect_size * (y + 1) - circle_offset, width=0, fill=color, tag=str(y) + '_' + str(x))
    app.after(10, ai)

canvas.place(y=0, x=0)
app.mainloop()
