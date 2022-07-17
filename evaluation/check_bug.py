import tkinter

hw = 8
hw2 = 64

lines = []

with open('check_bug_data.txt', 'r') as f:
    line = 'a'
    while line != '':
        line = f.readline()
        lines.append(line)

# 定数
offset_y = 10
offset_x = 10
rect_size = 60
circle_offset = 3

# GUI部分
app = tkinter.Tk()
app.geometry('500x700')
app.title('Isevot')
canvas = tkinter.Canvas(app, width=500, height = 700)

# 盤面の作成
for y in range(hw):
    for x in range(hw):
        canvas.create_rectangle(offset_x + rect_size * x, offset_y + rect_size * y, offset_x + rect_size * (x + 1), offset_y + rect_size * (y + 1), outline='black', width=2, fill='#16a085')

line_idx = 0

def show_grid():
    global line_idx
    line = lines[line_idx]
    print(line_idx, line)
    line_split = [int(elem) for elem in line.split()]
    cells = line_split[:10]
    mask_3bit = line_split[10]
    mask_1bit = line_split[11]
    mobility_place = line_split[12]
    for y in range(hw):
        for x in range(hw):
            try:
                canvas.delete(str(y) + '_' + str(x))
            except:
                pass
    idx = 0
    labels = []
    for cell in cells:
        if cell == 64:
            continue
        y = cell // hw
        x = cell % hw
        canvas.create_oval(offset_x + rect_size * x + circle_offset, offset_y + rect_size * y + circle_offset, offset_x + rect_size * (x + 1) - circle_offset, offset_y + rect_size * (y + 1) - circle_offset, width=2, fill='black', tag=str(y) + '_' + str(x))
        label = tkinter.Label(text=str(idx), bg='white')
        label.place(x=offset_x + rect_size * x + circle_offset, y=offset_y + rect_size * y + circle_offset)
        labels.append(label)
        idx += 1
    for cell in range(hw2):
        bit = 1 & (mask_3bit >> cell)
        if bit:
            y = cell // hw
            x = cell % hw
            canvas.create_oval(offset_x + rect_size * x + circle_offset, offset_y + rect_size * y + circle_offset, offset_x + rect_size * (x + 1) - circle_offset, offset_y + rect_size * (y + 1) - circle_offset, width=2, fill='blue', tag=str(y) + '_' + str(x))
    for cell in range(hw2):
        bit = 1 & (mask_1bit >> cell)
        if bit:
            y = cell // hw
            x = cell % hw
            canvas.create_oval(offset_x + rect_size * x + circle_offset, offset_y + rect_size * y + circle_offset, offset_x + rect_size * (x + 1) - circle_offset, offset_y + rect_size * (y + 1) - circle_offset, width=2, fill='yellow', tag=str(y) + '_' + str(x))
    if mobility_place != -1:
        y = mobility_place // hw
        x = mobility_place % hw
        canvas.create_oval(offset_x + rect_size * x + circle_offset, offset_y + rect_size * y + circle_offset, offset_x + rect_size * (x + 1) - circle_offset, offset_y + rect_size * (y + 1) - circle_offset, width=2, fill='red', tag=str(y) + '_' + str(x))
    app.update()
    input('next')
    for label in labels:
        label.place_forget()
    line_idx += 1
    show_grid()

canvas.place(y=0, x=0)
show_grid()
#app.mainloop()