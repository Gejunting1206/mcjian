from easygui import enterbox
from const import *


def input(key):
    if key == 'escape':
        msg = enterbox('恭喜你！你的成绩是：' + str(int(player.position.x)) + '。留下你的名字吧，谢谢', '跑酷')
        if msg == '':
            msg = '匿名'
        print(msg+':'+str(int(player.position.x)))
        with open("charts.txt", "r", encoding="utf-8") as f:
            _chats_list = f.readlines()
            chats_list = []
            for chat in _chats_list:
                if chat.isspace():
                    continue
                else:
                    chat = chat.replace("\n", "")
                    chat = chat.replace("\t", "")
                    chats_list.append(chat)

        # print(chats_list)
        flag = False
        for chat in chats_list:
            # print(msg, chat.split(':')[0])
            try:
                global i
                i = chats_list.index(chat)
            except:
                pass
            if msg == chat.split(':')[0]:
                if int(player.position.x) > int(chat.split(':')[1]):
                    chats_list[i] = msg + ':' + str(int(player.position.x))
                flag = True
        # print(flag)
        if not flag:
            chats_list.append(msg + ':' + str(int(player.position.x)))
        chats_list.sort(key=lambda x: int(x.split(':')[1]), reverse=True)
        flag = True
        with open("charts.txt", "w", encoding="utf-8") as f:
            for chat in chats_list:
                if flag:
                    f.write(chat)
                    flag = False
                else:
                    f.write('\n'+chat)
        quit()


flag = Flag()
sky = Sky()
hand = Hand()

cnt = 0


def update():
    global position, cnt
    if held_keys['left shift']:
        player.speed = 2
    else:
        player.speed = 5


y = 100
z = 0
for x in range(1000):
    if random.randint(0, 6) == 0 and x != 0:
        y += random.randint(-10, 1)
    if random.randint(0, 5) == 0 and x != 0:
        z += random.randint(-1, 1)
    if random.randint(0, 19) == 0 and x != 0:
        shooter = Shooter(position=(x, y, z-2))
        block = Block(position=(x, y-1, z-2), id=2)
        block = Block(position=(x, y-1, z-1), id=2)
        block = Block(position=(x, y-1, z), id=2)
    if random.randint(0, 29) == 0 and x != 0:
        pass
    else:
        block = Block(position=(x, y, z))

app.run()
