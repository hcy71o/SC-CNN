f = open("vctk_audio_sid_text_train_filelist.txt.cleaned", 'r')
lines = f.readlines()
test_id_list = ['p225', 'p234', 'p238', 'p245', 'p248', 'p261', 'p294', 'p302', 'p326', 'p335', 'p347']
train_list = []
test_list = []
for i, line in enumerate(lines):
    if i % 1000 == 0: print(i)
    test_case = False
    for test_id in test_id_list:
        if test_id in line:
            test_case = True
    if test_case: 
        test_list.append(line)
    else: 
        train_list.append(line)
f.close()


f1 = open("vctk_train.txt.cleaned", 'w')
for train_line in train_list:
    f1.write(train_line)
f1.close()

f2 = open("vctk_test.txt.cleaned", 'w')
for test_line in test_list:
    f2.write(test_line)
f2.close()
    