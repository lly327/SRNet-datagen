import os
import random
import numpy as np

def is_chinese(ch):
    #uc=ch.decode('utf-8')
    if u'\u4e00' <= ch<=u'\u9fff':
        return True
    else:
        return False

def check_symb_frac(txt, f=0.35):
        """
        T/F return : T iff fraction of symbol/special-charcters in
                     txt is less than or equal to f (default=0.25).
        """
        chcnt = 0
        line = txt  # .decode('utf-8')
        for ch in line:
            if ch.isalnum() or is_chinese(ch):
                chcnt += 1
        return float(chcnt) / (len(txt) + 0.0) > f
        # return np.sum([not ch.isalnum() for ch in txt])/(len(txt)+0.0) <= f

def is_good(txt, f=0.35):
        """
        T/F return : T iff the lines in txt (a list of txt lines)
                     are "valid".
                     A given line l is valid iff:
                         1. It is not empty.
                         2. symbol_fraction > f
                         3. Has at-least self.min_nchar characters
                         4. Not all characters are i,x,0,O,-
        """

        def is_txt(l):
            char_ex = ['i', 'I', 'o', 'O', '0', '-']
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)

        return [(len(l) > 1
                 and check_symb_frac(l, f)
                 and is_txt(l)) for l in txt]

def text_source(root_dir,min_char=1,max_char=5):
    files = os.listdir(root_dir)
    random.shuffle(files)
    txt = []
    for filename in files:
        fc = os.path.join(root_dir, filename)
        try:
            f = open(fc, 'r')
            info = f.readlines()
        except:
            continue
        for l in info:
            line = l.strip()
            line = line.encode('utf-8').decode('utf-8')
            if len(line)<min_char:
                continue
            if len(line) > max_char:
                start = np.random.randint(0,len(line)-max_char)
                line =line[start:start+np.random.randint(0,max_char)]
                new_line = ''
                for single in line:
                    if is_chinese(single):
                        new_line = new_line+single
                if is_good(new_line):
                    txt.append(new_line)
        f.close()
    return txt


if __name__=='__main__':
    root_dir = "/home/huyj/SRNet-master/SRNet-Datagen/Synthtext/data/newsgroup/"
    text = text_source(root_dir)
    print(len(text))
    with open('./result.txt','w') as f:
        for data in text:
            f.writelines(data+'\n')