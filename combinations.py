from utils import *

if __name__ == '__main__':
    os.remove('good.txt.2.w')
    os.remove('bad.txt.2.w')
    read_and_write('good.txt.2')
    read_and_write('bad.txt.2')
