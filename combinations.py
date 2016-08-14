import itertools


def filter_line(line, tuple_length=3):
    elements = line.split('.')
    new_elements = []
    for elt in elements:
        new_element = []
        for split_elt in elt.split():
            split_elt = split_elt.strip()
            split_elt = split_elt.replace("'", '')
            if split_elt == 'de':
                continue
            new_element.append(split_elt)
        new_elements.append(new_element)
    new_str = ' '.join(['_'.join(v) for v in new_elements])
    return ' '.join(['_'.join(v) for v in set(itertools.permutations(new_str.split(), tuple_length))])


def read_and_write(filename):
    lines = []
    with open(filename, 'r') as r:
        lines.extend(r.readlines())
    with open(filename + '.w', 'a') as w:
        for line in lines:
            w.write(filter_line(line))
            w.write('\n')


if __name__ == '__main__':
    import os
    os.remove('good.txt.2.w')
    os.remove('bad.txt.2.w')
    read_and_write('good.txt.2')
    read_and_write('bad.txt.2')
