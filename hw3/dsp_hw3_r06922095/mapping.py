from collections import defaultdict
import argparse


def parse_args():
    parser = argparse.ArgumentParser('Generate ZhuYin-Big5.map from Big5-ZhuYin.map')
    parser.add_argument('input')
    parser.add_argument('output')
    return parser.parse_args()


def main(args):
    zy2big5 = defaultdict(set)
    with open(args.input, 'r', encoding='cp950') as file:
        for line in file:
            big5, zhuyin = line.split(' ')
            zhuyin = zhuyin.split('/')
            for zy in zhuyin:
                zy2big5[zy[0]].add(big5)
            
    with open(args.output, 'w', encoding='cp950') as file:
        for zhuyin in sorted(zy2big5.keys()):
            file.write(zhuyin)
            for big5 in zy2big5[zhuyin]:
                file.write(' {}'.format(big5))
            file.write('\n')

            for big5 in zy2big5[zhuyin]:
                file.write('{} {}\n'.format(big5, big5))
            

if __name__ == "__main__":
    args = parse_args()
    main(args)