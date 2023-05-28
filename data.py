import re


text = (
    'They serve the purpose of changing hydrogen into breathable oxygen,\n'
    'And they are as necessary here,\n'
    'As the air is on earth.\n'
    'But i still say, they are flowers.\n'
    'If you like!\n'
    'Do you sell them?\n'
    'I am afraid not.\n'
    'But maybe we can make a deal.'
)

sentences = re.sub(pattern="[,.!?\\-]",repl='',string=text).split('\n')
word_list = list(set(" ".join(sentences).split()))
word2idx = {'[PAD]':0,'[CLS]':1,'[SEP]':2,'[MASK]':3}
for i,w in enumerate(word_list):
    word2idx[w] = i+4
idx2word = {i:w for i,w in enumerate(word2idx)}
vocab_size = len(word2idx)

token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)
