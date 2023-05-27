from data import *
from config import *
from datapro import *
from model import *


for epoch in range(180):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
      logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
      loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
      loss_lm = (loss_lm.float()).mean()
      loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
      loss = loss_lm + loss_clsf
      if (epoch + 1) % 10 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
