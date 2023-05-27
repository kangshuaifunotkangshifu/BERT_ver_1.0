# BERT Parameters
maxlen = 30  #每个batch中句子的最大长度，长度不够用PAD补充
batch_size = 2  #batch_size
max_pred = 5 # max tokens of prediction,每个句子中最多挖取的tokens的数量
n_layers = 6  # Encoder的层数
n_heads = 12  # 注意力头数
d_model = 768  # 输入模型的向量维度
d_ff = 768*4 # 4*d_model, FeedForward dimension,全连接层的宽度
d_k = d_v = 64  # dimension of K(=Q), V，多头注意力的特征维度
n_segments = 2  # 输入句子的最大数量
