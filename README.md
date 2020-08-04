## 用Bert进行文本分类

### Transformer简介

我们已经非常熟悉lstm的结构及其优势。Lstm可以在训练的过程中照顾到前文的信息，使用双向biListm更可以同时读取到上下文的全部内容。然而Lstm有两个明显的缺陷：

- 需要严格按照序列顺序进行前向传播，依赖性比较严重，难以实现并行化
- 虽然相较于普通RNN，Lstm已经能够处理梯度消失的问题，但面对很长的数据时，还会暴露问题。

Transformer就出现啦！它既能实现并行化，又能照顾到上下文的信息。下面我们简要介绍一下Transformer的结构。

#### Scaled Dot-Product Attention

为了依然可以获取上下文有关的信息，我们用attention来取代RNN做这样的工作。Transformer的attention结构如下所示：




$$
Attention(Q,K, V) = softmax(\frac {QK^T}{\sqrt{d_k}})
$$
