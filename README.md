# S2GNs-master
For the paper "Sampling Subgraph Network with Application to Graph Classification"
[Arxiv Link](https://arxiv.org/pdf/2102.05272.pdf)

Graphs are naturally used to describe the structures of various real-world systems in biology, society, computer science etc., where subgraphs or motifs as basic blocks play an important role in function expression and information processing. However, existing research focuses on the basic statistics of certain motifs, largely ignoring the connection patterns among them. Recently, a subgraph network (SGN) model is proposed to study the potential structure among motifs, and it was found that the integration of SGN can enhance a series of graph classification methods. However, SGN model lacks diversity and is of quite high time complexity, making it difficult to widely apply in practice. In this paper, we introduce sampling strategies into SGN, and design a novel sampling subgraph network model, which is scale-controllable and of higher diversity. We also present a hierarchical feature fusion framework to integrate the structural features of diverse sampling SGNs, so as to improve the performance of graph classification. Extensive experiments demonstrate that, by comparing with the SGN model, our new model indeed has much lower time complexity (reduced by two orders of magnitude) and can better enhance a series of graph classification methods (doubling the performance enhancement).

## S2GN framework
![S2GN](https://user-images.githubusercontent.com/26339035/125916743-023c834e-1842-4db5-be9d-2e7275704012.png)


## Cite

```
@article{wang2021sampling,
  title={Sampling Subgraph Network with Application to Graph Classification},
  author={Wang, Jinhuan and Chen, Pengtao and Ma, Bin and Zhou, Jiajun and Ruan, Zhongyuan and Chen, Guanrong and Xuan, Qi},
  journal={IEEE Transactions on Network Science and Engineering},
  year={2021}
}
```
