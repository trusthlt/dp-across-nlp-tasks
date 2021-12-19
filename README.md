# One size does not fit all: Investigating strategies for differentially-private learning across NLP tasks

Companion code to our arXiv preprint.

Pre-print available at: https://arxiv.org/abs/2112.08159

Please use the following citation

```plain
@journal{Senge.et.al.2021.arXiv,
    title = {{One size does not fit all: Investigating strategies
              for differentially-private learning across NLP tasks}},
    author = {Senge, Manuel and Igamberdiev, Timour and Habernal, Ivan},
    journal = {arXiv preprint},
    year = {2021},
    url = {https://arxiv.org/abs/2112.08159},
}
```

> **Abstract** Preserving privacy in training modern NLP models comes at a cost. We know that stricter privacy guarantees in differentially-private stochastic gradient descent (DP-SGD) generally degrade model performance. However, previous research on the efficiency of DP-SGD in NLP is inconclusive or even counter-intuitive. In this short paper, we provide a thorough analysis of different privacy preserving strategies on seven downstream datasets in five different `typical' NLP tasks with varying complexity using modern neural models. We show that unlike standard non-private approaches to solving NLP tasks, where bigger is usually better, privacy-preserving strategies do not exhibit a winning pattern, and each task and privacy regime requires a special treatment to achieve adequate performance.

**Contact person**: Ivan Habernal, ivan.habernal@tu-darmstadt.de. https://www.trusthlt.org

*This repository contains experimental software and is published for the sole purpose of giving additional background details on the publication.*
