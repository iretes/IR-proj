# Results Comparison

The table below compares the results from the original articles with those reproduced in this project. Differences may be attributed to the use of different datasets: the original results were based on the "sift" dataset [1], while the reproduced results used the "siftsmall" dataset [1] to reduce computational time. The smaller size of the "siftsmall" dataset required adjusting the values of the parameters $m, k'$ and $w$ to obtain similar results. Additionally, some experimental details were not specified in the original articles, and where specified, randomization (e.g., random subsampling) prevented exact replication of the results.

| Experiment Description | Results from the article | Reproduced Results | 
| ---------------------- | ------------------------ | ------------------ |
| Typical query of a SIFT vector in a set of 1000 vectors: comparison of the distance $d(x, y)$ obtained with the SDC and ADC estimators. We have used $m = 8$ and $k^* = 256$, i.e., 64-bit code vectors. | <img src=./img/fig3.png>  | <img src=./img/fig3_rep.png> |
| PDF of the error on the distance estimation (actual distance - estimated distance) for the asymmetric method in the original article and for both the symmetric and asymmetric methods in the reproduced results, evaluated on a set of 10000 SIFT vectors with $m=8$ and $k^*=256$. The bias of the estimator is corrected with the error quantization term $\epsilon(q, q(y))$. | <img src=./img/fig4.png> | <img src=./img/fig4_rep.png> |
| Quantization error associated with the parameters $m$ and $k^*$.| <img src=./img/fig1.png> | <img src=./img/fig1_rep.png> |
| ADC and SDC estimators evaluated on the SIFT data set: recall@100 as a function of the memory usage (code length $=m \times \log_2(k^*)$) for different parameters ($k^*=16, 64, 256, \ldots, 4096$ and $m=1,2,4,8,16$). The missing point in the original article results ($m=16,k^*=4096$) gives recall@100 $=1$ for both SDC and ADC. | <img src=./img/fig6b.png><img src=./img/fig6a.png> | <img src=./img/fig6_rep.png> |
| SIFT dataset: recall@100 for the IVFADC approach as a function of the memory usage for $k^*=256$ and varying values of $m, k' and w. |  <img src=./img/fig7.png> |  <img src=./img/fig7_rep.png> |
| Impact of the dimension grouping on the retrieval performance of ADC (recall@100 in the original article, recall@100 in the reproduced results, $k^*=256$, $m=4$) | Natural: $0.593$<br> Random: $0.501$<br> Structured: $0.640$| Natural: $0.720$<br> Random: $0.728$<br> Structured: $0.830$ |
| Search time in milliseconds and Recall@100 for different approaches. | SDC: Search time = $16.8$; Recall@100 = $0.446$<br> ADC: Search time = $17.2$; Recall@100 = $0.652$<br> IVFADC <br> $k'= 1024, w=1$: Search time = $1.5$; Recall@100 = $0.308$ <br> $k'= 1024, w=8$: Search time = $8.8$; Recall@100 = $0.682$ <br> $k'= 8192, w=8$: Search time = $10.2$; Recall@100 = $0.516$ <br> $k'= 8192, w=64$: Search time = $65.3$; Recall@100 = $0.610$ | SDC: Search time = $1.6$; Recall@100 = $1.000$ <br> ADC: Search time = $1.7$ ; Recall@100 = $1.000$ <br> IVFADC <br> $k'=16, w=1$: Search time = $0.2$; Recall@100 = $0.74$<br> $k'=16, w=2$: Search time = $0.49$; Recall@100 = $0.91$<br> $k'=128, w=2$: Search time = $0.4$; Recall@100 = $0.67$<br> $k'=128, w=8$: Search time = $0.96$; Recall@100 = $0.95$ |

### References
[1] http://corpus-texmex.irisa.fr



