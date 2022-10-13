Awesome-RSPapers
================

Included Conferences: SIGIR 2020, SIGKDD 2020, RecSys 2020, CIKM 2020,
AAAI 2021, WSDM 2021, WWW 2021, SIGIR 2021, RecSys 2021, KDD 2021,
SIGIR 2022, RecSys 2022, KDD 2022, AAAI 2022, CIKM 2021, CIKM 2022, 
WSDM 2022, WWW 2022, NIPS 2022, IJCAI 2022



-   [Task](#task)
    -   [Collaborative Filtering](#collaborative-filtering)
    -   [Sequential/Session-based
        Recommendations](#sequentialsession-based-recommendations)
    -   [CTR/CVR Prediction](#CTR/CVR-Prediction)
    -   [Knowledge-aware
        Recommendations](#knowledge-aware-recommendations)
    -   [Conversational Recommender
        System](#conversational-recommender-system)
    -   [Social Recommendation](#social-recommendation)
    -   [News Recommendation](#news-recommendation)
    -   [Music Recommendation](#music-recommendation)
    -   [Text-aware Recommendation](#text-aware-recommendation)
    -   [POI Recommendation](#poi-recommendation)
    -   [Online Recommendation](#online-recommendation)
    -   [Group Recommendation](#group-recommendation)
    -   [Multi-task/Multi-behavior/Cross-domain
        Recommendation](#multi-taskmulti-behaviorcross-domain-recommendation)
    -   [Multimodal
        Recommendations](#multimodal-recommendation)
    -   [Other Tasks](#other-tasks)
-   [Topic](#topic)
    -   [Debias in Recommender
        System](#debias-in-recommender-system)
    -   [Fairness in Recommender
        System](#fairness-in-recommender-system)
    -   [Attack in Recommender
        System](#attack-in-recommender-system)
    -   [Explanation in Recommender
        System](#explanation-in-recommender-system)
    -   [Long-tail/Cold-start in
        Recommendations](#long-tailcold-start-in-recommendations)
    -   [Diversity in Recommendation](#diversity-in-recommendation)
    -   [Denoising in Recommendation](#denoising-in-recommendation)
    -   [Privacy Protection in Recommendation](#privacy-protection-in-recommendation)
    -   [Evaluation of Recommender System](#evaluation-of-recommender-system)
    -   [Other Topics](#other-topics)
-   [Technique](#technique)
    -   [Pre-training in Recommender
        System](#pre-training-in-recommender-system)
    -   [Reinforcement Learning in
        Recommendation](#reinforcement-learning-in-recommendation)
    -   [Knowledge Distillation in
        Recommendation](#knowledge-distillation-in-recommendation)
    -   [Federated Learning in
        Recommendation](#federated-learning-in-recommendation)
    -   [GNN in Recommendation](#gnn-in-Recommendation)
    -   [Contrastive Learning based](#Contrastive-Learning-based)
    -   [Adversarial Learning based](#Adversarial-Learning-based)
    -   [Autoencoder based](#Autoencoder-based)
    -   [Meta Learning-based](#meta-learning-based)
    -   [AutoML-based](#AutoML-based)
    -   [Casual Inference/Counterfactual](#casual-inference/counterfactual)
    -   [Other Techniques](#other-techniques)

Task
----

### Collaborative Filtering

-   Neural Graph Collaborative Filtering. SIGIR 2019 【神经图协同过滤】
-   LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020 【推荐中的简化图神经网络】
-   Neighbor Interaction Aware Graph Convolution Networks for Recommendation. SIGIR 2020 【邻域交互感知图卷积网络】
-   A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks. KDD 2020 【贝叶斯图卷积神经网络】
-   Dual Channel Hypergraph Collaborative Filtering. KDD 2020 【双通道超图的协同过滤】
-   Probabilistic Metric Learning with Adaptive Margin for Top-K Recommendation. KDD 2020 【概率度量学习】
-   Content-Collaborative Disentanglement Representation Learning for Enhanced Recommendation. RecSys 2020 【协同内容的表示学习】
-   Neural Collaborative Filtering vs. Matrix Factorization Revisited. RecSys 2020 【深度协同过滤和矩阵分解的再思考】
-   Bilateral Variational Autoencoder for Collaborative Filtering. WSDM 2021 【用于协同过滤的双向变分自动编码器】
-   Learning User Representations with Hypercuboids for Recommender Systems. WSDM 2021 【超立方体学习】
-   Local Collaborative Autoencoders. WSDM 2021 【局部的协同自编码器】
-   A Scalable, Adaptive and Sound Nonconvex Regularizer for Low-rank Matrix Completion. WWW 2021 【低秩矩阵补全的正则化气】
-   HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering. WWW 2021 【双曲图卷积网络】
-   High-dimensional Sparse Embeddings for Collaborative Filtering. WWW 2021 【高维稀疏向量】
-   Collaborative Filtering with Preferences Inferred from Brain Signals. WWW 2021 【从脑信号中推断偏好】
-   Interest-aware Message-Passing GCN for Recommendation. WWW 2021 【兴趣感知的图卷积网络】
-   Neural Collaborative Reasoning. WWW 2021 【深度协同过滤的解释】
-   Sinkhorn Collaborative Filtering. WWW 2021 【Sinkhorn 协同过滤】
-   Bootstrapping User and Item Representations for One-Class Collaborative Filtering. SIGIR 2021 【单类协同过滤】
-   When and Whom to Collaborate with in a Changing Environment: A Collaborative Dynamic Bandit Solution. SIGIR 2021 【推荐中的动态 bandit】
-   Neural Graph Matching based Collaborative Filtering. SIGIR 2021 【基于神经图匹配的协同过滤】
-   Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization. SIGIR 2021 【通过互信息最大化进行图协同过滤】
-   SimpleX: A Simple and Strong Baseline for Collaborative Filtering. CIKM 2021【将Cosine Contrastive Loss引入协同过滤】
-   Incremental Graph Convolutional Network for Collaborative Filtering. CIKM 2021【增量图卷积神经网络用于协同过滤】
-   LT-OCF: Learnable-Time ODE-based Collaborative Filtering. CIKM 2021【Learnable-Time CF】
-   CausCF: Causal Collaborative Filtering for Recommendation Effect Estimation. CIKM 2021【applied paper，因果关系协同过滤用于推荐效果评估】
-   Vector-Quantized Autoencoder With Copula for Collaborative Filtering. CIKM 2021【short paper，用于协同过滤的矢量量化自动编码器】
-   Anchor-based Collaborative Filtering for Recommender Systems. CIKM 2021【short paper，Anchor-based推荐系统协同过滤】
-   XPL-CF: Explainable Embeddings for Feature-based Collaborative Filtering. CIKM 2021【short paper，可解释 Embedding 用于基于特征的协同过滤】
-   NMO: A Model-Agnostic and Scalable Module for Inductive Collaborative Filtering. SIGIR 2022 【模型无关的归纳式协同过滤模块】
-   Collaborative Filtering with Attribution Alignment for Review-based Non-overlapped Cross Domain Recommendation. WWW 2022 【通过属性对齐实现基于评论的跨域推荐】
-   Neuro-Symbolic Interpretable Collaborative Filtering for Attribute-based Recommendation. WWW 2022 【以模型为核心的神经符号可解释协同过滤】
-   Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW 2022 【通过邻居节点间的对比学习来改善图协同过滤】
-   Hypercomplex Graph Collaborative Filtering. WWW 2022 【超复图协同过滤】
-   Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW 2022 【通过邻居节点间的对比学习来改善图协同过滤】
-   On Sampling Collaborative Filtering Datasets. WSDM 2022【关于采样协同过滤数据集】
-   Profiling the Design Space for Graph Neural Networks based Collaborative Filtering. WSDM 2022【分析基于图神经网络的协同过滤的设计空间】
-   VAE++: Variational AutoEncoder for Heterogeneous One-Class Collaborative Filtering. WSDM 2022【异构单类协同过滤的变分自动编码器】
-   Geometric Inductive Matrix Completion:
    A Hyperbolic Approach with Unified Message Passing. WSDM 2022 【具有统一消息传递的双曲线方法】
-   Asymmetrical Context-aware Modulation for Collaborative Filtering Recommendation. CIKM 2022 【用于协同过滤推荐的非对称上下文感知调制】
-   Dynamic Causal Collaborative Filtering. CIKM 2022 【动态因果协同过滤】
-   Towards Representation Alignment and Uniformity in Collaborative Filtering. KDD 2022 【用户和物品表示对齐】
-   HICF: Hyperbolic Informative Collaborative Filtering. KDD 2022 【双曲空间增强表示】
-   Geometric Disentangled Collaborative Filtering. SIGIR 2022 【几何解耦的协同过滤】
-   Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering. SIGIR 2022 【数据去噪】
-   Self-Augmented Recommendation with Hypergraph Contrastive Collaborative Filtering. SIGIR 2022 【超图上的对比学习】
-   Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering. SIGIR 2022 【图协同过滤在准确度和新颖度上的表现】
-   Unify Local and Global Information for Top-N Recommendation. SIGIR 2022 【综合局部和全局信息】
-   Enhancing Top-N Item Recommendations by Peer Collaboration. SIGIR 2022 【short paper ，同龄人协同】
-   Evaluation of Herd Behavior Caused by Population-scale Concept Drift in Collaborative Filtering. SIGIR 2022 【short paper】
-   Matrix Factorization for Collaborative Filtering Is Just Solving an Adjoint Latent Dirichlet Allocation Model After All. RecSys 2021【矩阵分解式协同过滤的理论联系】
-   Negative Interactions for Improved Collaborative-Filtering: Don’t go Deeper, go Higher. RecSys 2021【线性全秩模型考虑高阶交互的扩展】
-   Privacy Preserving Collaborative Filtering by Distributed Mediation. RecSys 2021【隐私保护式协同过滤】
-   ProtoCF: Prototypical Collaborative Filtering for Few-shot Item Recommendation. RecSys 2021【少样本物品推荐的原型协同过滤】
-   Reenvisioning the comparison between Neural Collaborative Filtering and Matrix Factorization. RecSys 2021 【Reproducibility paper，神经协同过滤与矩阵分解的比较】
-   Investigating Overparameterization for Non-Negative Matrix Factorization in Collaborative Filtering. RecSys 2021【LBR，协同过滤中非负矩阵分解的过参数化研究】
-   ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations. RecSys 2022【基于原型的可解释协同过滤算法】
-   Revisiting the Performance of iALS on Item Recommendation Benchmarks. RecSys 2022【Reproducibility paper，隐式交替最小二乘法 iALS 基准表现的再思考】
-   Scalable Linear Shallow Autoencoder for Collaborative Filtering. RecSys 2022 【LBR，用于协同过滤的可扩展线性浅层自动编码器】
-   HCFRec: Hash Collaborative Filtering via Normalized Flow with Structural Consensus for Efficient Recommendation. IJCAI 2022 【为 hash-CF 学习最优哈希码】
-   Trading Hard Negatives and True Negatives: A Debiased Contrastive Collaborative Filtering Approach. IJCAI 2022 【探索正确的负样本】

### Sequential/Session-based Recommendations

-   Session-based recommendation with graph neural networks. AAAI 2019 【基于图神经网络的会话推荐】
-   Graph contextualized self-attention network for session-based recommendation. IJCAI 2019 【针对会话推荐的图上下文自注意力网络】
-   Session-based social recommendation via dynamic graph attention networks. WSDM 2019 【通过动态图注意力网络进行会话的社交推荐】
-   Sequential Recommendation with Self-attentive Multi-adversarial Network. SIGIR 2020 【自注意力机制的多对抗网络】
-   KERL: A Knowledge-Guided Reinforcement Learning Model for Sequential Recommendation. SIGIR 2020 【序列推荐中知识增强的强化学习模型】
-   Modeling Personalized Item Frequency Information for Next-basket Recommendation. SIGIR 2020 【下一篮推荐中建模个性化物品概率信息】
-   Incorporating User Micro-behaviors and Item Knowledge into Multi-task Learning for Session-based Recommendation. SIGIR 2020 【将用户行为和物品知识融入基于会话的推荐多任务学习】
-   GAG: Global Attributed Graph Neural Network for Streaming Session-based Recommendation. SIGIR 2020 【基于流会话推荐的全局属性图神经网络】
-   Next-item Recommendation with Sequential Hypergraphs. SIGIR 2020 【序列推荐中的超图】
-   A General Network Compression Framework for Sequential Recommender Systems. SIGIR 2020 【通用的网络压缩框架】
-   Make It a Chorus: Knowledge- and Time-aware Item Modeling for Sequential Recommendation. SIGIR 2020 【知识和时间感知的物品建模】
-   Global Context Enhanced Graph Neural Networks for Session-based Recommendation. SIGIR 2020 【上下文增强的图神经网络】
-   Self-Supervised Reinforcement Learning for Recommender Systems. SIGIR 2020 【自监督的强化学习】
-   Time Matters: Sequential Recommendation with Complex Temporal Information. SIGIR 2020 【结合时序信息的序列推荐】
-   Controllable Multi-Interest Framework for Recommendation. KDD 2020 【可控的多兴趣框架】
-   Disentangled Self-Supervision in Sequential Recommenders. KDD 2020 【序列推荐器中的自监督信号】
-   Handling Information Loss of Graph Neural Networks for Session-based Recommendation. KDD 2020 【基于会话推荐的图神经网络信息丢失处理】
-   Maximizing Cumulative User Engagement in Sequential Recommendation An Online Optimization Perspective. KDD 2020 【从在线优化的角度最大化序列推荐中的累积用户参与度】
-   Contextual and Sequential User Embeddings for Large-Scale Music Recommendation. RecSys 2020 【大规模音乐推荐的上下文和时序用户表示】
-   FISSA:Fusing Item Similarity Models with Self-Attention Networks for Sequential Recommendation. RecSys 2020 【融合物品相似性和自注意网络的序列推荐】
-   From the lab to production: A case study of session-based recommendations in the home-improvement domain. RecSys 2020 【家庭装修领域基于会话推荐的案例研究】
-   Recommending the Video to Watch Next: An Offline and Online Evaluation at YOUTV.de. RecSys 2020 【青年电视台的离线和在线视频评估】
-   SSE-PT:Sequential Recommendation Via Personalized Transformer. RecSys 2020 【基于个性化 Transformer 的序列推荐】
-   Improving End-to-End Sequential Recommendations with Intent-aware Diversification. CIKM 2020 【意向感知的多样性】
-   Quaternion-based self-Attentive Long Short-term User Preference Encoding for Recommendation. CIKM 2020 【基于四元数的自注意力长短期用户偏好】
-   Sequential Recommender via Time-aware Attentive Memory Network. CIKM 2020 【时间感知的注意力记忆网络】
-   Star Graph Neural Networks for Session-based Recommendation. CIKM 2020 【基于会话推荐的星形图神经网络】
-   S$^3$-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. CIKM 2020 【基于互信息最大化的序列推荐自监督学习】
-   Dynamic Memory Based Attention Network for Sequential Recommendation. AAAI 2021 【序列推荐中基于动态记忆的注意力网络】
-   Noninvasive Self-Attention for Side Information Fusion in Sequential Recommendation. AAAI 2021 【序列推荐中融合边信息的自注意力】
-   Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation. AAAI 2021 【自监督超图卷积网络】
-   An Efficient and Effective Framework for Session-based Social Recommendation. WSDM 2021 【基于会话的社交推荐】
-   Sparse-Interest Network for Sequential Recommendation. WSDM 2021 【序列推荐的稀疏兴趣网络】
-   Dynamic Embeddings for Interaction Prediction. WWW 2021 【交互预测的动态向量】
-   Session-aware Linear Item-Item Models for Session-based Recommendation. WWW 2021 【线性的物品-物品模型】
-   RetaGNN: Relational Temporal Attentive Graph Neural Networks for Holistic Sequential Recommendation. WWW 2021 【关系时间注意的图神经网络】
-   Adversarial and Contrastive Variational Autoencoder for Sequential Recommendation. WWW 2021 【对抗和对比式变分自动编码器】
-   Future-Aware Diverse Trends Framework for Recommendation. WWW 2021 【面向未来的多元化趋势推荐框架】
-   DeepRec: On-device Deep Learning for Privacy-Preserving Sequential Recommendation in Mobile Commerce. WWW 2021 【移动商务的隐私保护】
-   Linear-Time Self Attention with Codeword Histogram for Efficient Recommendation. WWW 2021 【通过 codeword 直方图进行线性时间的自注意力】
-   Category-aware Collaborative Sequential Recommendation. SIGIR 2021 【类别感知的协同序列推荐】
-   Sequential Recommendation with Graph Convolutional Networks. SIGIR 2021 【引入图卷积网络的序列推荐】
-   Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer. SIGIR 2021 【反向预训练 Transformer 增强序列推荐】
-   Dual Attention Transfer in Session-based Recommendation with Multi Dimensional Integration. SIGIR 2021 【多维集成会话推荐中的双重注意转移】
-   Unsupervised Proxy Selection for Session-based Recommender Systems. SIGIR 2021 【无监督代理选择】
-   StackRec: Efficient Training of Very Deep Sequential Recommender Models by Iterative Stacking. SIGIR 2021 【基于迭代叠加的序列推荐模型】
-   Counterfactual Data-Augmented Sequential Recommendation. SIGIR 2021 【反事实数据增强】
-   CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation. SIGIR 2021 【反事实用户序列合成】
-   The World is Binary: Contrastive Learning for Denoising Next Basket Recommendation. SIGIR 2021 【通过对比学习对下一篮推荐问题去噪】
-   Motif-aware Sequential Recommendation. SIGIR 2021 【模式感知的序列推荐】
-   Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation. SIGIR 2021 【short paper，低阶分解的自注意网络】
-   Seq2Bubbles: Region-Based Embedding Learning for User Behaviors in Sequential Recommenders. CIKM 2021【序列推荐中基于区域的用户行为Embedding学习】
-   Enhancing User Interest Modeling with Knowledge-Enriched Itemsets for Sequential Recommendation. CIKM 2021【序列推荐中使用物品集增强用户兴趣建模】
-   Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer. CIKM 2021【将时序图协同Transformer用于连续时间序列推荐】
-   Extracting Attentive Social Temporal Excitation for Sequential Recommendation. CIKM 2021【提取时序激励用于序列推荐】
-   Hyperbolic Hypergraphs for Sequential Recommendation. CIKM 2021【使用双曲超图进行序列推荐】
-   Learning Dual Dynamic Representations on Time-Sliced User-Item Interaction Graphs for Sequential Recommendation. CIKM 2021【用于序列推荐的在时间片用户物品交互图上的对偶动态表示】
-   Lightweight Self-Attentive Sequential Recommendation. CIKM 2021【使用CNN捕获局部特征，使用Self-Attention捕获全局特征】
-   What is Next when Sequential Prediction Meets Implicitly Hard Interaction? CIKM 2021【序列预测与交互】
-   Modeling Sequences as Distributions with Uncertainty for Sequential Recommendation. CIKM 2021【short paper，序列建模】
-   Locker: Locally Constrained Self-Attentive Sequential Recommendation. CIKM 2021【short paper，局部约束的自注意力序列推荐】
-   CBML: A Cluster-based Meta-learning Model for Session-based Recommendation. CIKM 2021【用于会话推荐的基于聚类的元学习】
-   Self-Supervised Graph Co-Training for Session-based Recommendation. CIKM 2021【用于会话推荐的自监督图协同训练】
-   S-Walk: Accurate and Scalable Session-based Recommendation with Random Walks. WSDM 2022【具有随机游走的准确且可扩展的基于会话的推荐】
-   Learning Multi-granularity Consecutive User Intent Unit for Session-based Recommendation. WSDM 2022【基于会话的推荐学习多粒度连续用户意图单元】
-   A Relevant and Diverse Retrieval-enhanced Data Augmentation Framework for Sequential Recommendation. CIKM 2022 【Applied Research Track, 用于顺序推荐的相关且多样化的检索增强数据增强框架】
-   Beyond Learning from Next Item: Sequential Recommendation via Personalized Interest Sustainability. CIKM 2022 【基于个性化兴趣可持续性的序列推荐】
-   Storage-saving Transformer for Sequential Recommendations. CIKM 2022 【用于序列推荐的节省存储的Transformer】
-   Contrastive Learning with Bidirectional Transformers for Sequential Recommendation. CIKM 2022 【用于序列推荐的双向 Transformer 对比学习】
-   FwSeqBlock: A Field-wise Approach for Modeling Behavior Representation in Sequential Recommendation. CIKM 2022 【建模行为表示】
-   Dually Enhanced Propensity Score Estimation in Sequential Recommendation. CIKM 2022 【双重增强倾向得分估计】
-   Hierarchical Item Inconsistency Signal learning for Sequence Denoising in Sequential Recommendation. CIKM 2022 【序列推荐中序列去噪的分层项目不一致信号学习】
-   ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation. CIKM 2022 【用于序列推荐的对比变分自动编码器】
-   Disentangling Past-Future Modeling in Sequential Recommendation via Dual Networks. CIKM 2022 【通过双网络解耦序列推荐中的过去未来】
-   Evolutionary Preference Learning via Graph Nested GRU ODE for Session-based Recommendation. CIKM 2022 【通过图嵌套 GRU ODE 进行进化偏好学习】
-   Spatiotemporal-aware Session-based Recommendation with Graph Neural Networks. CIKM 2022 【使用图神经网络的时空感知基于会话的推荐】
-   Time Lag Aware Sequential Recommendation. CIKM 2022 【延时感知序列推荐】
-   Temporal Contrastive Pre-Training for Sequential Recommendation. CIKM 2022 【时序推荐的时间对比预训练】
-   Debiasing the Cloze Task in Sequential Recommendation with Bidirectional Transformers. KDD 2022 【序列完形填空去偏】
-   PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions. KDD 2022 【用户个性化区分】
-   Multi-Behavior Hypergraph-Enhanced Transformer for Sequential Recommendation. KDD 2022 【多行为超图增强】
-   NxtPost: User To Post Recommendations In Facebook Groups. KDD 2022 【引入外部token扩充预料】
-   Recommendation in Offline Stores: A Gamification Approach for Learning the Spatiotemporal Representation of Indoor Shopping. KDD 2022 【游戏化方法时空表征】
-   DiPS: Differentiable Policy for Sketching in Recommender Systems. AAAI 2022 【可微分的草图策略】
-   FPAdaMetric: False-Positive-Aware Adaptive Metric Learning for Session-Based Recommendation. AAAI 2022 【度量学习正则化】
-   SEMI: A Sequential Multi-Modal Information Transfer Network for E-Commerce Micro-Video Recommendations. KDD 2021 【基于序列多模态信息传输网络的电商微视频推荐系统】
-   Decoupled Side Information Fusion for Sequential Recommendation. SIGIR 2022 【融合边缘特征的序列推荐】
-   On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation. SIGIR 2022 【自监督知识蒸馏】
-   Multi-Agent RL-based Information Selection Model for Sequential Recommendation. SIGIR 2022 【多智能体信息选择】
-   An Attribute-Driven Mirroring Graph Network for Session-based Recommendation. SIGIR 2022 【特征驱动的反射图网络】
-   When Multi-Level Meets Multi-Interest: A Multi-Grained Neural Model for Sequential Recommendation. SIGIR 2022 【多粒度网络】
-   Price DOES Matter! Modeling Price and Interest Preferences in Session-based Recommendation. SIGIR 2022 【考虑价格和兴趣的推荐】
-   AutoGSR: Neural Architecture Search for Graph-based Session Recommendation. SIGIR 2022 【面向图会话推荐的网络结构搜索】
-   Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for Sequential Recommendation. SIGIR 2022 【数据分布自适应排序】
-   Multi-Faceted Global Item Relation Learning for Session-Based Recommendation. SIGIR 2022 【多面全局商品关系学习】
-   ReCANet: A Repeat Consumption-Aware Neural Network for Next Basket Recommendation in Grocery Shopping. SIGIR 2022 【考虑重复消费的网络】
-   Determinantal Point Process Set Likelihood-Based Loss Functions for Sequential Recommendation. SIGIR 2022 【基于DPP的损失函数】
-   Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation. SIGIR 2022 【建模隐式反馈】
-   Coarse-to-Fine Sparse Sequential Recommendation. SIGIR 2022 【short paper，粗到细的稀疏序列化推荐】
-   Dual Contrastive Network for Sequential Recommendation. SIGIR 2022 【short paper，双对比网络】
-   Explainable Session-based Recommendation with Meta-Path Guided Instances and Self-Attention Mechanism. SIGIR 2022 【short paper， 基于元路径指导和自注意力机制的可解释会话推荐】 
-   Item-Provider Co-learning for Sequential Recommendation. SIGIR 2022 【short paper，商品-商家一同训练】
-   RESETBERT4Rec: A Pre-training Model Integrating Time And User Historical Behavior for Sequential Recommendation. SIGIR 2022 【short paper，融合时间和用户历史行为的预训练模型】
-   Enhancing Hypergraph Neural Networks with Intent Disentanglement for Session-based Recommendation. SIGIR 2022 【short paper，意图解耦增强超图神经网络】
-   CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space. SIGIR 2022 【short paper，在一致表示空间上的简单有效会话推荐】
-   DAGNN: Demand-aware Graph Neural Networks for Session-based Recommendation. SIGIR 2022 【short paper， 需求感知的图神经网络】
-   Progressive Self-Attention Network with Unsymmetrical Positional Encoding for Sequential Recommendation. SIGIR 2022 【short paper，使用非对称位置编码的自注意力网络】
-   ELECRec: Training Sequential Recommenders as Discriminators. SIGIR 2022 【short paper，训练序列推荐模型作为判别器】
-   Exploiting Session Information in BERT-based Session-aware Sequential Recommendation. SIGIR 2022 【short paper，在基于BERT的模型中利用会话信息】
-   Black-Box Attacks on Sequential Recommenders via Data-Free Model Extraction. RecSys 2021 【序列推荐的黑盒攻击】
-   Next-item Recommendations in Short Sessions. RecSys 2021 【短会话中的下一个物品推荐】
-   Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation. RecSys 2021 【建立自然语言处理和序列推荐的联系】
-   A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models. RecSys 2021 【Reproducibility paper，序列推荐模型评测策略的研究】
-   Sequence Adaptation via Reinforcement Learning in Recommender Systems. RecSys 2021 【LBR，用强化学习方法进行序列推荐】
-   Aspect Re-distribution for Learning Better Item Embeddings in Sequential Recommendation. RecSys 2022 【序列推荐中物品向量的方面重分布】
-   Context and Attribute-Aware Sequential Recommendation via Cross-Attention. RecSys 2022 【交叉注意力机制实现上下文和属性感知的序列推荐】
-   Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders. RecSys 2022 【序列推荐中基于替代的对抗性攻击算法】
-   Denoising Self-Attentive Sequential Recommendation. RecSys 2022 【基于 Transformer 的序列推荐中自注意力机制的去噪】
-   Don't recommend the obvious: estimate probability ratios. RecSys 2022 【通过拟合逐点的互信息来改进序列推荐的流行度采样指标】
-   Effective and Efficient Training for Sequential Recommendation using Recency Sampling. RecSys 2022 【基于接近度采样进行高效的序列推荐】
-   Global and Personalized Graphs for Heterogeneous Sequential Recommendation by Learning Behavior Transitions and User Intentions. RecSys 2022 【异构序列推荐中通过全局和个性化的图建模学习行为转换和用户意图】
-   Learning Recommendations from User Actions in the Item-poor Insurance Domain. RecSys 2022 【保险领域使用循环神经网络的跨会话模型】
-   A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation. RecSys 2022【Reproducibility paper，BERT4Rec 结果的系统回顾与可复现性研究】
-   Streaming Session-Based Recommendation: When Graph Neural Networks meet the Neighborhood. RecSys 2022【Reproducibility paper，图神经网络解决流会话推荐问题】
-   Learning to Ride a Buy-Cycle: A Hyper-Convolutional Model for Next Basket Repurchase Recommendation. RecSys 2022 【针对下一篮回购推荐问题的超卷积模型】
-   M2TRec: Metadata-aware Multi-task Transformer for Large-scale and Cold-start free Session-based Recommendations. RecSys 2022【LBR，基于元数据和多任务 Transformer 的冷启动会话推荐系统】
-   MLP4Rec: A Pure MLP Architecture for Sequential Recommendations. IJCAI 2022 【利用MLP捕捉商品特征中的序列关系】
-   Enhancing Sequential Recommendation with Graph Contrastive Learning. IJCAI 2022 【用于序列推荐的图对比学习】
-   Disentangling Long and Short-Term Interests for Recommendation. WWW 2022 【利用自监督对比学习解耦长短期兴趣】
-   Efficient Online Learning to Rank for Sequential Music Recommendation. WWW 2022 【将搜索空间限制在此前表现不佳的搜索方向的正交补上】
-   Filter-enhanced MLP is All You Need for Sequential Recommendation. www 2022 【通过可学习滤波器对用户序列进行编码】
-   Generative Session-based Recommendation. WWW 2022 【构建生成器来模拟用户序列行为，从而改善目标序列推荐模型】
-   GSL4Rec: Session-based Recommendations with Collective Graph Structure Learning and Next Interaction Prediction. WWW 2022 【图结构学习+推荐】
-   Intent Contrastive Learning for Sequential Recommendation. WWW 2022 【利用用户意图来增强序列推荐】
-   Learn from Past, Evolve for Future: Search-based Time-aware Recommendation with Sequential Behavior Data. WWW 2022 【检索相关历史行为并融合当前序列行为】
-   Sequential Recommendation via Stochastic Self-Attention. WWW 2022 【通过随机高斯分布和Wasserstein自注意力模块来引入不确定性】
-   Sequential Recommendation with Decomposed Item Feature Routing. WWW 2022 【解耦item特征，并分别利用软硬模型来路由最有特征序列】
-   Towards Automatic Discovering of Deep Hybrid Network Architecture for Sequential Recommendation. WWW 2022 【通过NAS来搜索每一层使用注意力/卷积模块】
-   Unbiased Sequential Recommendation with Latent Confounders. WWW 2022 【去除潜在混淆变量来实现无偏序列推荐】
-   Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation. WWW 2022 【通过re-contrast,re-attend,re-construct来增强解耦用户多兴趣表示】

### CTR/CVR Prediction

- Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction. WWW 2019 【CTR 预测中基于图神经网络的特征交互建模】
- Unbiased Ad Click Prediction for Position-aware Advertising Systems. RecSys 2020 【无偏的广告点击预测】
- Cross-Positional Attention for Debiasing Clicks. WWW 2021 【跨位置的注意力机制用于点击去偏】
- Enhanced Doubly Robust Learning for Debiasing Post-Click Conversion Rate Estimation. SIGIR 2021 【点击后转换率估计】
- Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction. SIGIR 2021 【点击率预测中冷启动广告】
- RLNF: Reinforcement Learning based Noise Filtering for Click-Through Rate Prediction. SIGIR 2021 【基于强化学习的点击率预测噪声滤波】
- Detecting Beneficial Feature Interactions for Recommender Systems. AAAI 2021 【推荐系统的特征交互】
- DeepLight: Deep Lightweight Feature Interactions for Accelerating CTR Predictions in Ad Serving. WSDM 2021 【广告服务的轻量级特征交互】

- Multi-Interactive Attention Network for Fine-grained Feature Learning in CTR Prediction. WSDM 2021 【CTR 预测中细粒度特征学习的多交互注意力网络】

- FM$^2$: Field-matrixed Factorization Machines for CTR Prediction. WWW 2021 【段矩阵分解机】
- Multi-task Learning for Bias-Free Joint CTR Prediction and Market Price Modeling in Online Advertising. CIKM 2021【在线广告无偏差联合CTR预估和市场价格建模的多任务学习】
- Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models. CIKM 2021【applied paper，用于并行 CTR 的显式和隐式特征交互增强】
- TSI: An Ad Text Strength Indicator using Text-to-CTR and Semantic-Ad-Similarity. CIKM 2021【applied paper，使用 Text-to-CTR 和 Semantic-Ad-Similarity 的广告文本强度指标】
- One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction. CIKM 2021【applied paper，用于多领域CTR预估的自适应推荐】
- Efficient Learning to Learn a Robust CTR Model for Web-scale Online Sponsored Search Advertising. CIKM 2021【applied paper，用于在线搜索广告的CTR模型】
- AutoIAS: Automatic Integrated Architecture Searcher for Click-Trough Rate Prediction. CIKM 2021【CTR预估的自动集成搜索架构】
- Click-Through Rate Prediction with Multi-Modal Hypergraphs. CIKM 2021【使用多模态超图的点击率预测】
- Open Benchmarking for Click-Through Rate Prediction. CIKM 2021【开源CTR预估Benchmark】
- Disentangled Self-Attentive Neural Networks for Click-Through Rate Prediction. CIKM 2021【short paper，用于CTR预估的自注意力网络】
- AutoHERI: Automated Hierarchical Representation Integration for Post-Click Conversion Rate Estimation. CIKM 2021【short paper，用于点击后转换率估计的分层表示学习】
- Sequential Modeling with Multiple Attributes for Watchlist Recommendation in E-Commerce. WSDM 2022【电子商务中观察列表推荐的多属性序列建模】
- Modeling Users’ Contextualized Page-wise Feedback for Click-Through Rate Prediction in E-commerce Search. WSDM 2022【电子商务搜索中点击率预测的用户情境化页面反馈建模】
- Learning-To-Ensemble by Contextual Rank Aggregation in E-Commerce. WSDM 2022【通过电子商务中的上下文排名聚合学习集成】
- CAN: Feature Co-Action Network for Click-Through Rate Prediction. WSDM 2022【用于点击率预测的特征协同网络】
- Triangle Graph Interest Network for Click-through Rate Prediction. WSDM 2022【用于点击率预测的三角图兴趣网络】
- OptEmbed: Learning Optimal Embedding Table for Click-through Rate Prediction. CIKM 2022 【点击率预测的最优嵌入表】
- Multi-Interest Refinement by Collaborative Attributes Modeling for Click-Through Rate Prediction. CIKM 2022 【通过协作属性建模进行多兴趣细化的点击率预测】
- GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction. CIKM 2022 【GIFT：用于冷启动视频点击率预测的图引导特征迁移】
- Graph Based Long-Term And Short-Term Interest Model for Click-Through Rate Prediction. CIKM 2022 【用于点击率预测的基于图的长期和短期兴趣模型】
- Hierarchically Fusing Long and Short-Term User Interests for Click-Through Rate Prediction in Product Search. CIKM 2022 【分层融合长期和短期用户兴趣以进行产品搜索中的点击率预测】
- Sparse Attentive Memory Network for Click-through Rate Prediction with Long Sequences. CIKM 2022 【用于长序列点击率预测的稀疏注意力记忆网络】
- Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Models. CIKM 2022 【了解深度点击率模型的过拟合现象】
- User-Event Graph Embedding Learning for Context-Aware Recommendation. KDD 2022 【CTR 任务引入用户-事件图进行表示学习】
- Adversarial Gradient Driven Exploration for Deep Click-Through Rate Prediction. KDD 2022 【CTR EE 探索】
- Combo-Fashion: Fashion Clothes Matching CTR Prediction with Item History. KDD 2022 【流行服饰匹配增强 CTR】
- Enhancing CTR Prediction with Context-Aware Feature Representation Learning. SIGIR 2022 【上下文相关的特征表示】
- HIEN: Hierarchical Intention Embedding Network for Click-Through Rate Prediction. SIGIR 2022 【层次化意图嵌入网络】
- NAS-CTR: Efficient Neural Architecture Search for Click-Through Rate Prediction. SIGIR 2022 【高效的网络结构搜索】
- NMO: A Model-Agnostic and Scalable Module for Inductive Collaborative Filtering. SIGIR 2022 【模型无关的归纳式协同过滤模块】
- Neighbour Interaction based Click-Through Rate Prediction via Graph-masked Transformer. SIGIR 2022 【图遮盖的Transformer】
- Neural Statistics for Click-Through Rate Prediction. SIGIR 2022 【short paper，神经统计学】
- Smooth-AUC: Smoothing the Path Towards Rank-based CTR Prediction. SIGIR 2022 【short paper，基于排序的CTR预估】
- DisenCTR: Dynamic Graph-based Disentangled Representation for Click-Through Rate Prediction. SIGIR 2022 【基于图的解耦表示】
- Deep Multi-Representational Item Network for CTR Prediction. SIGIR 2022 【short paper，多重表示商品网络】
- Gating-adapted Wavelet Multiresolution Analysis for Exposure Sequence Modeling in CTR prediction. SIGIR 2022 【short paper，多分辨率小波分析】
- MetaCVR: Conversion Rate Prediction via Meta Learning in Small-Scale Recommendation Scenarios. SIGIR 2022 【short paper，小规模推荐场景下的元学习】
- Adversarial Filtering Modeling on Long-term User Behavior Sequences for Click-Through Rate Prediction. SIGIR 2022 【short paper，对抗过滤建模用户长期行为序列】
- Clustering based Behavior Sampling with Long Sequential Data for CTR Prediction. SIGIR 2022 【short paper，长序列数据集基于聚类的行为采样】
- CTnoCVR: A Novelty Auxiliary Task Making the Lower-CTR-Higher-CVR Upper. SIGIR 2022 【short paper，新颖度辅助任务】
- Large-Scale Modeling of Mobile User Click Behaviors Using Deep Learning. RecSys 2021 【移动设备的用户的点击行为预测】
- Page-level Optimization of e-Commerce Item Recommendations. RecSys 2021 【商品详细信息页面的商品推荐模块】
- An Analysis Of Entire Space Multi-Task Models For Post-Click Conversion Prediction. RecSys 2021 【LBR，点击后转换率预测的全空间多任务模型分析】
- CAEN: A Hierarchically Attentive Evolution Network for Item-Attribute-Change-Aware Recommendation in the Growing E-commerce Environment. RecSys 2022 【物品属性变化感知的分层注意力动态网络】
- Dual Attentional Higher Order Factorization Machines. RecSys 2022 【双注意高阶因式分解机】
- Position Awareness Modeling with Knowledge Distillation for CTR Prediction. RecSys 2022 【LBR，位置感知的知识提取框架】
- APG: Adaptive Parameter Generation Network for Click-Through Rate Prediction. NIPS 2022 【点击率预测的自适应参数生成网络】
- Alleviating Cold-start Problem in CTR Prediction with A Variational Embedding Learning Framework. WWW 2022 【使用变分embedding学习框架缓解 CTR 预测中的冷启动问题】
- CBR: Context Bias aware Recommendation for Debiasing User Modeling and Click Prediction. WWW 2022 【去除由丰富交互造成的上下文偏差】

### Knowledge-aware Recommendations

- Knowledge Graph Convolutional Networks for Recommender Systems. WWW 2019 【推荐系统的知识图谱卷积网络】
- KGAT: Knowledge Graph Attention Network for Recommendation. KDD 2019 【推荐的知识图谱注意力机制网络】
- CKAN: Collaborative Knowledge-aware Attentive Network for Recommender Systems. SIGIR 2020 【推荐系统的协同知识感知注意力网络】
- Attentional Graph Convolutional Networks for Knowledge Concept Recommendation in MOOCs in a Heterogeneous View. SIGIR 2020 【MOOC 知识概念推荐的注意力图卷积网络】

- MVIN: Learning multiview items for recommendation. SIGIR 2020 【学习多视角的物品表示】

- Jointly Non-Sampling Learning for Knowledge Graph Enhanced Recommendation. SIGIR 2020 【知识图增强推荐的非采样学习】

- Joint Item Recommendation and Attribute Inference: An Adaptive Graph Convolutional Network Approach. SIGIR 2020 【结合物品推荐和属性推理的自适应图卷积网络】

- Leveraging Demonstrations for Reinforcement Recommendation Reasoning over Knowledge Graphs. SIGIR 2020 【利用演示增强知识图上的推荐推理】

- Interactive Recommender System via Knowledge Graph-enhanced Reinforcement Learning. SIGIR 2020 【基于知识图增强强化学习的交互式推荐系统】

- Fairness-Aware Explainable Recommendation over Knowledge Graphs. SIGIR 2020 【知识图谱上公平性感知的可解释推荐】

- SimClusters Community-Based Representations for Heterogenous Recommendations at Twitter. KDD 2020 【推特上基于相似社区的表示】

- Multi-modal Knowledge Graphs for Recommender Systems. CIKM 2020 【推荐系统的多模态知识图】

- DisenHAN Disentangled Heterogeneous Graph Attention Network for Recommendation. CIKM 2020 【异构图注意网络】

- Genetic Meta-Structure Search for Recommendation on Heterogeneous Information Network. CIKM 2020 【异构信息网络推荐的遗传元结构搜索】

- TGCN Tag Graph Convolutional Network for Tag-Aware Recommendation. CIKM 2020 【标签图卷积网络】

- Knowledge-Enhanced Top-K Recommendation in Poincaré Ball. AAAI 2021 【双曲空间中知识增强的推荐】

- Graph Heterogeneous Multi-Relational Recommendation. AAAI 2021 【图异构的多关系推荐】

- Knowledge-Enhanced Hierarchical Graph Transformer Network for Multi-Behavior Recommendation. AAAI 2021 【用于多行为推荐的知识增强层次图 Transformer 网络】

- Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph. WSDM 2021 【知识图上的伪标记缓解冷启动问题】

- Decomposed Collaborative Filtering Modeling Explicit and Implicit Factors For Recommender Systems. WSDM 2021 【分解协同过滤模型的显式和隐式因素】

- Temporal Meta-path Guided Explainable Recommendation. WSDM 2021 【时态元路径引导的可解释推荐】

- Learning Intents behind Interactions with Knowledge Graph for Recommendation. WWW 2021 【知识增强推荐的意图学习】
- A Knowledge-Aware Recommender with Attention-Enhanced Dynamic Convolutional Network. CIKM 2021【动态卷积用于知识感知的推荐】
- Entity-aware Collaborative Relation Network with Knowledge Graph for Recommendation. CIKM 2021【short paper，KG+RS】
- Conditional Graph Attention Networks for Distilling and Refining Knowledge Graphs in Recommendation. CIKM 2021【GNN+KG+RS】
- Knowledge Enhanced Multi-Interest Network for the Generation of Recommendation Candidates. CIKM 2022 【用于生成推荐候选的知识增强多兴趣网络】
- Tiger: Transferable Interest Graph Embedding for Domain-Level Zero-Shot Recommendation. CIKM 2022 【用于域级零样本推荐的可迁移兴趣图嵌入】
- Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning. CIKM 2022 【通过多层次交互式对比学习改进知识感知推荐】
- Modeling Scale-free Graphs with Hyperbolic Geometry for Knowledge-aware Recommendation. WSDM 2022【使用双曲几何建模无标度图以进行知识感知推荐】
- Knowledge Graph Contrastive Learning for Recommendation. SIGIR 2022 【知识图谱上的对比学习】
- Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System. SIGIR 2022 【多级交叉视图的对比学习】
- Alleviating Spurious Correlations in Knowledge-aware Recommendations through Counterfactual Generator. SIGIR 2022 【利用反事实生成器缓解假知识】
- HAKG: Hierarchy-Aware Knowledge Gated Network for Recommendation. SIGIR 2022 【层次化知识门控网络】
- KETCH: Knowledge Graph Enhanced Thread Recommendation in Healthcare Forums. SIGIR 2022 【医疗论坛上的知识图谱增强的推荐】
- Sparse Feature Factorization for Recommender Systems with Knowledge Graphs. RecSys 2021 【利用知识图谱分解推荐系统稀疏特征】
- TinyKG: Memory-Efficient Training Framework for Knowledge Graph Neural Recommender Systems. RecSys 2022 【知识图神经推荐系统的内存高效训练框架】
- Knowledge-aware Recommendations Based on Neuro-Symbolic Graph Embeddings and First-Order Logical Rules. RecSys 2022 【LBR，基于神经符号图表示的知识感知推荐框架】
- Multi-level Recommendation Reasoning over Knowledge Graphs with Reinforcement Learning. WWW 2022 【基于强化学习的知识图多级推荐推理】
- Path Language Modeling over Knowledge Graphs for Explainable Recommendation. WWW 2022 【在知识图谱上学习语言模型，实现推荐和解释】

### Conversational Recommender System

-   Towards Question-based Recommender Systems. SIGIR 2020 【基于问题的推荐系统】
-   Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion. KDD 2020 【知识图谱的语义融合增强对话推荐系统】

-   Interactive Path Reasoning on Graph for Conversational Recommendation. KDD 2020 【对话推荐中基于图路径的解释】

-   A Ranking Optimization Approach to Latent Linear Critiquing for Conversational Recommender Systems. RecSys 2020 【会话推荐系统线性评判的排序优化方法】

-   What does BERT know about books, movies and music: Probing BERT for Conversational Recommendation. RecSys 2020 【BERT 的对话推荐】

-   Adapting User Preference to Online Feedback in Multi-round Conversational Recommendation. WSDM 2021 【多轮会话推荐中调整用户偏好】

-   A Workflow Analysis of Context-driven Conversational Recommendation. WWW 2021 【上下文驱动会话推荐的工作流分析】

-   Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning. SIGIR 2021 【基于图强化学习的对话推荐】

-   Learning to Ask Appropriate Questions in Conversational Recommendation. SIGIR 2021 【对话推荐中学习问恰当的问题】

-   Comparison-based Conversational Recommender System with Relative Bandit Feedback. SIGIR 2021 【基于相对 bandit 反馈的对话推荐系统】
-   Popcorn: Human-in-the-loop Popularity Debiasing in Conversational Recommender Systems. CIKM 2021【采用人在回路方式进行对话推荐系统的流行度去偏】
-   A Neural Conversation Generation Model via Equivalent Shared Memory Investigation. CIKM 2021【对话生成】
-   C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System. WSDM 2022【对话式推荐系统的粗到细对比学习】
-   Rethinking Conversational Recommendations: Is Decision Tree All You Need? CIKM 2022 【重新思考对话推荐：决策树是否就是我们需要的？】
-   Two-level Graph Path Reasoning for Conversational Recommendation with User Realistic Preference. CIKM 2022 【具有用户现实偏好的会话推荐的两级图路径推理】
-   Extracting Relevant Information from User's Utterances in Conversational Search and Recommendation. KDD 2022 【用户话语权】
-   Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning. KDD 2022 【知识增强的提示学习】
-   Learning to Infer User Implicit Preference in Conversational Recommendation. SIGIR 2022 【学习推测用户隐偏好】
-   User-Centric Conversational Recommendation with Multi-Aspect User Modeling. SIGIR 2022 【多角度用户建模】
-   Variational Reasoning about User Preferences for Conversational Recommendation. SIGIR 2022 【用户偏好的变分推理】
-   Analyzing and Simulating User Utterance Reformulation in Conversational Recommender Systems. SIGIR 2022 【对话推荐中模仿用户言论】
-   Improving Conversational Recommender Systems via Transformer-based Sequential ModellingSIGIR 2022 【short paper，基于Transformer的序列化建模】
-   Conversational Recommendation via Hierarchical Information Modeling. SIGIR 2022 【short paper，层次化信息建模】
-   Large-scale Interactive Conversational Recommendation System. RecSys 2021 【大规模交互式对话推荐系统】
-   Partially Observable Reinforcement Learning for Dialog-based Interactive Recommendation. RecSys 2021 【将强化学习用于基于对话框的交互式推荐任务】
-   Generation-based vs. Retrieval-based Conversational Recommendation: A User-Centric Comparison. RecSys 2021 【Reproducibility paper，对话推荐中基于生成和基于检索的方法对比】
-   Soliciting User Preferences in Conversational Recommender Systems via Usage-related Questions. RecSys 2021 【LBR，对话推荐系统的偏好启发方法】
-   Bundle MCR: Towards Conversational Bundle Recommendation. RecSys 2022 【马尔可夫决策进行对话式捆绑推荐】
-   Self-Supervised Bot Play for Transcript-Free Conversational Recommendation with Rationales. RecSys 2022 【自监督对话推荐】
-   Multiple Choice Questions based Multi-Interest Policy Learning for Conversational Recommendation. WWW 2022 【基于多兴趣的对话推荐策略学习】

### Social Recommendation

- A Neural Influence Diffusion Model for Social Recommendation. SIGIR 2019 【社交推荐的影响扩散模型】
- Graph Neural Networks for Social Recommendation. WWW 2019 【社交推荐的图神经网络】
- Session-based social recommendation via dynamic graph attention networks. WSDM 2019 【通过动态图注意力网络进行会话的社交推荐】
- Partial Relationship Aware Influence Diffusion via a Multi-channel Encoding Scheme for Social Recommendation. CIKM 2020 【多通道编码方案的部分关系感知影响扩散】
- Random Walks with Erasure: Diversifying Personalized Recommendations on Social and Information Networks. WWW 2021 【社交网络个性化推荐的多样化】

- Dual Side Deep Context-aware Modulation for Social Recommendation. WWW 2021 【社交推荐的上下文感知调制】

- Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. WWW 2021 【用于社交推荐的自监督多通道超图卷积网络】
- Social Recommendation with Self-Supervised Metagraph Informax Network. CIKM 2021【使用自监督元图网络的社交推荐】
- Ranking Friend Stories on Social Platforms with Edge-Contextual Local Graph Convolutions. WSDM 2022 【基于图卷积神经网络的社交排序】
- Socially-Aware Self-Supervised Tri-Training for Recommendation. KDD 2021 【社会意识自监督的立体推荐系统】
- Affective Signals in a Social Media Recommender System. KDD 2022 【社交媒体情感信息】
- I want to break free! Recommending friends from outside the echo chamber. RecSys 2021 【有回声室意识的朋友推荐方法】
- The Dual Echo Chamber: Modeling Social Media Polarization for Interventional Recommending. RecSys 2021 【针对双回声室效应的社交媒体极化模型】
- Auditing the Effect of Social Network Recommendations on Polarization in Geometrical Ideological Spaces. RecSys 2021 【LBR，社交网络推荐对几何意识形态空间中极化的影响】
- Do recommender systems make social media more susceptible to misinformation spreaders?. RecSys 2022 【LBR，错误信息传播者对社交推荐的影响】
- Revisiting Graph based Social Recommendation: A Distillation Enhanced Social Graph Network. WWW 2022 【使用知识蒸馏来融入user-item交互图和user-user社交图的信息】
- Large-scale Personalized Video Game Recommendation via Social-aware Contextualized Graph Neural Network. WWW 202 【同时考虑个性化，游戏上下文，社交联系】

### News Recommendation

-   KRED:Knowledge-Aware Document Representation for News Recommendations. RecSys 2020 【新闻推荐知识感知的文档表示】
-   News Recommendation with Topic-Enriched Knowledge Graphs. CIKM 2020 【主题丰富知识图的新闻推荐】

-   The Interaction between Political Typology and Filter Bubbles in News Recommendation Algorithms. WWW 2021 【新闻推荐算法中政治类型与过滤气泡的相互作用】

-   Personalized News Recommendation with Knowledge-aware News Interactions. SIGIR 2021 【知识感知的个性化新闻推荐】

-   Joint Knowledge Pruning and Recurrent Graph Convolution for News Recommendation. SIGIR 2021 【新闻推荐中的联合知识剪枝和递归图卷积】
-   WG4Rec: Modeling Textual Content with Word Graph for News Recommendation. CIKM 2021【使用Word Graph为新闻推荐建模文本内容】
-   Popularity-Enhanced News Recommendation with Multi-View Interest Representation. CIKM 2021【多视角兴趣学习的流行度增强的新闻推荐】
-   Prioritizing Original News on Facebook. CIKM 2021【applied paper，原创新闻优先级排序】
-   DeepVT: Deep View-Temporal Interaction Network for News Recommendatio. CIKM 2022 【新闻推荐的深度视图-时间交互网络】
-   Generative Adversarial Zero-Shot Learning for Cold-Start News Recommendation. CIKM 2022 【冷启动新闻推荐的生成对抗零样本学习】
-   Personalized Chit-Chat Generation for Recommendation Using External Chat Corpora. KDD 2022 【个性化聊天】
-   Training Large-Scale News Recommenders with Pretrained Language Models in the Loop. KDD 2022 【轻量级编码pipeline】
-   Reinforced Anchor Knowledge Graph Generation for News Recommendation Reasoning. KDD 2021 【新闻推荐推理的增强锚点知识图生成】
-   ProFairRec: Provider Fairness-aware News Recommendation. SIGIR 2022 【商家公平的新闻推荐】
-   Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation. SIGIR 2022 【建模隐式反馈】
-   FUM: Fine-grained and Fast User Modeling for News Recommendation. SIGIR 2022 【short paper，细粒度快速的用户建模】
-   Is News Recommendation a Sequential Recommendation Task?. SIGIR 2022 【short paper，新闻推荐是序列化推荐吗】
-   News Recommendation with Candidate-aware User Modeling. SIGIR 2022 【short paper，候选感知的用户建模】
-   MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation. SIGIR 2022 【short paper，视觉语言学增强的多模态新闻推荐】
-   RADio – Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. RecSys 2022【衡量新闻推荐规范化多样性的度量框架】
-   Reducing Cross-Topic Political Homogenization in Content-Based News Recommendation. RecSys 2022【新闻推荐中减少跨主题的政治同质化】
-   FeedRec: News Feed Recommendation with Various User Feedbacks. WWW 2022 【融入各类显示/隐式反馈来建模用户兴趣】

### Music Recommendation

- Contextual and Sequential User Embeddings for Large-Scale Music Recommendation. RecSys 2020 【大规模音乐推荐的上下文和时序用户表示】
- A Case Study in Educational Recommenders:Recommending Music Partitures at Tomplay. CIKM 2022 【在 Tomplay 推荐音乐片段】
- Counteracting User Attention Bias in Music Streaming Recommendation via Reward Modification. KDD 2022 【用户注意力对抗】
- Follow the guides: disentangling human and algorithmic curation in online music consumption. RecSys 2021 【在线音乐消费中的人为和算法影响】
- The role of preference consistency, defaults and musical expertise in users' exploration behavior in a genre exploration recommender. RecSys 2021 【用户音乐偏好的行为探索】
- Analyzing Item Popularity Bias of Music Recommender Systems: Are Different Genders Equally Affected?. RecSys 2021 【LBR，音乐推荐物品流行度偏差的性别分析】
- Baby Shark to Barracuda: Analyzing Children’s Music Listening Behavior. RecSys 2021 【LBR，儿童音乐行为分析】
- Optimizing the Selection of Recommendation Carousels with Quantum Computing. RecSys 2021【LBR，用量子计算优化音乐歌单推荐】
- Play It Again, Sam! Recommending Familiar Music in Fresh Ways. RecSys 2021【LBR，用巡演推荐重复的音乐】
- Predicting Music Relistening Behavior Using the ACT-R Framework. RecSys 2021【LBR，基于心理学的理论来建模音乐重听行为】
- Siamese Neural Networks for Content-based Cold-Start Music Recommendation. RecSys 2021 【LBR，暹罗神经网络进行音乐冷启动推荐】
- A User-Centered Investigation of Personal Music Tours. RecSys 2022 【以用户为中心的音乐巡演推荐】
- Exploiting Negative Preference in Content-based Music Recommendation with Contrastive Learning. RecSys 2022 【利用对比学习挖掘基于内容的音乐推荐中的负面偏好】
- Exploring the longitudinal effect of nudging on users' genre exploration behavior and listening preference. RecSys 2022 【探索轻推对用户听歌体裁偏好的纵向效应】
- Discovery Dynamics: Leveraging Repeated Exposure for User and Music CharacterizationRecSys 2022 【LBR，探索轻推对用户听歌体裁偏好的纵向效应】
- Efficient Online Learning to Rank for Sequential Music Recommendation. WWW 2022 【将搜索空间限制在此前表现不佳的搜索方向的正交补上】

### Text-aware Recommendation

-   TAFA: Two-headed Attention Fused Autoencoder for Context-Aware Recommendations. RecSys 2020 【用于上下文感知推荐的双头注意力融合自编码器】
-   Set-Sequence-Graph A Multi-View Approach Towards Exploiting Reviews for Recommendation. CIKM 2020 【利用评论进行推荐的多视图方法】

-   TPR: Text-aware Preference Ranking for Recommender Systems. CIKM 2020 【文本感知的偏好排序】

-   Leveraging Review Properties for Effective Recommendation. WWW 2021 【利用评论属性实现高效建议】
-   Counterfactual Review-based Recommendation. CIKM 2021【基于评论的反事实推荐】
-   Review-Aware Neural Recommendation with Cross-Modality Mutual Attention. CIKM 2021【short paper，文本+RS+跨模态】
-   Aligning Dual Disentangled User Representations from Ratings and Textual Content. KDD 2022 【用户双重解耦表示对齐】
-   Collaborative Filtering with Attribution Alignment for Review-based Non-overlapped Cross Domain Recommendation. WWW 2022 【通过属性对齐实现基于评论的跨域推荐】
-   Accurate and Explainable Recommendation via Review Rationalization. WWW 2022 【提取评论中的因果关系】
-   Heterogeneous Interactive Snapshot Network for Review-Enhanced Stock Profiling and Recommendation. IJCAI 2022 【评论增强的股票分析和推荐】

### POI Recommendation

-   HME: A Hyperbolic Metric Embedding Approach for Next-POI Recommendation. SIGIR 2020 【POI 推荐的双曲线度量方法】
-   Spatial Object Recommendation with Hints: When Spatial Granularity Matters. SIGIR 2020 【带提示的空间对象推荐】

-   Geography-Aware Sequential Location Recommendation. KDD 2020 【地理感知的时序位置推荐】

-   Learning Graph-Based Geographical Latent Representation for Point-of-Interest Recommendation. CIKM 2020 【基于图的地理表示的 POI 推荐】

-   STP-UDGAT Spatial-Temporal-Preference User Dimensional Graph Attention Network for Next POI Recommendation. CIKM 2020 【POI 推荐的时空偏好用户维图注意力网络】

-   STAN: Spatio-Temporal Attention Network for next Point-of-Interest Recommendation. WWW 2021 【POI 推荐的时空注意力网络】

-   Incremental Spatio-Temporal Graph Learning for Online Query-POI Matching. WWW 2021 【在线查询 POI 匹配的增量时空图学习】
-   Answering POI-recommendation Questions using Tourism Reviews. CIKM 2021【使用旅游者评论回答POI问题】
-   SNPR: A Serendipity-Oriented Next POI Recommendation Model. CIKM 2021【面向偶然性的POI推荐】
-   You Are What and Where You Are: Graph Enhanced Attention Network for Explainable POI Recommendation. CIKM 2021【applied paper，Attention 图神经网络用于可解释推荐】
-   ST-PIL: Spatial-Temporal Periodic Interest Learning for Next Point-of-Interest Recommendation. CIKM 2021【short paper，用于POI推荐的时空周期兴趣学习】
-   Graph-Flashback Network for Next Location Recommendation. KDD 2022 【引入POI相似度权重】
-   Curriculum Meta-Learning for Next POI Recommendation. KDD 2021 【基于元学习的下一代兴趣点推荐系统】
-   Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation. SIGIR 2022 【多任务图循环网络】
-   Learning Graph-based Disentangled Representations for Next POI Recommendation. SIGIR 2022 【学习基于图的解耦表示】
-   GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation. SIGIR 2022 【轨迹图加强的Transformer】
-   Next Point-of-Interest Recommendation with Auto-Correlation Enhanced Multi-Modal Transformer Network. SIGIR 2022 【short paper，自修正的多模态Transformer】
-   Empowering Next POI Recommendation with Multi-Relational Modeling. SIGIR 2022 【多重关系建模】
-   Exploring the Impact of Temporal Bias in Point-of-Interest Recommendation. RecSys 2022 【LBR，时间偏差对兴趣点推荐的影响】
-   Modeling Spatio-temporal Neighbourhood for Personalized Point-of-interest Recommendation. IJCAI 2022 【融入知识图谱和时域信息，实现个性化POI推荐】
-   Next Point-of-Interest Recommendation with Inferring Multi-step Future Preferences. IJCAI 2022 【考虑用户未来偏好】

### Online Recommendation

-   Gemini: A novel and universal heterogeneous graph information fusing framework for online recommendations. KDD 2020 【用于在线推荐的新型通用异构图信息融合框架】
-   Exploring Clustering of Bandits for Online Recommendation System. RecSys 2020 【在线推荐系统中的 bandit 聚类研究】
-   Contextual User Browsing Bandits for Large-Scale Online Mobile Recommendation. RecSys 2020 【大规模在线移动推荐的上下文用户浏览 bandit】
-   A Hybrid Bandit Framework for Diversified Recommendation. AAAI 2021 【多元化推荐的混合 Bandit 框架】
-   Generative Inverse Deep Reinforcement Learning for Online Recommendation. CIKM 2021【用于在线推荐的生成式逆强化学习】
-   Long Short-Term Temporal Meta-learning in Online Recommendation. WSDM 2022【在线推荐中的长短期时间元学习】
-   A Cooperative-Competitive Multi-Agent Framework for Auto-bidding in Online Advertising. WSDM 2022【一种用于在线广告自动竞价的合作竞争多代理框架】
-   Knowledge Extraction and Plugging for Online Recommendation. CIKM 2022 【在线推荐的知识抽取与插入】
-   Real-time Short Video Recommendation on Mobile Devices. CIKM 2022 【移动端实时短视频推荐】
-   SASNet: Stage-aware sequential matching for online travel recommendation. CIKM 2022 【在线旅游推荐的阶段感知序列匹配】
-   Meta-Learning for Online Update of Recommender Systems AAAI 2022 【自适应最优学习率】
-   Architecture and Operation Adaptive Network for Online Recommendations. KDD 2021 【在线推荐系统的架构及其自适应网络】
-   Recommendation on Live-Streaming Platforms: Dynamic Availability and Repeat Consumption. RecSys 2021 【实时流平台推荐的动态可用性和重复消费】
-   A GPU-specialized Inference Parameter Server for Large-Scale Deep Recommendation Models. RecSys 2022 【大规模在线推理模型基于 GPU 高速缓存的分布式框架】
-   Modeling User Repeat Consumption Behavior for Online Novel Recommendation. RecSys 2022 【在线小说推荐的用户重复消费行为建模】

### Group Recommendation

- Bundle Recommendation with Graph Convolutional Networks. SIGIR 2020 【基于图卷积网络的捆绑推荐 】
- GAME: Learning Graphical and Attentive Multi-view Embeddings for Occasional Group Recommendation. SIGIR 2020 【多视图表示的组推荐】
- GroupIM: A Mutual Information Maximizing Framework for Neural Group Recommendation. SIGIR 2020 【组推荐的互信息最大化框架】

- Group-Aware Long- and Short-Term Graph Representation Learning for Sequential Group Recommendation. SIGIR 2020 【序列群推荐的群体感知长短期图表示学习】
- Ensuring Fairness in Group Recommendations by Rank-Sensitive Balancing of Relevance. RecSys 2020 【通过排名敏感的相关性平衡确保群组推荐的公平性】
- Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation. CIKM 2021【用于群组推荐的自监督超图学习】
- DeepGroup: Group Recommendation with Implicit Feedback. CIKM 2021【short paper，隐式反馈的群组推荐】
- Multi-View Intent Disentangle Graph Networks for Bundle Recommendation. AAAI 2022 【多意图捆绑推荐】
- Thinking inside The Box: Learning Hypercube Representations for Group Recommendation. SIGIR 2022 【超立方体表示用于组推荐】
- Enumerating Fair Packages for Group Recommendations. WSDM 2022【枚举组推荐的公平包】
- GBERT: Pre-training User representations for Ephemeral Group Recommendation. CIKM 2022 【为临时组推荐预训练用户表示】
- BRUCE – Bundle Recommendation Using Contextualized item Embeddings. RecSys 2022 【Transformer 建模上下文进行捆绑推荐】
- Bundle MCR: Towards Conversational Bundle Recommendation. RecSys 2022 【马尔可夫决策进行对话式捆绑推荐】

### Multi-task/Multi-behavior/Cross-domain Recommendations

-   Transfer Learning via Contextual Invariants for One-to-Many Cross-Domain Recommendation. SIGIR 2020 【一对多跨域推荐迁移学习】
-   CATN: Cross-Domain Recommendation for Cold-Start Users via Aspect Transfer Network. SIGIR 2020 【为冷启动用户提供跨域建议】

-   Multi-behavior Recommendation with Graph Convolution Networks. SIGIR 2020 【基于图卷积网络的多行为推荐】

-   Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation. SIGIR 2020 【用户建模和推荐中序列行为的参数有效转换】

-   Web-to-Voice Transfer for Product Recommendation on Voice. SIGIR 2020 【语音产品推荐的网络语音转换】

-   Incorporating User Micro-behaviors and Item Knowledge into Multi-task Learning for Session-based Recommendation. SIGIR 2020 【将用户行为和物品知识融入基于会话的推荐多任务学习】

-   Jointly Learning to Recommend and Advertise. KDD 2020 【推荐和广告的结合】

-   Progressive Layered Extraction (PLE) A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations. RecSys 2020 【个性化推荐的多任务学习模型】

-   Whole-Chain Recommendations. CIKM 2020 【全链推荐】

-   Personalized Approximate Pareto-Efficient Recommendation. WWW 2021 【近似帕累托效率的推荐】

-   Federated Collaborative Transfer for Cross-Domain Recommendation. SIGIR 2021 【跨域推荐的联邦协作转换】

-   Learning Domain Semantics and Cross-Domain Correlations for Paper Recommendation. SIGIR 2021 【领域语义和跨领域关联用于论文推荐】

-   Graph Meta Network for Multi-Behavior Recommendation with Interaction Heterogeneity and Diversity. SIGIR 2021 【具有交互异质性和多样性的多行为推荐图元网络】

-   Multi-task Learning for Bias-Free Joint CTR Prediction and Market Price Modeling in Online Advertising. CIKM 2021【在线广告无偏差联合CTR预估和市场价格建模的多任务学习】
-   Cross-Market Product Recommendation. CIKM 2021【跨市场产品推荐】
-   Expanding Relationship for Cross Domain Recommendation. CIKM 2021【扩展跨领域推荐的关系】
-   Learning Representations of Inactive Users: A Cross Domain Approach with Graph Neural Networks. CIKM 2021【short paper，跨领域方法结合图神经网络用于学习非活跃用户表示】
-   Low-dimensional Alignment for Cross-Domain Recommendation. CIKM 2021【short paper，跨领域推荐的低维对齐】
-   Multi-Sparse-Domain Collaborative Recommendation via Enhanced Comprehensive Aspect Preference Learning. WSDM 2022【通过增强的综合方面偏好学习的多稀疏域协作推荐】
-   Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling. WSDM 2022【一种用于广告商建模的多场景多任务元学习方法】
-   RecGURU: Adversarial Learning of Generalized User Representations for Cross-Domain Recommendation. WSDM 2022【用于跨域推荐的广义用户表示的对抗性学习】
-   Personalized Transfer of User Preferences for Cross-domain Recommendation. WSDM 2022【跨域推荐用户偏好的个性化传输】
-   Adaptive Domain Interest Network for Multi-domain Recommendation. CIKM 2022 【多域推荐的自适应域兴趣网络】
-   Multi-Scale User Behavior Network for Entire Space Multi-Task Learning. CIKM 2022 【全空间多任务学习的多尺度用户行为网络】
-   Gromov-Wasserstein Guided Representation Learning for Cross-Domain Recommendation. CIKM 2022 【跨域推荐表示学习】
-   Contrastive Cross-Domain Sequential Recommendation. CIKM 2022 【对比跨域序列推荐】
-   Cross-domain Recommendation via Adversarial Adaptation. CIKM 2022【通过对抗性适应进行跨域推荐】
-   Dual-Task Learning for Multi-Behavior Sequential Recommendation. CIKM 2022 【多行为序列推荐的双任务学习】
-   FedCDR: Federated Cross-Domain Recommendation for Privacy-Preserving Rating Prediction. CIKM 2022 【FedCDR：用于隐私保护评级预测的联合跨域推荐】
-   Leveraging Multiple Types of Domain Knowledge for Safe and Effective Drug Recommendation. CIKM 2022 【利用多种领域知识进行安全有效的药物推荐】
-   Multi-Faceted Hierarchical Multi-Task Learning for Recommender Systems. CIKM 2022 【推荐系统的多方面分层多任务学习】
-   Review-Based Domain Disentanglement without Duplicate Users or Contexts for Cross-Domain Recommendation. CIKM 2022 【没有重复用户或上下文的基于审查的域解耦，用于跨域推荐】
-   Scenario-Adaptive and Self-Supervised Model for Multi-Scenario Personalized Recommendation. CIKM 2022 【多场景个性化推荐的场景自适应自监督模型】
-   Cross-Task Knowledge Distillation in Multi-Task Recommendation. AAAI 2022 【跨任务知识蒸馏】
-   Towards Universal Sequence Representation Learning for Recommender Systems. KDD 2022  【引入物品描述文本】
-   Contrastive Cross-domain Recommendation in Matching. KDD 2022 【对比学习捕获用户兴趣】
-   Multi-Behavior Hypergraph-Enhanced Transformer for Sequential Recommendation. KDD 2022 【多行为超图增强】
-   Multi-Task Fusion via Reinforcement Learning for Long-Term User Satisfaction in Recommender Systems. KDD 2022 【Batch RL建模MTF】
-   CausalInt: Causal Inspired Intervention for Multi-Scenario Recommendation. KDD 2022 【多场景推荐统一框架】
-   Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising. KDD 2021 【序列依赖多任务学习】
-   Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising. KDD 2021 【混合场景多任务学习】
-   Adversarial Feature Translation for Multi-domain Recommendation. KDD 2021 【对抗特征迁移多任务学习】
-   Debiasing Learning based Cross-domain Recommendation. KDD 2021 【跨域推荐纠偏】
-   Co-training Disentangled Domain Adaptation Network for Leveraging Popularity Bias in Recommenders. SIGIR 2022 【训练解耦的域适应网络来利用流行度偏差】
-   DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation. SIGIR 2022 【解耦表示】
-   Doubly-Adaptive Reinforcement Learning for Cross-Domain Interactive Recommendation. SIGIR 2022 【双重适应的强化学习】
-   Exploiting Variational Domain-Invariant User Embedding for Partially Overlapped Cross Domain Recommendation. SIGIR 2022 【域不变的用户嵌入】
-   Multi-Behavior Sequential Transformer Recommender. SIGIR 2022 【多行为序列化Transformer】
-   Towards Source-Aligned Variational Models for Cross-Domain Recommendation. RecSys 2021 【变分自动编码器用于跨域推荐】
-   An Analysis Of Entire Space Multi-Task Models For Post-Click Conversion Prediction. RecSys 2021 【LBR，点击后转换率预测的全空间多任务模型分析】
-   MARRS: A Framework for multi-objective risk-aware route recommendation using Multitask-Transformer. RecSys 2022 【利用多任务 Transformer 进行多目标的路线推荐】
-   M2TRec: Metadata-aware Multi-task Transformer for Large-scale and Cold-start free Session-based Recommendations. RecSys 2022【LBR，基于元数据和多任务 Transformer 的冷启动会话推荐系统】
-   MARRS: A Framework for multi-objective risk-aware route recommendation using Multitask-Transformer. RecSys 2022 【利用多任务 Transformer 进行多目标的路线推荐】
-   Collaborative Filtering with Attribution Alignment for Review-based Non-overlapped Cross Domain Recommendation. WWW 2022 【通过属性对齐实现基于评论的跨域推荐】
-   Differential Private Knowledge Transfer for Privacy-Preserving Cross-Domain Recommendation. WWW 2022 【通过可微隐私知识迁移实现源域隐私保护的跨域推荐】
-   MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks. WWW 2022 【动态保持辅助任务和目标任务的梯度在同一个量级】
-   A Contrastive Sharing Model for Multi-Task Recommendation. WWW 2022 【使用对比掩码来解决多任务中的参数冲突问题】
-   Self-supervised Graph Neural Networks for Multi-behavior Recommendation. IJCAI 2022 【GNN + 多行为推荐】
-   Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation. WWW 2022 【通过re-contrast,re-attend,re-construct来增强解耦用户多兴趣表示】

### Multimodal Recommendation

-   Multi-modal Knowledge Graphs for Recommender Systems. CIKM 2020 【推荐系统的多模态知识图】
-   Pre-training Graph Transformer with Multimodal Side Information for Recommendation. ACM MM 2021 【多模态侧信息的预训练图 Transformer 用于推荐】
-   Review-Aware Neural Recommendation with Cross-Modality Mutual Attention. CIKM 2021【short paper，文本+RS+跨模态】
-   Click-Through Rate Prediction with Multi-Modal Hypergraphs. CIKM 2021【使用多模态超图的点击率预测】
-   Multimodal Graph Meta Contrastive Learning. CIKM 2021【short paper，多模态元图对比学习】
-   AutoMARS: Searching to Compress Multi-Modality Recommendation Systems. CIKM 2022 【搜索压缩多模态推荐系统】
-   Multimodal Meta-Learning for Cold-Start Sequential Recommendation. CIKM 2022 【冷启动序列推荐的多模态元学习】
-   MARIO: Modality-Aware Attention and Modality-Preserving Decoders for Multimedia Recommendation. CIKM 2022 【用于多媒体推荐的模态感知注意力和模态保留解码器】
-   ItemSage: Learning Product Embeddings for Shopping Recommendations at Pinterest. KDD 2022 【多模态商品推荐】
-   SEMI: A Sequential Multi-Modal Information Transfer Network for E-Commerce Micro-Video Recommendations. KDD 2021 【基于序列多模态信息传输网络的电商微视频推荐系统】
-   MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation. SIGIR 2022 【short paper，视觉语言学增强的多模态新闻推荐】
-   Semi-Supervised Visual Representation Learning for Fashion Compatibility. RecSys 2021 【搭配时装预测的半监督视觉表征学习】
-   Tops, Bottoms, and Shoes: Building Capsule Wardrobes via Cross-Attention Tensor Network. RecSys 2021 【通过视觉的交叉注意力张力网络提供服饰搭配推荐】
-   Multi-Modal Dialog State Tracking for Interactive Fashion Recommendation. RecSys 2022 【交互式时装推荐的多模态注意网络】

### Other Tasks

-   Octopus: Comprehensive and Elastic User Representation for the Generation of Recommendation Candidates. SIGIR 2020 【生成推荐候选的综合弹性用户表示】
-   How to Retrain a Recommender System? SIGIR 2020 【重训练推荐系统】
-   Learning Personalized Risk Preferences for Recommendation. SIGIR 2020 【个性化风险偏好】
-   Distributed Equivalent Substitution Training for Large-Scale Recommender Systems. SIGIR 2020 【分布式等效替换训练】

-   Beyond User Embedding Matrix: Learning to Hash for Modeling Large-Scale Users in Recommendation. SIGIR 2020 【推荐中为大规模用户哈希建模】
-   Improving Recommendation Quality in Google Drive. KDD 2020 【提升谷歌云盘的推荐质量】
-   Goal-driven Command Recommendations for Analysts. RecSys 2020 【针对分析员的目标驱动的命令推荐】
-   MultiRec: A Multi-Relational Approach for Unique Item Recommendation in Auction Systems. RecSys 2020 【拍卖系统中唯一物品推荐的多关系方法】
-   PURS: Personalized Unexpected Recommender System for Improving User Satisfaction. RecSys 2020 【提高用户满意度的个性化意外推荐系统】
-   RecSeats: A Hybrid Convolutional Neural Network Choice Model for Seat Recommendations at Reserved Seating Venues. RecSys 2020 【预约座位推荐的混合卷积神经网络模型】
-   Live Multi-Streaming and Donation Recommendations via Coupled Donation-Response Tensor Factorization. CIKM 2020 【实时多流媒体和捐赠推荐】
-   Learning to Recommend from Sparse Data via Generative User Feedback. AAAI 2021 【通过生成的用户反馈从稀疏数据中进行推荐】
-   Real-time Relevant Recommendation Suggestion. WSDM 2021 【实时的相关推荐建议】
-   FINN: Feedback Interactive Neural Network for Intent Recommendation. WWW 2021 【用于意向推荐的反馈交互式神经网络】
-   Drug Package Recommendation via Interaction-aware Graph Induction. WWW 2021 【通过交互感知图归纳推荐药品包装】
-   Large-scale Comb-K Recommendation. WWW 2021 【大规模 Comb-K 推荐】
-   Variation Control and Evaluation for Generative Slate Recommendations. WWW 2021 【生成性板岩推荐的变化控制和评估】
-   UGRec: Modeling Directed and Undirected Relations for Recommendation. SIGIR 2021 【为推荐建立有向和无向关系模型】
-   Learning Recommender Systems with Implicit Feedback via Soft Target Enhancement. SIGIR 2021 【通过软目标增强学习具有隐式反馈的推荐系统】
-   PreSizE: Predicting Size in E-Commerce using Transformers. SIGIR 2021 【利用 Transformer 预测电子商务的规模】
-   Path-based Deep Network for Candidate Item Matching in Recommenders. SIGIR 2021 【基于路径的候选物品匹配】
-   Learning An End-to-End Structure for Retrieval in Large-Scale Recommendations. CIKM 2021【在大规模推荐中学习一个端到端的结构用于检索】
-   USER: A Unified Information Search and Recommendation Model based on Integrated Behavior Sequence. CIKM 2021【基于集成行为序列的统一搜索与推荐模型】
-   Multi-hop Reading on Memory Neural Network with Selective Coverage for Medication Recommendation. CIKM 2021【药物推荐】
-   Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations. WSDM 2022【面向交互式个性化推荐的整个商品空间探索】
-   Adapting Triplet Importance of Implicit Feedback for Personalized Recommendation. CIKM 2022 【在个性化推荐中调整隐式反馈的三元组重要性】
-   GRP: A Gumbel-based Rating Prediction Framework for Imbalanced Recommendation. CIKM 2022 【基于 Gumbel 的不平衡推荐评级预测框架】
-   Rank List Sensitivity of Recommender Systems to Interaction Perturbations. CIKM 2022 【推荐系统对交互扰动的排名列表敏感性】
-   CROLoss: Towards a Customizable Loss for Retrieval Models in Recommender Systems. CIKM 2022 【推荐系统中检索模型的可定制损失】
-   Towards Principled User-side Recommender Systems. CIKM 2022 【迈向有原则的用户侧推荐系统】
-   UDM: A Unified Deep Matching Framework in Recommender System. CIKM 2022 【推荐系统中的统一深度匹配框架】
-   User Recommendation in Social Metaverse with VR. CIKM 2022 【VR 的用户推荐】
-   AdaFS: Adaptive Feature Selection in Deep Recommender System. KDD 2022 【自适应特征选择】
-   CAPTOR: A Crowd-Aware Pre-Travel Recommender System for Out-of-Town Users. SIGIR 2022 【为乡村用户提供的旅游推荐】
-   PERD: Personalized Emoji Recommendation with Dynamic User Preference. SIGIR 2022 【short paper，个性化表情推荐】
-   Item Similarity Mining for Multi-Market Recommendation. SIGIR 2022 【short paper，多市场推荐中的商品相似度挖掘】
-   A Content Recommendation Policy for Gaining Subscribers. SIGIR 2022 【short paper，为提升订阅者的内容推荐策略】
-   Accordion: A Trainable Simulator for Long-Term Interactive Systems. RecSys 2021 【长期交互系统的可训练模拟器】
-   Information Interactions in Outcome Prediction: Quantification and Interpretation using Stochastic Block Models. RecSys 2021 【结果预测中的信息交互】
-   Local Factor Models for Large-Scale Inductive Recommendation. RecSys 2021 【归纳推荐的局部因子模型】
-   Reverse Maximum Inner Product Search: How to efficiently find users who would like to buy my item?. RecSys 2021 【反向最大内部产品搜索】
-   “Serving Each User”: Supporting Different Eating Goals Through a Multi-List Recommender Interface. RecSys 2021 【通过多列表推荐界面个性化食品推荐】
-   An Interpretable Recommendation Model for Gerontological Care. RecSys 2021【LBR，老年护理的推荐研究】
-   Automatic Collection Creation and Recommendation. RecSys 2021【LBR，自动创建和推荐收藏夹】
-   Estimating and Penalizing Preference Shifts in Recommender Systems. RecSys 2021【LBR，推荐系统中偏好转移的分析】
-   Global-Local Item Embedding for Temporal Set Prediction. RecSys 2021【LBR，时间集预测任务】
-   Modeling Two-Way Selection Preference for Person-Job Fit. RecSys 2022 【建模双向选择偏好的人岗匹配模型】
-   Towards Psychologically Grounded Dynamic Preference Models. RecSys 2022 【基于心理学的动态偏好建模】



Topic
-----

### Debias in Recommender System

-   Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems. SIGIR 2020 【个性化排名系统中衡量和减轻推荐偏差】
-   Attribute-based Propensity for Unbiased Learning in Recommender Systems Algorithm and Case Studies. KDD 2020 【基于属性的无偏学习倾向】
-   Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions. KDD 2020 【结合序列奖励交互的反事实评估】
-   Debiasing Item-to-Item Recommendations With Small Annotated Datasets. RecSys 2020 【用带注释的小数据集对物-物推荐去偏】
-   Keeping Dataset Biases out of the Simulation : A Debiased Simulator for Reinforcement Learning based Recommender Systems. RecSys 2020 【基于强化学习的去偏模拟器】
-   Unbiased Ad Click Prediction for Position-aware Advertising Systems. RecSys 2020 【无偏的广告点击预测】
-   Unbiased Learning for the Causal Effect of Recommendation. RecSys 2020 【推荐因果效应的无偏学习】
-   E-commerce Recommendation with Weighted Expected Utility. CIKM 2020 【考虑加权预期效用的电子商务推荐】
-   Popularity-Opportunity Bias in Collaborative Filtering. WSDM 2021 【协同过滤中的流行机会偏差】
-   Combating Selection Biases in Recommender Systems with a Few Unbiased Ratings. WSDM 2021 【无偏评级对抗推荐的选择偏差】
-   Leave No User Behind Towards Improving the Utility of Recommender Systems for Non-mainstream Users. WSDM 2021 【关注小众的用户】
-   Non-Clicks Mean Irrelevant Propensity Ratio Scoring As a Correction. WSDM 2021 【不相关的倾向得分作为推荐结果的修正】
-   Diverse User Preference Elicitation with Multi-Armed Bandits. WSDM 2021 【多臂老虎机的多种用户偏好启发】
-   Unbiased Learning to Rank in Feeds Recommendation. WSDM 2021 【Feeds 推荐中的无偏学习】
-   Debiasing Career Recommendations with Neural Fair Collaborative Filtering. WWW 2021 【职业推荐的公平性去偏】
-   AutoDebias: Learning to Debias for Recommendation. SIGIR 2021 【推荐的自动去偏】
-   Mitigating Sentiment Bias for Recommender Systems. SIGIR 2021 【减轻推荐系统的情感偏见】
-   Causal Intervention for Leveraging Popularity Bias in Recommendation. SIGIR 2021 【在推荐中利用流行偏见的因果干预】
-   Enhanced Doubly Robust Learning for Debiasing Post-Click Conversion Rate Estimation. SIGIR 2021 【点击后转换率估计】
-   Cross-Positional Attention for Debiasing Clicks. WWW 2021 【跨位置的注意力机制用于点击去偏】
-   Popcorn: Human-in-the-loop Popularity Debiasing in Conversational Recommender Systems. CIKM 2021【采用人在回路方式进行对话推荐系统的流行度去偏】
-   Unbiased Filtering of Accidental Clicks in Verizon Media Native Advertising. CIKM 2021【applied paper，广告意外点击的无偏过滤】
-   CauSeR: Causal Session-based Recommendations for Handling Popularity Bias. CIKM 2021【short paper，用于流行度去偏的因果关系序列推荐】
-   It Is Different When Items Are Older: Debiasing Recommendations When Selection Bias and User Preferences are Dynamic. WSDM 2022【选择偏差和偏好偏差动态变化时的纠偏推荐系统】
-   Fighting Mainstream Bias in Recommender Systems via Local Fine Tuning. WSDM 2022【通过局部微调对抗推荐系统中的主流偏见】
-   Towards Unbiased and Robust Causal Ranking for Recommender Systems. WSDM 2022【推荐系统的无偏和稳健因果排名】
-   Quantifying and Mitigating Popularity Bias in Conversational Recommender Systems. CIKM 2022 【量化和减轻会话推荐系统中的流行度偏差】
-   Learning Unbiased User Behaviors Estimation with Hierarchical Recurrent Model on the Entire Space. CIKM 2022 【分层递归模型学习无偏用户行为估计】
-   Representation Matters When Learning From Biased Feedback in Recommendation. CIKM 2022 【从推荐中的有偏反馈中学习时，表征很重要】
-   Invariant Preference Learning for General Debiasing in Recommendation. KDD 2022 【不变偏好解耦】
-   Counteracting User Attention Bias in Music Streaming Recommendation via Reward Modification. KDD 2022 【用户注意力对抗】
-   Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation. KDD 2022 【视频时长去偏】
-   Debiasing the Cloze Task in Sequential Recommendation with Bidirectional Transformers. KDD 2022 【序列完形填空去偏】
-   Debiasing Learning for Membership Inference Attacks Against Recommender Systems. KDD 2022 【VAE 去偏】
-   Popularity Bias in Dynamic Recommendation. KDD 2021 【动态推荐系统的热度纠偏】
-   Debiasing Learning based Cross-domain Recommendation. KDD 2021 【跨域推荐纠偏】
-   Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System. KDD 2021 【不可知反事实推理模型消除推荐系统的流行偏差】
-   Interpolative Distillation for Unifying Biased and Debiased Recommendation 
-   Co-training Disentangled Domain Adaptation Network for Leveraging Popularity Bias in Recommenders. SIGIR 2022 【训练解耦的域适应网络来利用流行度偏差】
-   Bilateral Self-unbiased Recommender Learning from Biased Implicit Feedback. SIGIR 2022 【双边去偏】
-   Mitigating Consumer Biases in Recommendations with Adversarial Training. SIGIR 2022 【short paper，对抗训练去偏】
-   Neutralizing Popularity Bias in Recommendation Models. SIGIR 2022 【short paper，中和流行度偏差】
-   DeSCoVeR: Debiased Semantic Context Prior for Venue Recommendation. SIGIR 2022 【short paper，在场所推荐中去除语义上下文先验】
-   An Audit of Misinformation Filter Bubbles on YouTube: Bubble Bursting and Recent Behavior Changes. RecSys 2021 【虚假信息过滤气泡的审计：走出信息茧房并改变近期行为】
-   Debiased Explainable Pairwise Ranking from Implicit Feedback. RecSys 2021【去偏的可解释贝叶斯个性化排名】
-   Mitigating Confounding Bias in Recommendation via Information Bottleneck. RecSys 2021【信息瓶颈缓解混杂偏倚】
-   User Bias in Beyond-Accuracy Measurement of Recommendation Algorithms. RecSys 2021【超精度测量中的算法用户偏差】
-   Analyzing Item Popularity Bias of Music Recommender Systems: Are Different Genders Equally Affected?. RecSys 2021 【LBR，音乐推荐物品流行度偏差的性别分析】
-   The Idiosyncratic Effects of Adversarial Training on Bias in Personalized Recommendation Learning. RecSys 2021 【LBR，对抗训练对流行度偏差的影响研究】
-   Transfer Learning in Collaborative Recommendation for Bias Reduction. RecSys 2021 【LBR，减少偏差的迁移学习】
-   Countering Popularity Bias by Regularizing Score Differences. RecSys 2022 【利用正则化分数差异减少流行度偏差】
-   Off-Policy Actor Critic for Recommender Systems. RecSys 2022 【离线演员-评论家强化学习算法缓解分布偏差问题】
-   Exploring the Impact of Temporal Bias in Point-of-Interest Recommendation. RecSys 2022 【LBR，时间偏差对兴趣点推荐的影响】
-   Trading Hard Negatives and True Negatives: A Debiased Contrastive Collaborative Filtering Approach. IJCAI 2022 【探索正确的负样本】
-   Debiased, Longitudinal and Coordinated Drug Recommendation through Multi-Visit Clinic Records. NIPS 2022 【去偏药物推荐】
-   CBR: Context Bias aware Recommendation for Debiasing User Modeling and Click Prediction. WWW 2022 【去除由丰富交互造成的上下文偏差】
-   UKD: Debiasing Conversion Rate Estimation via Uncertainty-regularized Knowledge Distillation. WWW 2022 【基于不确定性正则化和知识提取的转化率估计去偏】

### Fairness in Recommender System

-   Fairness-Aware Explainable Recommendation over Knowledge Graphs. SIGIR 2020 【知识图谱上公平性感知的可解释推荐】
-   Ensuring Fairness in Group Recommendations by Rank-Sensitive Balancing of Relevance. RecSys 2020 【通过排名敏感的相关性平衡确保群组推荐的公平性】
-   Fairness-Aware News Recommendation with Decomposed Adversarial Learning. AAAI 2021 【基于分解对抗学习的公平新闻推荐】
-   Practical Compositional Fairness Understanding Fairness in Multi-Component Recommender Systems. WSDM 2021 【多组件推荐中的公平性】
-   Towards Long-term Fairness in Recommendation. WSDM 2021 【推荐中的长期公平性】
-   Learning Fair Representations for Recommendation: A Graph-based Perspective. WWW 2021 【图视角理解推荐中的公平性】
-   User-oriented Group Fairness In Recommender Systems. WWW 2021 【推荐系统中用户导向的群体公平性】
-   Personalized Counterfactual Fairness in Recommendation. SIGIR 2021 【推荐中个性化的反事实公平性】
-   TFROM: A Two-sided Fairness-Aware Recommendation Model for Both Customers and Providers. SIGIR 2021 【针对顾客和供应商双方的公平性感知推荐模型】
-   Fairness among New Items in Cold Start Recommender Systems. SIGIR 2021 【冷启动推荐系统中新物品的公平性】
-   SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios. CIKM 2021【applied paper，用于个性化公平推荐的场景感知排名网络】
-   Enumerating Fair Packages for Group Recommendations. WSDM 2022【枚举组推荐的公平包】
-   Toward Pareto Efficient Fairness-Utility Trade-off in Recommendation through Reinforcement Learning. WSDM 2022【通过强化学习在推荐中实现帕累托高效的公平-效用权衡】
-   Make Fairness More Fair: Fair Item Utility Estimation and Exposure Re-Distribution. KDD 2022 【物品公平效益】
-   Comprehensive Fair Meta-learned Recommender System. KDD 2022 【元学习公平性框架】
-   Online Certification of Preference-Based Fairness for Personalized Recommender Systems. AAAI 2022 【强化学习探索】
-   Joint Multisided Exposure Fairness for Recommendation. SIGIR 2022 【综合考虑多边的曝光公平性】
-   ProFairRec: Provider Fairness-aware News Recommendation. SIGIR 2022 【商家公平的新闻推荐】
-   CPFair: Personalized Consumer and Producer Fairness Re-ranking for Recommender Systems. SIGIR 2022 【用户和商家公平的重排序】
-   Explainable Fairness for Feature-aware Recommender Systems. SIGIR 2022 【考虑特征的推荐系统中的可解释公平】
-   Selective Fairness in Recommendation via Prompts. SIGIR 2022 【short paper，通过提示保证可选的公平性】
-   Regulating Provider Groups Exposure in Recommendations. SIGIR 2022 【short paper，调整商家组曝光】
-   Adversary or Friend? An adversarial Approach to Improving Recommender Systems. RecSys 2022 【对抗式方法促进推荐系统公平性】
-   Fairness-aware Federated Matrix Factorization. RecSys 2022 【结合差异隐私技术的公平感知的联邦矩阵分解】
-   Toward Fair Federated Recommendation Learning: Characterizing the Inter-Dependence of System and Data Heterogeneity. RecSys 2022 【推荐中公平的联邦学习】


-   Link Recommendations for PageRank Fairness. WWW 2022 【PageRank算法链接预测中的公平性】
-   FairGAN: GANs-based Fairness-aware Learning for Recommendations with Implicit Feedback. WWW 2022 【将物品曝光公平性问题映射为隐式反馈数据中缺乏负反馈的问题】


### Attack in Recommender System

-   Revisiting Adversarially Learned Injection Attacks Against Recommender Systems. RecSys 2020 【推荐系统对抗学习的注入攻击】
-   Attacking Recommender Systems with Augmented User Profiles. CIKM 2020 【推荐中对增强用户简介的攻击】

-   A Black-Box Attack Model for Visually-Aware Recommenders. WSDM 2021 【视觉感知推荐的黑盒攻击模型】

-   Denoising Implicit Feedback for Recommendation. WSDM 2021 【推荐隐反馈的去噪】

-   Adversarial Item Promotion: Vulnerabilities at the Core of Top-N Recommenders that Use Images to Address Cold Start. WWW 2021 【使用图像解决冷启动推荐的漏洞】

-   Graph Embedding for Recommendation against Attribute Inference Attacks. WWW 2021 【针对属性推理攻击的图表示】

-   Fight Fire with Fire: Towards Robust Recommender Systems via Adversarial Poisoning Training. SIGIR 2021 【对抗性 poisoning 训练】
-   PipAttack: Poisoning Federated Recommender Systems for Manipulating Item Promotion. WSDM 2022【用于操纵项目促销的中毒联合推荐系统】
-   FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling. KDD 2022 【联邦推荐的对抗攻击】
-   Knowledge-enhanced Black-box Attacks for Recommendations. KDD 2022 【知识图谱增强】
-   Debiasing Learning for Membership Inference Attacks Against Recommender Systems. KDD 2022 【VAE去偏】
-   Data Poisoning Attack against Recommender System Using Incomplete and Perturbed Data. KDD 2021 【不完整及扰动数据攻击推荐系统】
-   Initialization Matters: Regularizing Manifold-informed Initialization for Neural Recommendation Systems. KDD 2021 【基于正则化信息的流形神经网络推荐系统】
-   Triple Adversarial Learning for Influence based Poisoning Attack in Recommender Systems. KDD 2021 【三元对抗学习在推荐系统中毒攻击】
-   Black-Box Attacks on Sequential Recommenders via Data-Free Model Extraction. RecSys 2021 【序列推荐的黑盒攻击】
-   Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders. RecSys 2022 【序列推荐中基于替代的对抗性攻击算法】
-   Revisiting Injective Attacks on Recommender Systems. NIPS 2022 【重新审视对推荐系统的注入式攻击】

### Explanation in Recommender System

- Try This Instead: Personalized and Interpretable Substitute Recommendation. KDD 2020 【个性化和可解释的替代推荐】
- CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation. CIKM 2020 【用于可解释推荐的符号推理】

- Explainable Recommender Systems via Resolving Learning Representations. CIKM 2020 【解决表示学习进行可解释推荐】

- Generate Neural Template Explanations for Recommendation. CIKM 2020 【生成神经模板解释以供推荐】

- Explainable Recommendation with Comparative Constraints on Product Aspects. WSDM 2021 【产品方面有比较约束的可解释推荐】

- Explanation as a Defense of Recommendation. WSDM 2021 【解释作为推荐的防御】

- EX$^3$: Explainable Product Set Recommendation for Comparison Shopping. WWW 2021 【可解释的产品集推荐】

- Learning from User Feedback on Explanations to Improve Recommender Models. WWW 2021 【从用户反馈中学习解释】

- Cost-Effective and Interpretable Job Skill Recommendation with Deep Reinforcement Learning. WWW 2021 【结合强化学习的可解释工作技能推荐】

- On Interpretation and Measurement of Soft Attributes for Recommendation. SIGIR 2021 【推荐软属性的解释和衡量】

- ReXPlug: Explainable Recommendation using Plug-and-Play Language Model. SIGIR 2021 【即插即用语言模型的可解释推荐】

- User-Centric Path Reasoning towards Explainable Recommendation. SIGIR 2021 【以用户为中心的路径推理进行可解释推荐】
- Interactive Path Reasoning on Graph for Conversational Recommendation. KDD 2020 【对话推荐中基于图路径的解释】
- Leveraging Demonstrations for Reinforcement Recommendation Reasoning over Knowledge Graphs. SIGIR 2020 【利用演示增强知识图上的推荐推理】
- Fairness-Aware Explainable Recommendation over Knowledge Graphs. SIGIR 2020 【知识图谱上公平性感知的可解释推荐】
- Temporal Meta-path Guided Explainable Recommendation. WSDM 2021 【时态元路径引导的可解释推荐】
- Neural Collaborative Reasoning. WWW 2021 【深度协同过滤的解释】
- Counterfactual Explainable Recommendation. CIKM 2021【反事实可解释推荐】
- On the Diversity and Explainability of Recommender Systems: A Practical Framework for Enterprise App Recommendation. CIKM 2021【applied paper，推荐系统的多样性和可解释性】
- You Are What and Where You Are: Graph Enhanced Attention Network for Explainable POI Recommendation. CIKM 2021【applied paper，Attention 图神经网络用于可解释推荐】
- XPL-CF: Explainable Embeddings for Feature-based Collaborative Filtering. CIKM 2021【short paper，可解释 Embedding 用于基于特征的协同过滤】
- Reinforcement Learning over Sentiment-Augmented Knowledge Graphs towards Accurate and Explainable Recommendation. WSDM 2022【对情感增强知识图的强化学习以实现准确和可解释的推荐】
- Explanation Guided Contrastive Learning for Sequential Recommendation. CIKM 2022 【序列推荐的解释引导对比学习】
- PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions. KDD 2022 【用户个性化区分】
- Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations. SIGIR 2022 【使用知识图谱为推荐生成崭新的、多样的解释】
- PEVAE: A hierarchical VAE for personalized explainable recommendation.. SIGIR 2022 【利用层次化VAE进行个性化可解释推荐】
- Explainable Session-based Recommendation with Meta-Path Guided Instances and Self-Attention Mechanism. SIGIR 2022 【short paper， 基于元路径指导和自注意力机制的可解释会话推荐】 
- Debiased Explainable Pairwise Ranking from Implicit Feedback. RecSys 2021【去偏的可解释贝叶斯个性化排名】
- EX3: Explainable Attribute-aware Item-set Recommendations. RecSys 2021【基于物品集属性的可解释推荐】
- Do Users Appreciate Explanations of Recommendations? An Analysis in the Movie Domain. RecSys 2021【LBR，电影推荐系统中用户对解释的满意度分析】
- ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations. RecSys 2022【基于原型的可解释协同过滤算法】
- Graph-based Extractive Explainer for Recommendations. WWW 2022 【使用图注意力网络来实现可解释推荐】
- ExpScore: Learning Metrics for Recommendation Explanation. WWW 2022 【可解释推荐评价指标】
- Path Language Modeling over Knowledge Graphs for Explainable Recommendation. WWW 2022 【在知识图谱上学习语言模型，实现推荐和解释】
- Accurate and Explainable Recommendation via Review Rationalization. WWW 2022 【提取评论中的因果关系】
- AmpSum: Adaptive Multiple-Product Summarization towards Improving Recommendation Captions. WWW 2022 【生成商品标题】
- Comparative Explanations of Recommendations. WWW 2022 【可比较的推荐解释】
- Neuro-Symbolic Interpretable Collaborative Filtering for Attribute-based Recommendation. WWW 2022 【以模型为核心的神经符号可解释协同过滤】
- Multi-level Recommendation Reasoning over Knowledge Graphs with Reinforcement Learning. WWW 2022 【基于强化学习的知识图多级推荐推理】

### Long-tail/Cold-start in Recommendations

-   Sequential and Diverse Recommendation with Long Tail. IJCAI 2019. [PDF](https://www.ijcai.org/proceedings/2019/0380.pdf) 【长尾的序列和多样推荐】
-   Content-aware Neural Hashing for Cold-start Recommendation. SIGIR 2020 【内容感知的神经散列用于冷启动推荐】
-   Recommendation for New Users and New Items via Randomized Training and Mixture-of-Experts Transformation. SIGIR 2020 【通过随机训练和专家转型混合对新用户和新物品进行推荐】
-   MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation. KDD 2020 【冷启动推荐的内存增强元优化】
-   Learning Transferrable Parameters for Long-tailed Sequential User Behavior Modeling. KDD 2020 【长尾用户的行为建模】
-   Meta-learning on Heterogeneous Information Networks for Cold-start Recommendation. KDD 2020 【异构信息网络的元学习用于冷启动推荐】
-   Cold-Start Sequential Recommendation via Meta Learner. AAAI 2021 【通过元学习冷启动序列推荐】
-   Personalized Adaptive Meta Learning for Cold-start User Preference Prediction. AAAI 2021 【用于冷启动用户偏好预测的个性化自适应元学习】
-   Task-adaptive Neural Process for User Cold-Start Recommendation. WWW 2021 【用于用户冷启动推荐的任务自适应】
-   A Model of Two Tales: Dual Transfer Learning Framework for Improved Long-tail Item Recommendation. WWW 2021 【双转移学习框架用于改进长尾推荐】
-   FORM: Follow the Online Regularized Meta-Leader for Cold-Start Recommendation. SIGIR 2021 【遵循在线规范化的元领导者进行冷启动推荐】
-   Privileged Graph Distillation for Cold-start Recommendation. SIGIR 2021 【冷启动推荐的特权图蒸馏】
-   Learning to Warm Up Cold Item Embeddings for Cold-start Recommendation with Meta Scaling and Shifting Networks. SIGIR 2021 【元缩放和偏移网络进行冷启动推荐】
-   CATN: Cross-Domain Recommendation for Cold-Start Users via Aspect Transfer Network. SIGIR 2020 【为冷启动用户提供跨域建议】
-   Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction. SIGIR 2021 【点击率预测中冷启动广告】
-   Fairness among New Items in Cold Start Recommender Systems. SIGIR 2021 【冷启动推荐系统中新物品的公平性】
-   Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph. WSDM 2021 【知识图上的伪标记缓解冷启动问题】
-   Adversarial Item Promotion: Vulnerabilities at the Core of Top-N Recommenders that Use Images to Address Cold Start. WWW 2021 【使用图像解决冷启动推荐的漏洞】
-   CMML: Contextual Modulation Meta Learning for Cold-Start Recommendation. CIKM 2021【元学习+冷启动】
-   Reinforcement Learning to Optimize Lifetime Value in Cold-Start Recommendation. CIKM 2021【增强学习+冷启动】
-   Zero Shot on the Cold-Start Problem: Model-Agnostic Interest Learning for Recommender Systems. CIKM 2021【零样本学习+冷启动】
-   Memory Bank Augmented Long-tail Sequential Recommendation. CIKM 2022 【记忆库增强】
-   GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction. CIKM 2022 【GIFT：用于冷启动视频点击率预测的图引导特征迁移】
-   Task-optimized User Clustering based on Mobile App Usage for Cold-start Recommendations. KDD 2022 【用户聚类】
-   A Dynamic Meta-Learning Model for Time-Sensitive Cold-Start Recommendations. AAAI 2022 【元学习动态推荐框架】
-   SMINet: State-Aware Multi-Aspect Interests Representation Network for Cold-Start Users Recommendation. AAAI 2022 【状态感知多方面兴趣网络】
-   Multi-view Denoising Graph Auto-Encoders on Heterogeneous Information Networks for Cold-start Recommendation. KDD 2021 【异构信息网络多视图去噪图自动编码器实现冷启动】
-   A Semi-Personalized System for User Cold Start Recommendation on Music Streaming Apps. KDD 2021 【半个性化的音乐流媒体应用冷启动推荐系统】
-   Socially-aware Dual Contrastive Learning for Cold-Start Recommendation. SIGIR 2022 【short paper，社交感知的双重对比学习】
-   Transform Cold-Start Users into Warm via Fused Behaviors in Large-Scale Recommendation. SIGIR 2022 【short paper，通过融合行为转换冷启动用户】
-   Generative Adversarial Framework for Cold-Start Item Recommendation. SIGIR 2022 【short paper，针对冷启动商品的生成对抗框架】
-   Improving Item Cold-start Recommendation via Model-agnostic Conditional Variational Autoencoder. SIGIR 2022 【short paper，模型无关的自编码器提升商品冷启动推荐】
-   Cold Start Similar Artists Ranking with Gravity-Inspired Graph Autoencoders. RecSys 2021 【图自动编码器冷启动相似艺术家排序问题】
-   Shared Neural Item Representations for Completely Cold Start Problem. RecSys 2021 【通过共享物品表示解决完全冷启动问题】
-   Siamese Neural Networks for Content-based Cold-Start Music Recommendation. RecSys 2021 【LBR，暹罗神经网络进行音乐冷启动推荐】
-   Fast And Accurate User Cold-Start Learning Using Monte Carlo Tree Search. RecSys 2022 【蒙特卡洛树搜索进行用户冷启动学习】
-   M2TRec: Metadata-aware Multi-task Transformer for Large-scale and Cold-start free Session-based Recommendations. RecSys 2022【LBR，基于元数据和多任务 Transformer 的冷启动会话推荐系统】
-   Alleviating Cold-start Problem in CTR Prediction with A Variational Embedding Learning Framework. WWW 2022 【使用变分embedding学习框架缓解 CTR 预测中的冷启动问题】
-   PNMTA: A Pretrained Network Modulation and Task Adaptation Approach for User Cold-Start Recommendation. WWW 2022 【加入编码调制器和预测调制器，使得编码器和预测器可以自适应处理冷启动用户。】
-   KoMen: Domain Knowledge Guided Interaction Recommendation for Emerging Scenarios. WWW 2022 【元学习+图网络】


### Diversity in Recommendation

-   Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity. NIPS 2018. [PDF](https://proceedings.neurips.cc/paper/2018/file/dbbf603ff0e99629dda5d75b6f75f966-Paper.pdf) 【行列式点过程的快速贪婪映射推理提高推荐多样性】
-   PD-GAN: Adversarial Learning for Personalized Diversity-Promoting Recommendation. IJCAI 2019. [PDF](https://www.ijcai.org/proceedings/2019/0537.pdf) 【考虑推荐多样性的对抗性学习】
-   Sequential and Diverse Recommendation with Long Tail. IJCAI 2019. [PDF](https://www.ijcai.org/proceedings/2019/0380.pdf) 【长尾的序列和多样推荐】
-   Diversified Interactive Recommendation with Implicit Feedback. AAAI 2020. [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/5931/5787) 【隐反馈的多样化交互式推荐】
-   ART (Attractive Recommendation Tailor): How the Diversity of Product Recommendations Affects Customer Purchase Preference in Fashion Industry. CIKM 2020. [PDF](https://dl.acm.org/doi/pdf/10.1145/3340531.3412687?casa_token=pLrkqMKPqS4AAAAA:SROCQTKY_rSZVDJa2gTQf5bRGKky_BJEnNOcXXi0A1nYDNO9cBb1FjmIucxg7NN_K42IZA5RGye3XA) 【多样性对时尚行业客户购买偏好的影响】
-   A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks. KDD 2020 【贝叶斯图卷积神经网络】
-   Enhancing Recommendation Diversity using Determinantal Point Processes on Knowledge Graphs. SIGIR 2020. [PDF](https://dl.acm.org/doi/pdf/10.1145/3397271.3401213?casa_token=WLxAwJzJFhwAAAAA:k_i4GlcKpa6xxtdxPfZLuyyVZ_cp61I4C6pf361chyAYvYzWFQeZJjNQ95POg2UiKPECCcaF-hz9YA) 【知识图谱上使用行列式点过程增强推荐多样性】
-   A Hybrid Bandit Framework for Diversified Recommendation. AAAI 2021. [PDF](https://arxiv.org/pdf/2012.13245) 【多样性推荐的混合 bandit 框架】
-   P-Companion:  A Principled Framework for Diversified Complementary Product Recommendation. CIKM 2021. [PDF](https://dl.acm.org/doi/pdf/10.1145/3340531.3412732) 【多元化补充产品推荐的原则框架】
-   Sliding Spectrum Decomposition for Diversified Recommendation. KDD 2021. [PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467108?casa_token=-ODzWqfs760AAAAA:4NO7u-6ARd9lVBIuyiqCN2WKK0QCKbQUCW532MVJduRVV_0dawflyQ1y8YUXaKxMo342KmqF7Q) 【滑动频谱分解以实现多样化推荐】
-   Enhancing Domain-Level and User-Level Adaptivity in Diversified Recommendation.  SIGIR 2021. [PDF](https://dl.acm.org/doi/abs/10.1145/3404835.3462957) 【在多元化推荐中提高域级和用户级适应性】
-   Graph Meta Network for Multi-Behavior Recommendation with Interaction Heterogeneity and Diversity. SIGIR 2021 【结合交互异质性和多样性的多行为推荐图元网络】
-   Modeling Intent Graph for Search Result Diversification. SIGIR 2021 【建模意向网络用于搜索结果多样性】
-   Dynamic Graph Construction for Improving Diversity of Recommendation. RecSys 2021. [PDF](https://dl.acm.org/doi/abs/10.1145/3460231.3478845) 【改进推荐多样性的动态图构建】
-   Towards Unified Metrics for Accuracy and Diversity for Recommender Systems. RecSys 2021. [PDF](https://dl.acm.org/doi/pdf/10.1145/3460231.3474234) 【为推荐系统建立准确性和多样性的统一指标】
-   DGCN: Diversified Recommendation with Graph Convolutional Networks.  WWW 2021. [PDF](http://fi.ee.tsinghua.edu.cn/public/publications/b344fd48-92b0-11eb-96bc-0242ac120003.pdf) 【使用图卷积网络的多样化推荐】
-   Future-Aware Diverse Trends Framework for Recommendation. WWW 2021. [PDF](https://arxiv.org/pdf/2011.00422) 【面向未来的多元化趋势推荐框架】
-   Random Walks with Erasure: Diversifying Personalized Recommendations on Social and Information Networks.  WWW 2021. [PDF](https://arxiv.org/pdf/2102.09635) 【社交网络个性化推荐的多样化】
-   Improving End-to-End Sequential Recommendations with Intent-aware Diversification. CIKM 2020 【意向感知的多样性】
-   Diversified Recommendation Through Similarity-Guided Graph Neural Networks. WWW 2021 【基于相似引导图神经网络的多样化推荐】
-   On the Diversity and Explainability of Recommender Systems: A Practical Framework for Enterprise App Recommendation. CIKM 2021. [PDF](https://dl.acm.org/doi/pdf/10.1145/3459637.3481940?casa_token=x4k2BTtSkmMAAAAA:J-83ktko4ediMOH7qhGal9GmLiop5gLEUgsmffLKfJK2ITuC6m8SDdKvbG4xnttmjH2gCLL66geyog) 【applied paper，推荐系统的多样性和可解释性】
-   Choosing the Best of All Worlds: Accurate, Diverse, and Novel Recommendations through Multi-Objective Reinforcement Learning. WSDM 2022【通过多目标强化学习的准确、多样化和新颖的推荐】
-   Feature-aware Diversified Re-ranking with Disentangled Representations for Relevant Recommendation. KDD 2022 【特征表示解耦去噪】
-   DAWAR: Diversity-aware Web APIs Recommendation for Mashup Creation based on Correlation Graph. SIGIR 2022 【多样化Web API推荐】
-   Mitigating the Filter Bubble while Maintaining Relevance: Targeted Diversification with VAE-based Recommender Systems. SIGIR 2022 【short paper，定向多样化】
-   Diversity vs Relevance: a practical multi-objective study in luxury fashion recommendations. SIGIR 2022 【short paper，奢侈品推荐中的多目标研究】
-   Towards Unified Metrics for Accuracy and Diversity for Recommender Systems. RecSys 2021 【推荐系统准确性和多样性的统一度量】
-   Dynamic Graph Construction for Improving Diversity of Recommendation. RecSys 2021 【LBR，改进推荐系统多样性的动态多样图框架】
-   Solving Diversity-Aware Maximum Inner Product Search Efficiently and Effectively. RecSys 2022 【多样性感知的最大内部产品搜索】

### Denoising in Recommendation

- Denoising Implicit Feedback for Recommendation. WSDM 2021 【推荐隐反馈的去噪】
- The World is Binary: Contrastive Learning for Denoising Next Basket Recommendation. SIGIR 2021 【通过对比学习对下一篮推荐问题去噪】
- Concept-Aware Denoising Graph Neural Network for Micro-Video Recommendation. CIKM 2021【用于微视频推荐的去噪GNN】
- Hierarchical Item Inconsistency Signal learning for Sequence Denoising in Sequential Recommendation. CIKM 2022 【序列推荐中序列去噪的分层项目不一致信号学习】
- Multi-view Denoising Graph Auto-Encoders on Heterogeneous Information Networks for Cold-start Recommendation. KDD 2021 【面向冷启动推荐的异构信息网络多视图去噪图自动编码器】
- Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering. SIGIR 2022 【数据去噪】
- Less is More: Reweighting Important Spectral Graph Features for Recommendation. SIGIR 2022 【评估重要的图谱特征】
- Denoising Time Cycle Modeling for Recommendation. SIGIR 2022 【short paper，去噪时间循环建模】
- Denoising User-aware Memory Network for Recommendation. RecSys 2021 【用户感知记忆网络的去噪】
- Denoising Self-Attentive Sequential Recommendation. RecSys 2022 【基于 Transformer 的序列推荐中自注意力机制的去噪】

### Privacy Protection in Recommendation

- DeepRec: On-device Deep Learning for Privacy-Preserving Sequential Recommendation in Mobile Commerce. WWW 2021 【移动商务的隐私保护】
- FedCDR: Federated Cross-Domain Recommendation for Privacy-Preserving Rating Prediction. CIKM 2022 【FedCDR：用于隐私保护评级预测的联合跨域推荐】
- PipAttack: Poisoning Federated Recommender Systems for Manipulating Item Promotion. WSDM 2022【用于操纵项目促销的中毒联合推荐系统】
- FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling. KDD 2022 【联邦推荐的对抗攻击】
- Privacy-Preserving Synthetic Data Generation for Recommendation. SIGIR 2022 【隐私保护的仿真数据生成】
- Privacy Preserving Collaborative Filtering by Distributed Mediation. RecSys 2021【隐私保护式协同过滤】
- A Payload Optimization Method for Federated Recommender Systems. RecSys 2021 【联邦推荐的有效负载优化方法】
- Stronger Privacy for Federated Collaborative Filtering With Implicit Feedback. RecSys 2021【隐式数据的联邦推荐系统】
- FR-FMSS: Federated Recommendation via Fake Marks and Secret Sharing. RecSys 2021【LBR，跨用户联邦推荐框架】
- Horizontal Cross-Silo Federated Recommender Systems. RecSys 2021【LBR，联邦推荐对多类利益相关者的影响研究】
- Dynamic Global Sensitivity for Differentially Private Contextual Bandits. RecSys 2022 【通过差分私有的上下文 Bandit 算法保护隐私】
- EANA: Reducing Privacy Risk on Large-scale Recommendation Models. RecSys 2022 【降低大规模推荐模型的隐私风险】
- Fairness-aware Federated Matrix Factorization. RecSys 2022 【结合差异隐私技术的公平感知的联邦矩阵分解】
- Toward Fair Federated Recommendation Learning: Characterizing the Inter-Dependence of System and Data Heterogeneity. RecSys 2022 【推荐中公平的联邦学习】
- Poisoning Deep Learning based Recommender Model in Federated Learning Scenarios. IJCAI 2022 【设计针对基于深度学习的推荐模型的攻击】
- Differential Private Knowledge Transfer for Privacy-Preserving Cross-Domain Recommendation. WWW 2022 【通过可微隐私知识迁移实现源域隐私保护的跨域推荐】

### Evaluation of Recommender System

- How Dataset Characteristics Affect the Robustness of Collaborative Recommendation Models. SIGIR 2020 【分析数据集特征如何影响协同推荐模型的稳健性】
- Agreement and Disagreement between True and False-Positive Metrics in Recommender Systems Evaluation. SIGIR 2020 【推荐系统评估中真假正例指标的一致性和不一致性】

- Critically Examining the Claimed Value of Convolutions over User-Item Embedding Maps for Recommender Systems. CIKM 2020 【批判性分析推荐系统的用户物品图卷积的作用】

- On Estimating Recommendation Evaluation Metrics under Sampling. AAAI 2021 【推荐中采样评测指标的分析】

- Beyond Point Estimate Inferring Ensemble Prediction Variation from Neuron Activation Strength in Recommender Systems. WSDM 2021 【推荐系统中神经元激活强度变化】

- Bias-Variance Decomposition for Ranking. WSDM 2021 【排序中偏差-方差的分解】

- Theoretical Understandings of Product Embedding for E-commerce Machine Learning. WSDM 2021 【电子商务机器学习中物品表征的理论理解】
- Measuring Recommendation Explanation Quality: The Conflicting Goals of Explanations. SIGIR 2020 【衡量推荐解释的质量】
- Evaluating Conversational Recommender Systems via User Simulation. KDD 2020 【通过用户仿真评估会话推荐系统】

- On Sampled Metrics for Item Recommendation. KDD 2020 【物品推荐的采样指标】

- On Sampling Top-K Recommendation Evaluation. KDD 2020 【推荐采样的评估】

- Are We Evaluating Rigorously： Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison. RecSys 2020 【可重复评估和公平比较的基准推荐】

- On Target Item Sampling in Offline Recommender System Evaluation. RecSys 2020 【离线推荐系统评估中的目标物品采样】

- Exploiting Performance Estimates for Augmenting Recommendation Ensembles. RecSys 2020 【利用性能估算来增强推荐集合】

- A Method to Anonymize Business Metrics to Publishing Implicit Feedback Datasets. RecSys 2020 【匿名业务指标发布隐反馈数据集的方法】

- Standing in Your Shoes: External Assessments for Personalized Recommender Systems. SIGIR 2021 【个性化推荐系统的外部评估】

- librec-auto: A Tool for Recommender Systems Experimentation. CIKM 2021 【librec-auto：推荐系统实验的工具】
- RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms. CIKM 2021 【RecBole：迈向统一、全面和高效的推荐算法框架】
- RecBole 2.0: Towards a More Up-to-Date Recommendation Library. CIKM 2022 【RecBole 2.0：与时俱进的推荐库】
- Best practices for top-N recommendation evaluation: Candidate set sampling and Statistical inference techniques. CIKM 2022 【候选集抽样和统计推断技术】
- KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems. CIKM 2022 【用于评估推荐系统的完全观察数据集和见解】
- FPAdaMetric: False-Positive-Aware Adaptive Metric Learning for Session-Based Recommendation. AAAI 2022 【度量学习正则化】
- Evaluation of Herd Behavior Caused by Population-scale Concept Drift in Collaborative Filtering. SIGIR 2022 【short paper】
- Evaluating Off-Policy Evaluation: Sensitivity and Robustness. RecSys 2021 【离线评估的可解释性评估】
- Online Evaluation Methods for the Causal Effect of Recommendations. RecSys 2021 【推荐中因果效应的在线评估方法】
- Towards Unified Metrics for Accuracy and Diversity for Recommender Systems. RecSys 2021 【推荐系统准确性和多样性的统一度量】
- Values of Exploration in Recommender Systems. RecSys 2021 【基于强化学习推荐算法的用户探索和指标评测】
- A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models. RecSys 2021 【Reproducibility paper，序列推荐模型评测策略的研究】
- Quality Metrics in Recommender Systems: Do We Calculate Metrics Consistently?. RecSys 2021 【LBR，推荐系统评估指标的再思考】
- Don't recommend the obvious: estimate probability ratios. RecSys 2022 【通过拟合逐点的互信息来改进序列推荐的流行度采样指标】
- RADio – Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. RecSys 2022【衡量新闻推荐规范化多样性的度量框架】
- A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation. RecSys 2022【Reproducibility paper，BERT4Rec 结果的系统回顾与可复现性研究】
- Revisiting the Performance of iALS on Item Recommendation Benchmarks. RecSys 2022【Reproducibility paper，隐式交替最小二乘法 iALS 基准表现的再思考】
- Measuring Commonality in Recommendation of Cultural Content: Recommender Systems to Enhance Cultural Citizenship. RecSys 2022【LBR，通用性作为文化内容推荐的度量】
- Multiobjective Evaluation of Reinforcement Learning Based Recommender Systems. RecSys 2022【LBR，基于强化学习的推荐系统的多目标评价】
- Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems. NIPS 2022 【推荐系统的大规模多用途基准数据集】
- On the Generalizability and Predictabilityof Recommender Systems. NIPS 2022 【提出一种元学习方法，针对一个全新的，未见过的数据集选择最优算法和参数】

### Other Topics

- Multi-granularity Fatigue in Recommendation. CIKM 2022 【推荐中的多粒度疲劳】
- A Multi-Interest Evolution Story: Applying Psychology in Query-based Recommendation for Inferring Customer Intention. CIKM 2022 【在基于查询的推荐中应用心理学以推断客户意图】
- Improving Text-based Similar Product Recommendation for Dynamic Product Advertising at Yahoo. CIKM 2022 【改进雅虎动态产品广告的基于文本的相似产品推荐】
- E-Commerce Promotions Personalization via Online Multiple-Choice Knapsack with Uplift Modeling. CIKM 2022 【在线的电子商务促销个性化】
- Hierarchical Imitation Learning via Subgoal Representation Learning for Dynamic Treatment Recommendation. WSDM 2022【基于动态治疗推荐的子目标表示学习的分层模仿学习】
- PLdFe-RR:Personalized Long-distance Fuel-efficient Route Recommendation Based On Historical Trajectory. WSDM 2022【基于历史轨迹的个性化长途省油路线推荐】
- The Datasets Dilemma: How Much Do We Really Know About Recommendation Datasets?. WSDM 2022【数据集困境：我们对推荐数据集真正了解多少？】
- Obtaining Calibrated Probabilities with Personalized Ranking Models. AAAI 2022 【提出排序模型的两种校准方法】
- Addressing Unmeasured Confounder for Recommendation with Sensitivity Analysis. KDD 2022 【混淆因子的鲁棒性】
- Learning Elastic Embeddings for Customizing On-Device Recommenders. KDD 2021 【定制设备上的弹性 embedding】
- Learning to Embed Categorical Features without Embedding Tables for Recommendation. KDD 2021 【无 embedding 表的推荐系统特征建模】
- Preference Amplification in Recommender Systems. KDD 2021 【推荐系统中的偏好放大】
- Where are we in embedding spaces? A Comprehensive Analysis on Network Embedding Approaches for Recommender System. KDD 2021 【推荐系统中网络嵌入方法的综合分析】
- User-Aware Multi-Interest Learning for Candidate Matching in Recommenders. SIGIR 2022 【使用用户多兴趣学习进行候选匹配】
- User-controllable Recommendation Against Filter Bubbles. SIGIR 2022 【用户可控的推荐】
- Rethinking Correlation-based Item-Item Similarities for Recommender Systems. SIGIR 2022 【short paper，反思基于关系的商品相似度】
- ReLoop: A Self-Correction Learning Loop for Recommender Systems. SIGIR 2022 【short paper，推荐系统中的自修正循环学习】
- Towards Results-level Proportionality for Multi-objective Recommender Systems. SIGIR 2022 【short paper，动量对比方法实现结果均衡的多目标推荐系统】
- Recommender Systems and Algorithmic Hate. RecSys 2022 【LBR，对用户方案推荐系统算法的探究性工作】
- The Effect of Feedback Granularity on Recommender Systems Performance. RecSys 2022 【LBR，评分和反馈粒度对推荐性能的影响】
- The trade-offs of model size in large recommendation models : A 10000 ×× compressed criteo-tb DLRM model (100 GB parameters to mere 10MB). NIPS 2022 【推荐模型压缩】
- DreamShard: Generalizable Embedding Table Placement for Recommender Systems. NIPS 2022 【推荐系统的通用embedding表】

Technique
---------

### Pre-training in Recommender System

-   S$^3$-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. CIKM 2020 【基于互信息最大化的序列推荐自监督学习】
-   Contrastive Pre-Training of GNNs on Heterogeneous Graphs. CIKM 2021【图神经网络的对比预训练】
-   Self-supervised Learning for Large-scale Item Recommendations. arXiv 2020 【大规模物品推荐的自监督学习】
-   U-BERT Pre-Training User Representations for Improved Recommendation. AAAI 2021 【预训练用户表示】
-   Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation. WSDM 2021 【冷启动用户和物品表示的预训练图神经网络】
-   Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer. SIGIR 2021 【反向预训练 Transformer 增强序列推荐】
-   Self-supervised Graph Learning for Recommendation. SIGIR 2021 【推荐的自监督图学习】
-   Pre-training Graph Transformer with Multimodal Side Information for Recommendation. ACM MM 2021 【多模态侧信息的预训练图 Transformer 用于推荐】
-   SelfCF: A Simple Framework for Self-supervised Collaborative Filtering. arXiv 2021 【自监督协同过滤】
-   UPRec: User-Aware Pre-training for Recommender Systems. arXiv 2021 【推荐系统用户感知的预训练】
-   Curriculum Pre-Training Heterogeneous Subgraph Transformer for Top-N Recommendation. arXiv 2021 【用于 Top-N 推荐的课程预训练异构子图 Transformer】
-   One4all User Representation for Recommender Systems in E-commerce. arXiv 2021 【电子商务中用户表示的预训练】
-   Graph Neural Pre-training for Enhancing Recommendations using Side Information. arXiv 2021 【侧信息增强推荐的图神经预训练】
-   GBERT: Pre-training User representations for Ephemeral Group Recommendation. CIKM 2022 【为临时组推荐预训练用户表示】
-   Temporal Contrastive Pre-Training for Sequential Recommendation. CIKM 2022 【时序推荐的时间对比预训练】
-   TwHIN: Embedding the Twitter Heterogeneous Information Network for Personalized Recommendation. KDD 2022 【推特知识图谱增强】
-   Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). RecSys 2022 【通用的预训练、个性化提示和预测范式建模推荐任务】

### Reinforcement Learning in Recommendation

-   MaHRL: Multi-goals Abstraction based Deep Hierarchical Reinforcement Learning for Recommendations. SIGIR 2020 【基于多目标抽象的层次强化学习】
-   Interactive Recommender System via Knowledge Graph-enhanced Reinforcement Learning. SIGIR 2020 【基于知识图增强强化学习的交互式推荐系统】
-   Joint Policy-Value Learning for Recommendation. KDD 2020 【联合 policy-value 的强化学习推荐】
-   BLOB: A Probabilistic Model for Recommendation that Combines Organic and Bandit Signals. KDD 2020 【有机和 bandit 信号的推荐概率模型】
-   Learning to Collaborate in Multi-Module Recommendation via Multi-Agent Reinforcement Learning without Communication. RecSys 2020 【多代理强化学习进行多模块推荐】
-   Exploring Clustering of Bandits for Online Recommendation System. RecSys 2020 【在线推荐系统中的 bandit 聚类研究】
-   Contextual User Browsing Bandits for Large-Scale Online Mobile Recommendation. RecSys 2020 【大规模在线移动推荐的上下文用户浏览 bandit】
-   Keeping Dataset Biases out of the Simulation : A Debiased Simulator for Reinforcement Learning based Recommender Systems. RecSys 2020 【基于强化学习的去偏模拟器】
-   Leveraging Demonstrations for Reinforcement Recommendation Reasoning over Knowledge Graphs. SIGIR 2020 【利用演示增强知识图上的推荐推理】
-   KERL: A Knowledge-Guided Reinforcement Learning Model for Sequential Recommendation. SIGIR 2020 【序列推荐中知识增强的强化学习模型】
-   A Hybrid Bandit Framework for Diversified Recommendation. AAAI 2021 【多元化推荐的混合 Bandit 框架】
-   RLNF: Reinforcement Learning based Noise Filtering for Click-Through Rate Prediction. SIGIR 2021 【基于强化学习的点击率预测噪声滤波】
-   Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning. SIGIR 2021 【基于图强化学习的对话推荐】
-   Diverse User Preference Elicitation with Multi-Armed Bandits. WSDM 2021 【多臂老虎机的多种用户偏好启发】
-   Explore, Filter and Distill: Distilled Reinforcement Learning in Recommendation. CIKM 2021【applied paper，推荐中的蒸馏强化学习】
-   Reinforcement Learning with a Disentangled Universal Value Function for Item Recommendation. AAAI 2021 【使用统一价值函数的强化学习】
-   User Response Models to Improve a REINFORCE Recommender System. WSDM 2021 【用户响应模型以改进强化推荐系统】
-   Cost-Effective and Interpretable Job Skill Recommendation with Deep Reinforcement Learning. WWW 2021 【结合强化学习的可解释工作技能推荐】
-   A Multi-Agent Reinforcement Learning Framework for Intelligent Electric Vehicle Charging Recommendation. WWW 2021 【智能电动汽车充电推荐的多智能体强化学习框架】
-   Reinforcement Recommendation with User Multi-aspect Preference. WWW 2021 【用户多方面偏好的强化推荐】
-   A Peep into the Future: Adversarial Future Encoding in Recommendation. WSDM 2022【推荐中的对抗性未来编码】
-   Toward Pareto Efficient Fairness-Utility Trade-off in Recommendation through Reinforcement Learning. WSDM 2022【通过强化学习在推荐中实现帕累托高效的公平-效用权衡】
-   Supervised Advantage Actor-Critic for Recommender Systems. WSDM 2022【推荐系统的监督优势Actor-Critic】
-   Surrogate for Long-Term User Experience in Recommender Systems. KDD 2022 【用户长期行为建模】
-   AutoShard: Automated Embedding Table Sharding for Recommender Systems. KDD 2022 【Embedding划分代价预估】
-   Multi-Task Fusion via Reinforcement Learning for Long-Term User Satisfaction in Recommender Systems. KDD 2022 【Batch RL建模MTF】
-   Modeling Attrition in Recommender Systems with Departing Bandits. AAAI 2022 【新颖的多臂老虎机】
-   Context Uncertainty in Contextual Bandits with Applications to Recommender Systems. AAAI 2022 【新型循环神经网络】
-   Learning the Optimal Recommendation from Explorative Users. AAAI 2022 【更现实的用户行为模型】
-   Offline Interactive Recommendation with Natural-Language Feedback. AAAI 2022 【对抗训练离线策略】
-   Locality-Sensitive State-Guided Experience Replay Optimization for Sparse-Reward in Online Recommendation. SIGIR 2022 【在线推荐中的稀疏奖励问题】
-   Multi-Agent RL-based Information Selection Model for Sequential Recommendation. SIGIR 2022 【多智能体信息选择】
-   Rethinking Reinforcement Learning for Recommendation: A Prompt Perspective. SIGIR 2022 【从提示视角看用于推荐的强化学习】
-   Doubly-Adaptive Reinforcement Learning for Cross-Domain Interactive Recommendation. SIGIR 2022 【双重适应的强化学习】
-   MGPolicy: Meta Graph Enhanced Off-policy Learning for Recommendations. SIGIR 2022 【元图增强的离线策略学习】
-   Value Penalized Q-Learning for Recommender Systems. SIGIR 2022 【short paper，值惩罚的Q-Learning】
-   Revisiting Interactive Recommender System with Reinforcement Learning. SIGIR 2022 【short paper，回顾基于强化学习的交互推荐】
-   Designing Online Advertisements via Bandit and Reinforcement Learning. RecSys 2021 【通过 bandit 和强化学习方法设计在线广告】
-   Partially Observable Reinforcement Learning for Dialog-based Interactive Recommendation. RecSys 2021 【将强化学习用于基于对话框的交互式推荐任务】
-   Values of Exploration in Recommender Systems. RecSys 2021 【基于强化学习推荐算法的用户探索和指标评测】
-   Sequence Adaptation via Reinforcement Learning in Recommender Systems. RecSys 2021 【LBR，用强化学习方法进行序列推荐】
-   Burst-induced Multi-Armed Bandit for Learning Recommendation. RecSys 2021 【触发式多臂老虎机问题】
-   Pessimistic Reward Models for Off-Policy Learning in Recommendation. RecSys 2021 【改进离线推荐中 Bandit 方法奖励建模】
-   Top-K Contextual Bandits with Equity of Exposure. RecSys 2021 【曝光感知的 Bandit 算法】
-   Off-Policy Actor Critic for Recommender Systems. RecSys 2022 【离线演员-评论家强化学习算法缓解分布偏差问题】
-   Multiobjective Evaluation of Reinforcement Learning Based Recommender Systems. RecSys 2022【LBR，基于强化学习的推荐系统的多目标评价】
-   Dynamic Global Sensitivity for Differentially Private Contextual Bandits. RecSys 2022 【通过差分私有的上下文 Bandit 算法保护隐私】
-   Identifying New Podcasts with High General Appeal Using a Pure Exploration Infinitely-Armed Bandit Strategy. RecSys 2022 【通过 Bandit 策略进行播客推荐】
-   MINDSim: User Simulator for News Recommenders. WWW 2022 【用户模拟+新闻推荐】
-   Multi-level Recommendation Reasoning over Knowledge Graphs with Reinforcement Learning. WWW 2022 【基于强化学习的知识图多级推荐推理】
-   Multiple Choice Questions based Multi-Interest Policy Learning for Conversational Recommendation. WWW 2022 【基于多兴趣的对话推荐策略学习】
-   Off-policy Learning over Heterogeneous Information for Recommendation. WWW 2022 【异构信息离线学习推荐】

### Knowledge Distillation in Recommendation

-   Privileged Features Distillation at Taobao Recommendations. KDD 2020 【淘宝推荐的特征蒸馏】
-   DE-RRD: A Knowledge Distillation Framework for Recommender System. CIKM 2020 【推荐系统的知识蒸馏框架】

-   Bidirectional Distillation for Top-K Recommender System. WWW 2021 【推荐系统的双向蒸馏】
-   A General Knowledge Distillation Framework for Counterfactual Recommendation via Uniform Data. SIGIR 2020 【反事实推荐的知识蒸馏框架】
-   Graph Structure Aware Contrastive Knowledge Distillation for Incremental Learning in Recommender Systems. CIKM 2021【short paper，推荐系统中用于增量学习的图结构感知的对比知识蒸馏】
-   Target Interest Distillation for Multi-Interest Recommendation. CIKM 2022 【多兴趣推荐的目标兴趣蒸馏】
-   Cross-Task Knowledge Distillation in Multi-Task Recommendation. AAAI 2022 【跨任务知识蒸馏】
-   Topology Distillation for Recommender System. KDD 2021 【基于拓扑蒸馏的推荐系统】
-   On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation. SIGIR 2022 【自监督知识蒸馏】
-   Position Awareness Modeling with Knowledge Distillation for CTR Prediction. RecSys 2022 【LBR，位置感知的知识提取框架】
-   UKD: Debiasing Conversion Rate Estimation via Uncertainty-regularized Knowledge Distillation. WWW 2022 【基于不确定性正则化和知识提取的转化率估计去偏】

### Federated Learning in Recommendation

-   FedFast Going Beyond Average for Faster Training of Federated Recommender Systems. KDD 2020 【联邦推荐系统】
-   Federated Collaborative Transfer for Cross-Domain Recommendation. SIGIR 2021 【跨域推荐的联邦协作转换】
-   FedCDR: Federated Cross-Domain Recommendation for Privacy-Preserving Rating Prediction. CIKM 2022 【FedCDR：用于隐私保护评级预测的联合跨域推荐】
-   PipAttack: Poisoning Federated Recommender Systems for Manipulating Item Promotion. WSDM 2022【用于操纵项目促销的中毒联合推荐系统】
-   FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling. KDD 2022 【联邦推荐的对抗攻击】
-   A Payload Optimization Method for Federated Recommender Systems. RecSys 2021 【联邦推荐的有效负载优化方法】
-   Stronger Privacy for Federated Collaborative Filtering With Implicit Feedback. RecSys 2021【隐式数据的联邦推荐系统】
-   FR-FMSS: Federated Recommendation via Fake Marks and Secret Sharing. RecSys 2021【LBR，跨用户联邦推荐框架】
-   Horizontal Cross-Silo Federated Recommender Systems. RecSys 2021【LBR，联邦推荐对多类利益相关者的影响研究】
-   Fairness-aware Federated Matrix Factorization. RecSys 2022 【结合差异隐私技术的公平感知的联邦矩阵分解】
-   Toward Fair Federated Recommendation Learning: Characterizing the Inter-Dependence of System and Data Heterogeneity. RecSys 2022 【推荐中公平的联邦学习】
-   Poisoning Deep Learning based Recommender Model in Federated Learning Scenarios. IJCAI 2022 【设计针对基于深度学习的推荐模型的攻击】

### GNN in Recommendation

-   Neural Graph Collaborative Filtering. SIGIR 2019 【神经图协同过滤】
-   A Neural Influence Diffusion Model for Social Recommendation. SIGIR 2019 【社交推荐的影响扩散模型】
-   Graph Neural Networks for Social Recommendation. WWW 2019 【社交推荐的图神经网络】
-   Knowledge Graph Convolutional Networks for Recommender Systems. WWW 2019 【推荐系统的知识图谱卷积网络】
-   KGAT: Knowledge Graph Attention Network for Recommendation. KDD 2019 【推荐的知识图谱注意力机制网络】
-   Session-based recommendation with graph neural networks. AAAI 2019 【基于图神经网络的会话推荐】
-   Graph contextualized self-attention network for session-based recommendation. IJCAI 2019 【针对会话推荐的图上下文自注意力网络】
-   Session-based social recommendation via dynamic graph attention networks. WSDM 2019 【通过动态图注意力网络进行会话的社交推荐】
-   Bundle Recommendation with Graph Convolutional Networks. SIGIR 2020 【基于图卷积网络的捆绑推荐 】
-   Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction. WWW 2019 【CTR 预测中基于图神经网络的特征交互建模】
-   LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020 【推荐中的简化图神经网络】
-   Neighbor Interaction Aware Graph Convolution Networks for Recommendation. SIGIR 2020 【邻域交互感知图卷积网络】
-   GAG: Global Attributed Graph Neural Network for Streaming Session-based Recommendation. SIGIR 2020 【基于流会话推荐的全局属性图神经网络】
-   Next-item Recommendation with Sequential Hypergraphs. SIGIR 2020 【序列推荐中的超图】
-   Global Context Enhanced Graph Neural Networks for Session-based Recommendation. SIGIR 2020 【上下文增强的图神经网络】
-   CKAN: Collaborative Knowledge-aware Attentive Network for Recommender Systems. SIGIR 2020 【推荐系统的协同知识感知注意力网络】
-   Attentional Graph Convolutional Networks for Knowledge Concept Recommendation in MOOCs in a Heterogeneous View. SIGIR 2020 【MOOC 知识概念推荐的注意力图卷积网络】
-   Joint Item Recommendation and Attribute Inference: An Adaptive Graph Convolutional Network Approach. SIGIR 2020 【结合物品推荐和属性推理的自适应图卷积网络】
-   Multi-behavior Recommendation with Graph Convolution Networks. SIGIR 2020 【基于图卷积网络的多行为推荐】
-   Hierarchical Fashion Graph Network for Personalized Outfit Recommendation. SIGIR 2020
-   A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks. KDD 2020 【贝叶斯图卷积神经网络】
-   Dual Channel Hypergraph Collaborative Filtering. KDD 2020 【双通道超图的协同过滤】
-   Handling Information Loss of Graph Neural Networks for Session-based Recommendation. KDD 2020 【基于会话推荐的图神经网络信息丢失处理】
-   Star Graph Neural Networks for Session-based Recommendation. CIKM 2020 【基于会话推荐的星形图神经网络】
-   TGCN Tag Graph Convolutional Network for Tag-Aware Recommendation. CIKM 2020 【标签图卷积网络】
-   DisenHAN Disentangled Heterogeneous Graph Attention Network for Recommendation. CIKM 2020 【异构图注意网络】
-   STP-UDGAT Spatial-Temporal-Preference User Dimensional Graph Attention Network for Next POI Recommendation. CIKM 2020 【POI 推荐的时空偏好用户维图注意力网络】
-   Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation. AAAI 2021 【自监督超图卷积网络】
-   Sequential Recommendation with Graph Convolutional Networks. SIGIR 2021 【引入图卷积网络的序列推荐】
-   Structured Graph Convolutional Networks with Stochastic Masks for Recommender Systems. SIGIR 2021 【推荐系统随机掩模的结构化图卷积网络】
-   Adversarial-Enhanced Hybrid Graph Network for User Identity Linkage. SIGIR 2021 【用户身份链接的对抗性增强混合图网络】
-   Should Graph Convolution Trust Neighbors? A Simple Causal Inference Method. SIGIR 2021 【因果推断方法分析图卷积网络】
-   A Graph-Convolutional Ranking Approach to Leverage the Relational Aspects of User-Generated Content. SIGIR 2021 【图卷积排序方法】
-   Neural Graph Matching based Collaborative Filtering. SIGIR 2021 【基于神经图匹配的协同过滤】
-   Modeling Intent Graph for Search Result Diversification. SIGIR 2021 【建模意向网络用于搜索结果多样性】
-   User-Centric Path Reasoning towards Explainable Recommendation. SIGIR 2021 【以用户为中心的路径推理进行可解释推荐】
-   Joint Knowledge Pruning and Recurrent Graph Convolution for News Recommendation. SIGIR 2021 【新闻推荐中的联合知识剪枝和递归图卷积】
-   Privileged Graph Distillation for Cold-start Recommendation. SIGIR 2021 【冷启动推荐的特权图蒸馏】
-   Decoupling Representation Learning and Classification for GNN-based Anomaly Detection. SIGIR 2021 【基于 GNN 的异常检测】
-   Meta-Inductive Node Classification across Graphs. SIGIR 2021 【跨图的元归纳节点分类】
-   Graph Meta Network for Multi-Behavior Recommendation with Interaction Heterogeneity and Diversity. SIGIR 2021 【结合交互异质性和多样性的多行为推荐图元网络】
-   AdsGNN: Behavior-Graph Augmented Relevance Modeling in Sponsored Search. SIGIR 2021 【赞助搜索中的行为图增强建模】
-   Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization. SIGIR 2021 【通过互信息最大化进行图协同过滤】
-   WGCN: Graph Convolutional Networks with Weighted Structural Features. SIGIR 2021 【结合加权结构特征的图卷积网络】
-   Learning Representations of Inactive Users: A Cross Domain Approach with Graph Neural Networks. CIKM 2021【short paper，跨领域方法结合图神经网络用于学习非活跃用户表示】
-   Heterogeneous Graph Augmented Multi-Scenario Sharing Recommendation with Tree-Guided Expert Networks. WSDM 2021 【基于树引导专家网络的异构图增强多场景共享推荐】
-   Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation. WSDM 2021 【冷启动用户和物品表示的预训练图神经网络】
-   Diversified Recommendation Through Similarity-Guided Graph Neural Networks. WWW 2021 【基于相似引导图神经网络的多样化推荐】
-   DGCN: Diversified Recommendation with Graph Convolutional Networks.  WWW 2021. [PDF](http://fi.ee.tsinghua.edu.cn/public/publications/b344fd48-92b0-11eb-96bc-0242ac120003.pdf) 【使用图卷积网络的多样化推荐】
-   Interest-aware Message-Passing GCN for Recommendation. WWW 2021 【兴趣感知的图卷积网络】
-   HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering. WWW 2021 【双曲图卷积网络】
-   Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. WWW 2021 【用于社交推荐的自监督多通道超图卷积网络】
-   RetaGNN: Relational Temporal Attentive Graph Neural Networks for Holistic Sequential Recommendation. WWW 2021 【关系时间注意的图神经网络】
-   Learning Intents behind Interactions with Knowledge Graph for Recommendation. WWW 2021 【知识增强推荐的意图学习】
-   A Knowledge-Aware Recommender with Attention-Enhanced Dynamic Convolutional Network. CIKM 2021【动态卷积用于知识感知的推荐】
-   Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation. CIKM 2021【用于群组推荐的自监督超图学习】
-   Conditional Graph Attention Networks for Distilling and Refining Knowledge Graphs in Recommendation. CIKM 2021【GNN+KG+RS】
-   Contrastive Pre-Training of GNNs on Heterogeneous Graphs. CIKM 2021【图神经网络的对比预训练】
-   Entity-aware Collaborative Relation Network with Knowledge Graph for Recommendation. CIKM 2021【short paper，KG+RS】
-   Incremental Graph Convolutional Network for Collaborative Filtering. CIKM 2021【增量图卷积神经网络用于协同过滤】
-   Self-supervised Graph Learning for Recommendation. SIGIR 2021 【推荐的自监督图学习】
-   Self-Augmented Recommendation with Hypergraph Contrastive Collaborative Filtering. SIGIR 2022 【超图上的对比学习】
-   Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering. SIGIR 2022 【图协同过滤在准确度和新颖度上的表现】
-   A Review-aware Graph Contrastive Learning Framework for Recommendation. SIGIR 2022 【考虑评论的图对比学习】
-   Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation. SIGIR 2022 【简单的图对比学习方法】
-   Spatiotemporal-aware Session-based Recommendation with Graph Neural Networks. CIKM 2022 【使用图神经网络的时空感知基于会话的推荐】
-   CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space. SIGIR 2022 【short paper，在一致表示空间上的简单有效会话推荐】
-   Enhancing Sequential Recommendation with Graph Contrastive Learning. IJCAI 2022 【用于序列推荐的图对比学习】
-   GSL4Rec: Session-based Recommendations with Collective Graph Structure Learning and Next Interaction Prediction. WWW 2022 【图结构学习+推荐】
-   DSKReG: Differentiable Sampling on Knowledge Graph for Recommendation with Relational GNN. CIKM 2021【short paper，用于推荐的知识图谱采样】
-   UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation. CIKM 2021【GNN+RS】
-   How Powerful is Graph Convolution for Recommendation? CIKM 2021【GNN+RS】
-   Concept-Aware Denoising Graph Neural Network for Micro-Video Recommendation. CIKM 2021【用于微视频推荐的去噪GNN】
-   Graph Logic Reasoning for Recommendation and Link Prediction. WSDM 2022【用于推荐和链接预测的图逻辑推理】
-   Heterogeneous Global Graph Neural Networks for Personalized Session-based Recommendation. WSDM 2022【用于个性化基于会话的推荐的异构全局图神经网络】
-   Profiling the Design Space for Graph Neural Networks based Collaborative Filtering. WSDM 2022【分析基于图神经网络的协同过滤的设计空间】
-   Community Trend Prediction on Heterogeneous Graph in E-commerce. WSDM 2022【电子商务异构图的社区趋势预测】
-   Joint Learning of E-commerce Search and Recommendation with A Unified Graph Neural Network. WSDM 2022【电子商务搜索和推荐与统一图神经网络的联合学习】
-   Modeling Scale-free Graphs with Hyperbolic Geometry for Knowledge-aware Recommendation. WSDM 2022【使用双曲几何建模无标度图以进行知识感知推荐】
-   Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation. CIKM 2022 【大规模推荐的神经相似度度量下的近似最近邻搜索】
-   Automatic Meta-Path Discovery for Effective Graph-Based Recommendation. CIKM 2022 【基于图的有效推荐的自动元路径发现】
-   HySAGE: A Hybrid Static and Adaptive Graph Embedding Network for Context-Drifting Recommendations. CIKM 2022 【用于上下文漂移推荐的混合静态和自适应图嵌入网络】
-   Multi-Aggregator Time-Warping Heterogeneous Graph Neural Network for Personalized Micro-video Recommendation. CIKM 2022 【用于个性化微视频推荐的多聚合器时间扭曲异构图神经网络】
-   PlatoGL: Effective and Scalable Deep Graph Learning System for Graph-enhanced Real-Time Recommendation. CIKM 2022 【用于图增强实时推荐的有效且可扩展的深度图学习系统】
-   SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation. CIKM 2022 【用于推荐的简化图卷积范式】
-   The Interaction Graph Auto-encoder Network Based on Topology-aware for Transferable Recommendation. CIKM 2022 【基于拓扑感知的可迁移推荐交互图自动编码器网络】
-   Tiger: Transferable Interest Graph Embedding for Domain-Level Zero-Shot Recommendation. CIKM 2022 【用于域级零样本推荐的可迁移兴趣图嵌入】
-   Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation. KDD 2022 【多面量化增强】
-   User-Event Graph Embedding Learning for Context-Aware Recommendation. KDD 2022 【CTR 任务引入用户-事件图进行表示学习】
-   Self-Supervised Hypergraph Transformer for Recommender Systems. KDD 2022 【自监督超图增强用户表示】
-   Friend Recommendations with Self-Rescaling Graph Neural Networks. KDD 2022 【自放缩解决尺度失真】
-   Low-Pass Graph Convolutional Network for Recommendation. AAAI 2022 【提出了低通协同滤波器】
-   Multi-View Intent Disentangle Graph Networks for Bundle Recommendation. AAAI 2022 【多意图捆绑推荐】
-   MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems. KDD 2021 【高效图神经网络训练】
-   Multi-view Denoising Graph Auto-Encoders on Heterogeneous Information Networks for Cold-start Recommendation. KDD 2021 【面向冷启动推荐的异构信息网络多视图去噪图自动编码器】
-   Reinforced Anchor Knowledge Graph Generation for News Recommendation Reasoning. KDD 2021 【新闻推荐推理的增强锚点知识图生成】
-   Detecting Arbitrary Order Beneficial Feature Interactions for Recommender Systems. KDD 2022 【超图捕获任意阶交互】
-   Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering. SIGIR 2022 【数据去噪】
-   Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation. SIGIR 2022 【多任务图循环网络】
-   An Attribute-Driven Mirroring Graph Network for Session-based Recommendation. SIGIR 2022 【特征驱动的反射图网络】
-   Co-clustering Interactions via Attentive Hypergraph Neural Network. SIGIR 2022 【超图神经网络聚类交互】
-   Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System. SIGIR 2022 【多级交叉视图的对比学习】
-   Graph Trend Filtering Networks for Recommendation. SIGIR 2022 【图趋势过滤网络】
-   Knowledge Graph Contrastive Learning for Recommendation. SIGIR 2022 【知识图谱上的对比学习】
-   EFLEC: Efficient Feature-LEakage Correction in GNN based Recommendation Systems. SIGIR 2022 【short paper，高效的特征泄露修正】
-   DH-HGCN: Dual Homogeneity Hypergraph Convolutional Network for Multiple Social Recommendations. SIGIR 2022 【short paper，双同质超图卷积网络】
-   Enhancing Hypergraph Neural Networks with Intent Disentanglement for Session-based RecommendationSIGIR 2022 【short paper，意图解耦增强超图神经网络】
-   DAGNN: Demand-aware Graph Neural Networks for Session-based Recommendation. SIGIR 2022 【short paper， 需求感知的图神经网络】
-   Global and Personalized Graphs for Heterogeneous Sequential Recommendation by Learning Behavior Transitions and User Intentions. RecSys 2022 【异构序列推荐中通过全局和个性化的图建模学习行为转换和用户意图】
-   TinyKG: Memory-Efficient Training Framework for Knowledge Graph Neural Recommender Systems. RecSys 2022 【知识图神经推荐系统的内存高效训练框架】
-   Streaming Session-Based Recommendation: When Graph Neural Networks meet the Neighborhood. RecSys 2022【Reproducibility paper，图神经网络解决流会话推荐问题】
-   Heterogeneous Interactive Snapshot Network for Review-Enhanced Stock Profiling and Recommendation. IJCAI 2022 【评论增强的股票分析和推荐】
-   Self-supervised Graph Neural Networks for Multi-behavior Recommendation. IJCAI 2022 【GNN + 多行为推荐】
-   RecipeRec: A Heterogeneous Graph Learning Model for Recipe Recommendation. IJCAI 2022 【用于食谱推荐的新型异构图学习模型】
-   Graph Convolution Network based Recommender Systems: Learning Guarantee and Item Mixture Powered Strategy. NIPS 2022 【基于图卷积网络的推荐系统：学习保证和商品混合驱动策略】
-   Contrastive Graph Structure Learning via Information Bottleneck for Recommendation. NIPS 2022 【基于信息瓶颈的对比图结构学习】
-   Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW 2022 【通过邻居节点间的对比学习来改善图协同过滤】
-   Revisiting Graph based Social Recommendation: A Distillation Enhanced Social Graph Network. WWW 2022 【使用知识蒸馏来融入user-item交互图和user-user社交图的信息】
-   STAM: A Spatiotemporal Aggregation Method for Graph Neural Network-based Recommendation. WWW 2022 【同时聚合时空领域信息】
-   Hypercomplex Graph Collaborative Filtering. WWW 2022 【超复图协同过滤】
-   Graph Neural Transport Networks with Non-local Attentions for Recommender Systems. WWW 2022 【使用非局部注意力来实现不加深GNN的同时捕捉节点间的长距离依赖】
-   FIRE: Fast Incremental Recommendation with Graph Signal Processing. WWW 2022 【通过图信号处理来实现快速增量推荐】
-   Large-scale Personalized Video Game Recommendation via Social-aware Contextualized Graph Neural Network. WWW 202 【同时考虑个性化，游戏上下文，社交联系】

### Contrastive Learning based

- Self-supervised Graph Learning for Recommendation. SIGIR 2021 【推荐的自监督图学习】
- The World is Binary: Contrastive Learning for Denoising Next Basket Recommendation. SIGIR 2021 【通过对比学习对下一篮推荐问题去噪】
- Contrastive Pre-Training of GNNs on Heterogeneous Graphs. CIKM 2021【图神经网络的对比预训练】
- Hyperbolic Hypergraphs for Sequential Recommendation. CIKM 2021【使用双曲超图进行序列推荐】
- Contrastive Curriculum Learning for Sequential User Behavior Modeling via Data Augmentation. CIKM 2021【applied paper，通过数据增强进行序列用户行为建模的对比课程学习】
- Semi-deterministic and Contrastive Variational Graph Autoencoder for Recommendation. CIKM 2021【用于推荐的半确定性和对比变分图自动编码器】
- Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation. CIKM 2021【用于群组推荐的自监督超图学习】
- Contrastive Meta Learning with Behavior Multiplicity for Recommendation. WSDM 2022【具有行为多样性的对比元学习推荐】
- Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation. WSDM 2022【序列推荐中表征退化问题的对比学习】
- C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System. WSDM 2022【对话式推荐系统的粗到细对比学习】
- Prototypical Contrastive Learning And Adaptive Interest Selection for Candidate Generation in Recommendations. CIKM 2022 【候选生成的原型对比学习和自适应兴趣选择】
- Multi-level Contrastive Learning Framework for Sequential Recommendation. CIKM 2022 【序列推荐多层次对比学习框架】
- Contrastive Learning with Bidirectional Transformers for Sequential Recommendation. CIKM 2022 【用于序列推荐的双向 Transformer 对比学习】
- Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning. CIKM 2022 【通过多层次交互式对比学习改进知识感知推荐】
- Contrastive Cross-Domain Sequential Recommendation. CIKM 2022 【对比跨域序列推荐】
- MIC:Model-agnostic Integrated Cross-channel Recommender. CIKM 2022 【与模型无关的集成跨渠道推荐器】
- Temporal Contrastive Pre-Training for Sequential Recommendation. CIKM 2022 【时序推荐的时间对比预训练】
- Explanation Guided Contrastive Learning for Sequential Recommendation. CIKM 2022 【序列推荐的解释引导对比学习】
- Contrastive Cross-domain Recommendation in Matching. KDD 2022 【对比学习捕获用户兴趣】
- A Review-aware Graph Contrastive Learning Framework for Recommendation. SIGIR 2022 【考虑评论的图对比学习】
- Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation. SIGIR 2022 【简单的图对比学习方法】
- Knowledge Graph Contrastive Learning for Recommendation. SIGIR 2022 【知识图谱上的对比学习】
- Self-Augmented Recommendation with Hypergraph Contrastive Collaborative Filtering. SIGIR 2022 【超图上的对比学习】
- Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System. SIGIR 2022 【多级交叉视图的对比学习】
- Dual Contrastive Network for Sequential Recommendation. SIGIR 2022 【short paper，双对比网络】
- Improving Micro-video Recommendation via Contrastive Multiple Interests. SIGIR 2022 【short paper，对比多兴趣提升短视频推荐】
- An MLP-based Algorithm for Efficient Contrastive Graph Recommendations. SIGIR 2022 【short paper，基于MLP的算法实现高效图对比】
- Multi-modal Graph Contrastive Learning for Micro-video Recommendation. SIGIR 2022 【short paper，多模态图对比学习】
- Towards Results-level Proportionality for Multi-objective Recommender Systems. SIGIR 2022 【short paper，动量对比方法实现结果均衡的多目标推荐系统】
- Socially-aware Dual Contrastive Learning for Cold-Start Recommendation. SIGIR 2022 【short paper，社交感知的双重对比学习】
- Exploiting Negative Preference in Content-based Music Recommendation with Contrastive Learning. RecSys 2022 【利用对比学习挖掘基于内容的音乐推荐中的负面偏好】
- Enhancing Sequential Recommendation with Graph Contrastive Learning. IJCAI 2022 【用于序列推荐的图对比学习】
- Intent Contrastive Learning for Sequential Recommendation. WWW 2022 【利用用户意图来增强序列推荐】
- A Contrastive Sharing Model for Multi-Task Recommendation. WWW 2022 【使用对比掩码来解决多任务中的参数冲突问题】
- Trading Hard Negatives and True Negatives: A Debiased Contrastive Collaborative Filtering Approach. IJCAI 2022 【探索正确的负样本】
- Contrastive Graph Structure Learning via Information Bottleneck for Recommendation. NIPS 2022 【基于信息瓶颈的对比图结构学习】
- Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW 2022 【通过邻居节点间的对比学习来改善图协同过滤】

### Adversarial Learning based

- PD-GAN: Adversarial Learning for Personalized Diversity-Promoting Recommendation. IJCAI 2019. [PDF](https://www.ijcai.org/proceedings/2019/0537.pdf) 【考虑推荐多样性的对抗性学习】
- Sequential Recommendation with Self-attentive Multi-adversarial Network. SIGIR 2020 【自注意力机制的多对抗网络】
- Revisiting Adversarially Learned Injection Attacks Against Recommender Systems. RecSys 2020 【推荐系统对抗学习的注入攻击】
- Fairness-Aware News Recommendation with Decomposed Adversarial Learning. AAAI 2021 【基于分解对抗学习的公平新闻推荐】
- User Simulation via Supervised Generative Adversarial Network. WWW 2021 【通过监督生成对抗网络进行用户仿真模拟】
- A Study of Defensive Methods to Protect Visual Recommendation Against Adversarial Manipulation of Images. SIGIR 2021 【保护视觉推荐免受图像对抗性操纵的防御方法研究】
- Adversarial Feature Translation for Multi-domain Recommendation. KDD 2021 【对抗特征迁移多任务学习】
- Adversarial Gradient Driven Exploration for Deep Click-Through Rate Prediction. KDD 2022 【CTR EE 探索】
- Triple Adversarial Learning for Influence based Poisoning Attack in Recommender Systems. KDD 2021 【三元对抗学习在推荐系统中毒攻击】
- Adversarial Filtering Modeling on Long-term User Behavior Sequences for Click-Through Rate Prediction. SIGIR 2022 【short paper，对抗过滤建模用户长期行为序列】
- Mitigating Consumer Biases in Recommendations with Adversarial Training. SIGIR 2022 【short paper，对抗训练去偏】
- Generative Adversarial Framework for Cold-Start Item Recommendation. SIGIR 2022 【short paper，针对冷启动商品的生成对抗框架】
- Adversarial Graph Perturbations for Recommendations at Scale. SIGIR 2022 【short paper，大规模推荐中的对抗图扰动】
- The Idiosyncratic Effects of Adversarial Training on Bias in Personalized Recommendation Learning. RecSys 2021 【LBR，对抗训练对流行度偏差的影响研究】
- Adversary or Friend? An Adversarial Approach to Improving Recommender Systems. RecSys 2022 【对抗式方法促进推荐系统公平性】

### Autoencoder based

- Deep Critiquing for VAE-based Recommender Systems. SIGIR 2020 【基于 VAE 推荐系统的深度评判】
- TAFA: Two-headed Attention Fused Autoencoder for Context-Aware Recommendations. RecSys 2020 【用于上下文感知推荐的双头注意力融合自编码器】
- Adversarial and Contrastive Variational Autoencoder for Sequential Recommendation. WWW 2021 【对抗和对比式变分自动编码器】
- Bilateral Variational Autoencoder for Collaborative Filtering. WSDM 2021 【用于协同过滤的双向变分自动编码器】
- Local Collaborative Autoencoders. WSDM 2021 【局部的协同自编码器】
- Vector-Quantized Autoencoder With Copula for Collaborative Filtering. CIKM 2021【short paper，用于协同过滤的矢量量化自动编码器】
- Semi-deterministic and Contrastive Variational Graph Autoencoder for Recommendation. CIKM 2021【用于推荐的半确定性和对比变分图自动编码器】
- Disentangling Preference Representations for Recommendation Critiquing with $$\beta$$-VAE. CIKM 2021【用于推荐的 VAE 偏好表示】
- The Interaction Graph Auto-encoder Network Based on Topology-aware for Transferable Recommendation. CIKM 2022 【基于拓扑感知的可迁移推荐交互图自动编码器网络】
- ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation. CIKM 2022 【用于序列推荐的对比变分自动编码器】
- Vector-Quantized Autoencoder With Copula for Collaborative Filtering. CIKM 2021【short paper，用于协同过滤的矢量量化自动编码器】
- VAE++: Variational AutoEncoder for Heterogeneous One-Class Collaborative Filtering. WSDM 2022【异构单类协同过滤的变分自动编码器】
- Debiasing Learning for Membership Inference Attacks Against Recommender Systems. KDD 2022 【VAE 去偏】
- Multi-view Denoising Graph Auto-Encoders on Heterogeneous Information Networks for Cold-start Recommendation. KDD 2021 【异构信息网络多视图去噪图自动编码器实现冷启动】
- PEVAE: A hierarchical VAE for personalized explainable recommendation. SIGIR 2022 【利用层次化VAE进行个性化可解释推荐】
- Improving Item Cold-start Recommendation via Model-agnostic Conditional Variational Autoencoder. SIGIR 2022 【short paper，模型无关的自编码器提升商品冷启动推荐】
- Mitigating the Filter Bubble while Maintaining Relevance: Targeted Diversification with VAE-based Recommender Systems. SIGIR 2022 【short paper，定向多样化】
- Cold Start Similar Artists Ranking with Gravity-Inspired Graph Autoencoders. RecSys 2021 【图自动编码器冷启动相似艺术家排序问题】
- Fast Multi-Step Critiquing for VAE-based Recommender Systems. RecSys 2021 【基于多模态建模假设的变分自动编码器】
- Towards Source-Aligned Variational Models for Cross-Domain Recommendation. RecSys 2021 【变分自动编码器用于跨域推荐】
- Scalable Linear Shallow Autoencoder for Collaborative Filtering. RecSys 2022 【LBR，用于协同过滤的可扩展线性浅层自动编码器】

### Meta Learning-based 

- Meta-learning on Heterogeneous Information Networks for Cold-start Recommendation. KDD 2020 【异构信息网络的元学习用于冷启动推荐】
- MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation. KDD 2020 【冷启动推荐的内存增强元优化】
- Curriculum Meta-Learning for Next POI Recommendation. KDD 2021 【基于元学习的下一代兴趣点推荐系统】
- Cold-Start Sequential Recommendation via Meta Learner. AAAI 2021 【通过元学习冷启动序列推荐】
- Personalized Adaptive Meta Learning for Cold-start User Preference Prediction. AAAI 2021 【用于冷启动用户偏好预测的个性化自适应元学习】
- FORM: Follow the Online Regularized Meta-Leader for Cold-Start Recommendation. SIGIR 2021 【遵循在线规范化的元领导者进行冷启动推荐】
- Learning to Warm Up Cold Item Embeddings for Cold-start Recommendation with Meta Scaling and Shifting Networks. SIGIR 2021 【元缩放和偏移网络进行冷启动推荐】
- Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction. SIGIR 2021 【点击率预测中冷启动广告】
- CBML: A Cluster-based Meta-learning Model for Session-based Recommendation. CIKM 2021【用于会话推荐的基于聚类的元学习】
- CMML: Contextual Modulation Meta Learning for Cold-Start Recommendation. CIKM 2021【元学习+冷启动】
- Multimodal Graph Meta Contrastive Learning. CIKM 2021【short paper，多模态元图对比学习】
- Multimodal Meta-Learning for Cold-Start Sequential Recommendation. CIKM 2022 【冷启动序列推荐的多模态元学习】
- Curriculum Meta-Learning for Next POI Recommendation. KDD 2021 【基于元学习的下一代兴趣点推荐系统】
- Contrastive Meta Learning with Behavior Multiplicity for Recommendation. WSDM 2022【具有行为多样性的对比元学习推荐】
- Long Short-Term Temporal Meta-learning in Online Recommendation. WSDM 2022【在线推荐中的长短期时间元学习】
- Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling. WSDM 2022【一种用于广告商建模的多场景多任务元学习方法】
- Contrastive Meta Learning with Behavior Multiplicity for Recommendation. WSDM 2022【具有行为多样性的对比元学习推荐】
- Meta-Learning for Online Update of Recommender Systems AAAI2022 【自适应最优学习率】
- A Dynamic Meta-Learning Model for Time-Sensitive Cold-Start Recommendations. AAAI 2022 【元学习动态推荐框架】
- Deployable and Continuable Meta-Learning-Based Recommender System with Fast User-Incremental Updates. SIGIR 2022 【基于元学习的可部署可拓展推荐系统】
- MetaCVR: Conversion Rate Prediction via Meta Learning in Small-Scale Recommendation Scenarios. SIGIR 2022 【short paper，小规模推荐场景下的元学习】
- Comprehensive Fair Meta-learned Recommender System. KDD 2022 【元学习公平性框架】
- Learning An Adaptive Meta Model-Generator for Incrementally Updating Recommender Systems. RecSys 2021 【自适应元模型生成器】

### AutoML-based 

- AutoFAS: Automatic Feature and Architecture Selection for Pre-Ranking System. KDD 2022 【粗排 NAS 搜索】
- Single-shot Embedding Dimension Search in Recommender System. SIGIR 2022 【嵌入维度搜索】
- AutoLossGen: Automatic Loss Function Generation for Recommender Systems. SIGIR 2022 【自动损失函数生成】
- NAS-CTR: Efficient Neural Architecture Search for Click-Through Rate Prediction. SIGIR 2022 【高效的网络结构搜索】

### Causal Inference/Counterfactual

- Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions. KDD 2020 【结合序列奖励交互的反事实评估】
- A General Knowledge Distillation Framework for Counterfactual Recommendation via Uniform Data. SIGIR 2020 【反事实推荐的知识蒸馏框架】
- Unbiased Learning for the Causal Effect of Recommendation. RecSys 2020 【推荐因果效应的无偏学习】
- Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue. SIGIR 2021 【缓解虚假点击问题的反事实建议】
- Personalized Counterfactual Fairness in Recommendation. SIGIR 2021 【推荐中个性化的反事实公平性】
- Should Graph Convolution Trust Neighbors? A Simple Causal Inference Method. SIGIR 2021 【因果推断方法分析图卷积网络】
- Counterfactual Data-Augmented Sequential Recommendation. SIGIR 2021 【反事实数据增强】
- CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation. SIGIR 2021 【反事实用户序列合成】
- Causal Intervention for Leveraging Popularity Bias in Recommendation. SIGIR 2021 【在推荐中利用流行偏见的因果干预】
- Disentangling User Interest and Conformity for Recommendation with Causal Embedding. WWW 2021 【利用因果解耦用户兴趣和推荐一致性】
- Counterfactual Explainable Recommendation. CIKM 2021【反事实可解释推荐】
- Counterfactual Review-based Recommendation. CIKM 2021【基于评论的反事实推荐】
- Top-N Recommendation with Counterfactual User Preference Simulation. CIKM 2021【反事实用户偏好模拟的 Top-N 推荐】
- CausCF: Causal Collaborative Filtering for Recommendation Effect Estimation. CIKM 2021【applied paper，因果关系协同过滤用于推荐效果评估】
- CauSeR: Causal Session-based Recommendations for Handling Popularity Bias. CIKM 2021【short paper，用于流行度去偏的因果关系序列推荐】
- Dynamic Causal Collaborative Filtering. CIKM 2022 【动态因果协同过滤】
- Towards Unbiased and Robust Causal Ranking for Recommender Systems. WSDM 2022【推荐系统的无偏和稳健因果排名】
- A Counterfactual Modeling Framework for Churn Prediction. WSDM 2022 【客户流失预测的反事实建模框架】
- Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System. KDD 2021 【不可知反事实推理模型消除推荐系统的流行偏差】
- Practical Counterfactual Policy Learning for Top-K Recommendations. KDD 2022 【反事实框架】
- Alleviating Spurious Correlations in Knowledge-aware Recommendations through Counterfactual Generator. SIGIR 2022 【利用反事实生成器缓解假知识】
- Online Evaluation Methods for the Causal Effect of Recommendations. RecSys 2021 【推荐中因果效应的在线评估方法】
- Causal Representation Learning for Out-of-Distribution Recommendation. WWW 2022 【利用因果模型解决用户特征变化问题】
- A Model-Agnostic Causal Learning Framework for Recommendation using Search Data. WWW 2022 【将搜索数据作为工具变量，解耦推荐中的因果部分和非因果部分】
- Causal Preference Learning for Out-of-Distribution Recommendation. WWW 2022 【从观察数据可用的正反馈中联合学习具有因果结构的不变性偏好，再用发现的不变性偏好继续做预测】
- Learning to Augment for Casual User Recommendation. WWW 2022 【通过数据增强来增强对随机用户的推荐性能】

### Other Techniques

- Neural Input Search for Large Scale Recommendation Models. KDD 2020 【大规模推荐模型的神经输入搜索】
- Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems. KDD 2020 【使用互补分区的组合嵌入用于内存高效的推荐系统】
- Field-aware Embedding Space Searching in Recommender Systems. WWW 2021 【推荐系统中的域感知向量空间搜索】
- PROPN: Personalized Probabilistic Strategic Parameter Optimization in Recommendations. CIKM 2022 【推荐中的个性化概率策略参数优化】
- Generalized Deep Mixed Models. KDD 2022 【集成 DNN 和 LR 增强】
- Leaping through Time with Gradient-Based Adaptation for Recommendation. AAAI 2022 【基于轨迹的元学习】
- Forest-based Deep Recommender. SIGIR 2022 【深度森林】
- cDLRM: Look Ahead Caching for Scalable Training of Recommendation Models. RecSys 2021 【用于推荐模型可扩展训练的高速缓存】
- Hierarchical Latent Relation Modeling for Collaborative Metric Learning. RecSys 2021 【协作度量学习的层次潜在关系建模】
- Learning to Represent Human Motives for Goal-directed Web Browsing. RecSys 2021 【引入人类动机的心理学构建浏览框架】
- Together is Better: Hybrid Recommendations Combining Graph Embeddings and Contextualized Word Representations. RecSys 2021 【结合图表示和上下文单词表示的混合推荐】
- A Constrained Optimization Approach for Calibrated Recommendations. RecSys 2021 【LBR，结合精度和校准的约束优化方法】
- Eigenvalue Perturbation for Item-based Recommender Systems. RecSys 2021 【LBR，物品推荐中特征值扰动的分析】
- Optimizing the Selection of Recommendation Carousels with Quantum Computing. RecSys 2021【LBR，用量子计算优化音乐歌单推荐】
- Towards Recommender Systems with Community Detection and Quantum Computing. RecSys 2022【LBR，利用量子计算进行社区检测】
- Dual Attentional Higher Order Factorization Machines. RecSys 2022 【双注意高阶因式分解机】
- You Say Factorization Machine, I Say Neural Network – It’s All in the Activation. RecSys 2022 【通过激活函数建立因子分解机和神经网络的联系】
