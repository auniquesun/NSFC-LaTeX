### 文献调研
1. Frustrately Easy Test-time Adaptation of VLMs, NeurIPS 2025
    - 将 softmax 中的温度系数设置为机器精度 - `torch.finfo(torch.float64).eps`，相当于**置零**操作
    - 方法不训练，符合预期

2. Test-time adaptation with CLIP reward for zero-shot generalization in vision-language models, ICLR 2024
    - 这篇文章也理解了大致内容，但是 test-time training（利用 CLIP reward 作为反馈信号更新模型参数），我希望方法不要 training

3. Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models, AAAI 2026
    1. 跟进我 CVPR 2025 的工作 Point-Cache
    2. 为每个类别做个聚类：每个生成 K 个 prototypes
    3. cache logits 得到初始预测标签后，用 graph label smoothing 做平滑

### 2025/12/03
1. 项目正式立项，开始大量投入时间精力做

2. 依据研究基础，结合最新文献和研究热点，确定题目
    - 程明明建议先画2个图：研究内容框架图、详细技术路线
    - 我连题目都没确定，做什么都没确定，怎么画这2个图？

    - 看了刘瑞的本子，**一个青年项目体现着一个体系，三大环环相扣的研究内容构成研究体系**，_并不是单拎一个点做论文的思路_

3. 之前想了一个题目，大致是 “基于检索增强和测试时自适应机制的鲁棒可泛化点云分析技术”
    1. 但这个题目构不成3个研究点，不包括实验的3个研究点，也就是形不成体系
    2. 标题不够明确？点云分析，到底包括哪些哪些任务？在标题中没有体现
        - 目前主要的研究基础是 **点云识别**
    3. 研读一些物体检测/语义分割的论文？这方面也开辟一些道路？
        - 有了这方面基础，可以和上述 **点云识别** 构成一个系列，这也是未来一两的研究目标

    4. 无论如何，题目应当把**开放环境**这样的关键词加进来
        - 这是本文想突出解释的一个点，这也是当前研究未解决的问题

4. 备选题目
    1. **开放环境下基于检索增强的测试时自适应点云分析方法研究**
        - 这里 _检索增强_ 不容易和缓存联系起来，让人觉得我没有相关基础？那应该改成什么呢？
            1. 动态分层缓存模型（这个词挺贴切，但是不如 retrieval augmentation 听起来 fancy）
            2. 缓存模型，这个范围大一点，但上述问题仍存在
        - 可以在写作过程中，讲明 **检索增强实质是要构建一个缓存模型，从缓存模型内检索相关知识，增强点云处理任务**
    2. **开放环境下基于缓存模型的测试时自适应点云分析方法研究**
        - 将关键科学问题抽象为  ->  缓存模型    ->  检索增强范式
            1. 缓存建立
            2. 缓存管理
                - 动态更新机制
            3. 基于缓存模型的测试时自适应方法

### 立项依据
1. 研究背景和科学意义
    - 之前觉得英文论文不好写
    - 现在感觉中文项目书也好难写，关键是内容丰富和逻辑联系
    - 2025/12/22，国自然本子很长时间没有进展了，而且是内生动力不足，这不行啊，得不顾一切往前推
        - 把握现在，没有后悔药

2. 国内外研究现状
    - 这部分可以先写起来，把 point-cache 的 `.bib` 文件拿过来
        - 综合考虑国际/国内相关工作
        - 充分引用自己发表的论文

    - 开放环境下点云处理
        - 这块按照什么组织呢？先把相关工作放上
        - 识别/检测都读了一些论文，**分割方面比较欠缺**

    - 测试时自适应方法
        - 读了这么多文献，该总结一下主要思想和技术流派了，单一列出来价值不大
            - 我记得在哪里做过总结，`Point-Cache-origin/NOTE.md`
        - 下面分类总结差不多了，要把文字写出来
        
        1. parameter tuning based methods (learnable)
            - training objective: **entropy minimization**
                - [经典，但这篇文章不属于TTA方法] Sharpness-aware minimization for efficiently improving generalization, ICLR 2021
                - [经典] Towards stable test-time adaptation in dynamic wild world，ICLR 2023
                - [经典，但这篇文章属于TTT，不属于TTA] Test-time training with self-supervision for generalization under distribution shifts, ICML 2020
            - adapt prompts
                - TPT, DiffTPT
            - adapt BN statistics
                - Improving robustness against common corruptions by covariate shift adaptation, neurips 2020
                    - **core idea**: Replacing the activation statistics estimated by batch normalization on the training set with the statistics of the corrupted images consistently improves the robustness across
                - [经典] Tent: Fully test-time adaptation by entropy minimization, iclr 2021 
                - [经典] MEMO: Test-time Adaptation via Augmentation and Adaptation, neurips 2022
                
            - adapt other parts
                - Tent: Fully test-time adaptation by entropy minimization, iclr 2021
                - Adaptive Risk Minimization: Learning to Adapt to Domain Shift, neurips 2021
                - AdaContrast: Contrastive Test-Time Adaptation, cvpr 2022
                - [adapt image encoder] RLCF: Test-time adaptation with CLIP reward for zero-shot generalization in vision-language models, ICLR 2024

        2. cache based methods (training-free)
            - TDA
            - point-cache
            - BFTT3D
            - Uni-Adapter
            - SCA, statistics caching test-time adaptation

        3. 其他分类 (training-free)
            - Zero: Frustratingly Easy Test-Time Adaptation of Vision-Language Models (set temperature to zero when conducting softmax), NeurIPS 2024
            - On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning? CVPR 2024 
                - we introduce a robust **MeanShift** for Test-time Augmentation (MTA), which surpasses prompt-based methods without requiring this intensive training procedure. This positions MTA as an ideal solution for both standalone and API-based applications.

    - 缓存模型/检索增强方法
        - 基础的想法是什么？从哪里找参考？
        - 3D domain
            - Point-PEFT
            - Point-NN (**not belong to test-time methods**)
            - Point-Cache
            - BFTT3D
            - Uni-Adapter

        - 2D domain
            - Matching networks for one shot learning, NeurIPS 2016
            - Prototypical networks for few-shot learning. NeurIPS 2017
            - Model-agnostic meta-learning for fast adaptation of deep networks, ICML 2017
            - A closer look at few-shot classification, ICLR 2019
            - Meta-baseline: Exploring simple meta-learning for few-shot learning, ICCV 2021
            - Improving test-time adaptation via shift-agnostic weight regularization and nearest source prototypes, ECCV 2022
            - Test-time classifier adjustment module for model-agnostic domain generalization. NeurIPS 2021
            - Tip-Adapter
            - CaFo
            - TDA

        - language modeling
            - Matching networks for one shot learning, NeurIPS 2016 (classic paper)
            - Unbounded cache model for online language modeling with open vocabulary, NeurIPS 2017
            - Pointer sentinel mixture models, NeurIPS 2017
            - Generalization through memorization: Nearest neighbor language models, NeurIPS 2020

    - 写到这里，遇到一大问题就是点不出研究现状存在的问题，如果不能明确说明问题所在，那么怎么提出自己的方案和创新？这是一直以来强调的**问题导向**
        - 大的问题在前面讲过了，就是测试环境会出现新情况新变化，这会导致模型鲁棒性和泛化能力大幅降低
        - 但这个问题太大了，还要进一步细化，现**有方法在哪些方面做得不好，导致鲁棒性和泛化能力不好**？
            - 【这一点是目前没明确的】
            - **解决思路**
                1. 延续我的 Point-Cache 写作思路，但目前来看该方法已经被 Uni-Adapter 超越，我觉得要有信心和底气，博采众长，再次实现超越
                2. 技术层面升级 Point-Cache，实现反超
                    - 对特征聚类的时间成本很高，用得内存也很多，考虑用几个 pooling?
                3. 任务层面升级 Point-Cache，实现多任务统一，变得更通用

    - 对于TTA
        1. 问题1：鲁棒性+泛化性仍然很差，这是根本问题，只是在缓解，没有解决
            - 基础模型准确性不够，adaptation 只是锦上添花，所以目前的研究没太大意义？
        2. 问题2：当前的TTA方法存在很多假设，限制了在实际应用的部署
            - 需要训练集构建
            - 需要 training
        3. 问题3：当前主要是识别任务在做 “缓存模型+测试时自适应”，没看到更复杂的 “检测+分割” 任务

    - 对于 cache models
        1. 问题1：需要训练集构建
        2. 问题2：缺乏动态更新机制
        3. 问题3：存储的数据不够全面有代表性
            - Point-Cache 存储 global + local features
            - 以往的 cache models 仅存储 global features

3. 主要挑战总结

4. 概述研究基础
    - 怎么写起来这么土，自己都看不下去那种

### 研究内容
1. 研究目标

2. 研究内容
    - 这部分是重中之重，要花大量时间攻关
    - 给出总览，再分别讲解每部分内容
    - 各部分内容层层递进

    - 设想1
        1. 缓存构建
            - 开始干，不犹豫
        2. 缓存更新
        3. 点云处理测试时自适应 
            - 这部分也可设置为自适应融合策略

    - 设想2【目前还没有另一种想法】

3. 拟解决的关键科学问题【立项依据写不出来，要不要先写这部分？】
    - 这部分是重中之重，要花大量时间攻关
    - 给出总览，再分别讲解每部分内容
    - 各部分内容层层递进

    - 设想1
        1. 缓存构建
            - 开始干，不犹豫
        2. 缓存更新
        3. 点云处理测试时自适应 
            - 这部分也可设置为自适应融合策略

    - 设想2【目前还没有另一种想法】

### 研究方案和可行性分析
1. 研究方案
    - 与研究内容一一对应，这里要展开阐述

2. 可行性分析

### 本项目特色与创新之处
1. 创新1

2. 创新2

3. 创新3

### 年度研究计划与预期结果
1. 年度研究计划

2. 预期结果

### 研究基础与工作条件
1. 研究基础

2. 工作条件

### 写作灵感
0. 标题部分
    - 是否有必要引入“基于缓存模型”的表述
        - 目前的标题比较长，有的预审人指出了这个问题
        - 这部分体现主要研究方法，在立项依据引出比较好？

1. 讲测试时自适应，**和人类的学习过程作类比：在新的环境快速适应调整**
    - 物竞天择，适者生存，自然之法

2. 相关工作，其实是三大流派
    - traing only methods
    - methods based on uni-modal models
        - cannot conduct open-set tasks
    - methods based on large multi-modal models
        - zero-shot inference for each sample
        - do not utilize statistics of online test samples