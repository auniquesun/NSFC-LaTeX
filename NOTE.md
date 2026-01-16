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
            - 分割的相关文献也加上了
        - 这个就是受到了CLIP等模型开放词表/强泛化的启发，其他领域逐步发展起来

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
    - 我的研究目标是什么？至今连这个也没搞清楚，这不行啊
        - 本项目想做啥，做到什么程度，直白的语言先写一段出来

2. 研究内容
    - 这部分是重中之重，要花大量时间攻关
    - 给出总览，再分别讲解每部分内容
    - 各部分内容层层递进
    - 明确本项目的挑战问题是什么？根据挑战问题设定研究内容

    - 设想1
        0. 开放环境中点云数据流特征提取
            - 得开始干了，不能发呆了
            - 这块特征提取要研究什么东西？

        1. 缓存构建 + 更新
            - 开始干，不犹豫
        3. 点云处理测试时自适应 
            - 这部分也可设置为自适应融合策略
        
        * 存在的问题
            - 我的场景设定是处理实时测试数据流，这就需要对缓存模型进行实时的构建和更新
                1. 也就是说**构建和更新**是紧密联系在一起的，我现在把它们分解成**两部分研究内容**，是不是不恰当？
                2. 如果设成一个内容，研究内容目前就成了两块，整体少了一块，显得太少了，再添加个什么内容呢？
                3. 想到一个点：**开放环境下3D点云测试数据流特征提取**
                    - 这个点能不能当作一个研究内容？研究内容要针对挑战性问题，**数据流特征提取**是一个挑战性问题吗？
                        1. 有一定道理，存储成本高,表达能力不够
                        3. 把特征提取当作一个研究点，就得真正做点工作
                            - hierarchical design 太简单了

    - 设想2【目前还没有另一种想法】
        - 开放环境下点云特征**鲁棒**编码，不同于封闭数据集单模态设定（要让点云特征含有语义信息，这是实现开放环境感知的关键）
            - 这里鲁棒形容得未必准确，想表达的含义是“将开放场景中的概念赋予模型”，后面要用合适的词替换
            - 这样一来，创新性就出来了
        - 开放环境下，基于鲁棒特征编码的缓存模型构建
        - 开放环境下，基于缓存模型的点云自适应分析策略

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

    - 新情况：
        1. 2026年国自然，还需要写关键科学问题吗？
        2. 模板里面没这部分内容了，但是我觉得还得把科学问题总结出来，科学问题对于科研项目是必要的
        3. 所以本项目的关键科学问题是什么？

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
    - 提一下内蒙古大学**时空智能中心**？这和自己的申请题目非常接近
    - 平台/合作导师，能为本项目带来什么？

### 写作灵感
0. 标题部分
    - 是否有必要引入“基于缓存模型”的表述
        - 目前的标题比较长，有的预审人指出了这个问题
        - 这部分体现主要研究方法，在立项依据引出比较好？
        - 【已确定】**不引入**“基于缓存模型”
    
    - 新构想，体现创新性
        - 开放环境下检索增强驱动的3D点云自适应分析方法研究
        - Research on retrieval augmentation driven adaptive analysis method for 3D point cloud in open world

1. 讲立项依据大背景时，可以借鉴 李飞飞 教授关于 **spatial intelligence** 的论述
    - 在 deepseek, gemini 上做了相关搜索
    - 我的研究范畴属于 **spatial intelligence**

1. 讲测试时自适应，**和人类的学习过程作类比：在新的环境快速适应调整**
    - 物竞天择，适者生存，自然之法

1. 讲测试时自适应，可以联系到当前大预言模型做的 test-time scaling，NVIDA 在发布会上老用这个概念和图，想想怎么提及一下
    - [参考博客](https://blogs.nvidia.com/blog/ai-scaling-laws/)

2. 相关工作，其实是三大流派
    - traing only methods
    - methods based on uni-modal models
        - cannot conduct open-set tasks
    - methods based on large multi-modal models
        - zero-shot inference for each sample
        - do not utilize statistics of online test samples

3. 讲缓存模型，首先讲大的方面，大模型对工具的使用 -> AI agent -> search agent
    - 检索增强机制：互联网的知识 -> 外部知识库 -> 信息锚点
    - 在此基础上阐明 “缓存模型”

4. 研究基础
    - 可以参考“程明明的写法”，分几个部分讲，每个部分用图片展示相关概念和技术

5. 概念解释
    1. 行文中经常提到**开放环境**/**开放集合**，在我的上下文应该是等价的，在合适的地方应该解释一下。

6. 对于科学问题的阐述，可以参考 sharpness-aware minimization (SAM) 的数学表达，当测试输入带有噪声时，模型仍要保持正确预测能力。

7. 画图的时候，可以考虑给出 "prompts + example images"，扔给市场上公开大模型，让大模型生成图片

### 2025/12/27
1. 我现在写到哪里了？为什么写不下去？
    - 立项依据没完全写明白，靶子有待瞄准，后来决定将靶子和 Point-Cache 对齐

2. 把相关工作列出来，分类组织，根据项目主题作总结和评述
    - 最后一步还没做呢
    - 硬着头皮总结

3. 国自然2026申请书改版了
    - 我的申请书模板还是旧版，研究内容部分怎么写呢？
    - 之前分为研究目标/研究内容/拟解决的关键科学问题
    - 把技术方案写出来，上述部分更好写一点？

4. 研究基础部分，可以把发表的论文**做个表格总结一下**，特别是突出代表作和本项目的关系

### 2026/01/02
1. 对本项目和未来研究方向的思考
    - 本人前期研究注重对现有方法的技巧性提升，没有专注于具体任务本身，这让研究意义显得不是非常重要（**个人切身感受**）；后面的研究要从问题出发，注重任务/问题的意义，不要老是想着走捷径发论文
        - 这是当下的体会，很可能发生变化，哪个有价值的方向都不是容易做的

2. 提出的这个项目到底行不行，本子写起来这么困难？
    - 核心原因是本项目里 “*点云特征编码*” “*基于缓存模型的测试时自适应机制*” 是**流程性的**，不是创造性的，只有其中 “缓存模型的构建和动态更新” 是**核心创新**

    - 点云特征编码这部分，可以*真刀真枪研究方法和架构*，这样就能当做一个研究点

3. 项目书里反复用“*缓存模型*”到底恰不恰当？
    - 存数据的其实就是个缓存，`构建/更新/自适应`合起来才叫模型
    - 修改的时候想想怎么更正

### 2026/01/03
1. 写到“*拟采取的研究方案和可行性分析*”部分了
    - 这部分先画个图？
    - 再按照设定的框架去写
    - 写不出来啊，一天坐在这没效率，都是在刷视频，何必呢？
    - 就照着已经发表论文 Point-Cache “Mehtod Section” 写一版出来

### 2026/01/05
1. 写到可行性分析部分了，之前设想了很多内容，现在为啥写不出来了？
    - deep think，这么好的基础不能被浪费了

    - 研究团队这部分，应该怎么写？
        - 内蒙古大学
        - 计算机学院
        - 时空智能研究中心

2. 写到创新点了，本项目有啥创新点？
    1. 引入文本-图像的点云特征编码，奠定开放环境下理解能力
    2. 基于文本-图像-点云多模态点云特征表示的动态分层缓存方法
    3. 基于动态分层缓存方法的测试时自适应策略

### 2026/01/07
1. 写到年度研究计划了，这部分怎么写？
    1. 参考下别人的格式
    2. 做个甘特图吧，是用PPT还是Excel，后者吧，做起来更方便
    3. 相当于分解每个阶段的任务，思考一下怎么划分？
        - 目前已经划分出来了，问题是不能一句话就说完，要展开内容啊
        - 展开讲的话要把具体要做啥搞明白，目前写不出来就是因为做的事情不明确，但是思考了一下，缓存模型的建立和更新，这部分还是挺明确的。
        - 现在句子都不完整，有明显问题啊

2. 写到预期研究成果了，分哪几个部分？这几个部分不一定全写，根据项目情况来
    - 数据构建
        - 这块有没有？
        - 目前的情况数据集方面没有要做的
    - 方法和系统
        - 提出了什么方法，做了什么系统
            - 方法就是不同阶段的解法
            - 1套演示系统
        - 这是关键，方法要达到怎样的性能，那得找到当前SOTA，性能要超越SOTA
            - 3D物体识别
            - 3D物体检测
            - 3D语义分割
    - 发表论文
        - 5篇及以上论文
    - 培养学生
        - 1-2名博士，3-4名硕士
    - 学术交流合作
        - 参加CVPR，ICCV，NeurIPS等高水平国际会议

### 2026/01/09
1. 写到研究基础部分了，之前写了初稿，要优化升级一下
    1. 在这部分讲到了“动态分层缓存模型”，这和正文部分的创新点冲突了吧，相当于本项目中声明的创新点已经在之前的论文发表过了，这就不能说是创新点
        - CVPR 2025 论文，换一种说法
    2. 之前的可行性分析部分，做了一个表格
        - 考虑将这个表格移到这里
        - 表格加一栏“联系与区别”
    3. 参考的cmm模板，他把自己各个研究方向的工作总结了一遍，我要不要这样做？
        - 倒是可以，分为哪几个方向呢？都是点云处理，下面分的任务就不一样了，识别、分割、检索，这么划分合理吗？
        - 我的想法是用表格总结各个研究工作，不用都写

2. 写到工作基础部分了，这部分就是介绍
    - 已具备的实验条件
        - 内蒙古大学简介
        - 所属团队
        - 办公场地
        - 实验设备：
            - 目前有什么？
                - 笔记本电脑
                - 台式机
                - 打印机
            - 目前缺什么？
                - 服务器
        - 软件
            - python
            - pytorch
            - cuda toolkit
            - open3d
            - https://github.com/torch-points3d/torch-points3d
            - mmdetection3d
        - 数据
            - 公开数据集
                - 识别
                - 检测
                - 分割
            - 实地采集一些开放场景数据？
                - 必要性有待验证，现在的公开数据已经不少了，而且都是开放场景

    - 尚缺少的实验条件、拟解决途径
        - 我基本没有服务器，要写点这个
    
    - 内自然写了一个版本，在这个基础上修改

### 2026/01/12
1. 全面进入修改完善阶段，先要确定问题，再逐步修改
    - 按照顺序，先回顾一下之前的笔记

2. 重新审视标题
    - 去掉了*检索增强*描述，因为第一步还有“开放环境下点云编码架构设计”，单纯放*检索增强*不合适

3. 重新审视立项依据【不能浪了，一步步改】
    1. 首先，从宏观到微观的展开缺乏吸引力的故事，不够吸引人，没有国家战略需求和科技计划
        - 把 spatial ai 引进来
        - 读了 dr. feifei li 这方面的讲述，该总结进去了
        - 【已完成】
    
    2. 当前最大的问题是接不上 spatial intelligence 开启的上文，制定解决思路
        0. 参考前人本子
        1. 明确下文想讲的内容
        2. 找到逻辑联系
        3. 根据逻辑联系写内容
        4. 【已解决】引出挑战问题
            1. 挑战性问题总结不到位，当时还没考虑到 “开放环境下点云特征编码架构”
                - 几大挑战，该好好总结一下了
                - 之前写的内容和上文没有很好接上，该怎么接上？
                - 这就要认真读下文，设计一个思路
            2. 开放环境下点云特征编码架构 —— 这一研究内容只是壳子，要明确到底研究什么内容
                - 以 open yolo 3d 为例

    4. 写作思路: 一个大系统分成3个研究内容的方式去写
        - 【问题】但目前**只有2大块研究内容**
            1. 第一块是点云特征编码
                - Open YOLO 3D (ICLR 2025)，这个文章写作一般，特别是 VAcc & MVPDist 模块，我的项目能否用上？【能用上部分，也得参考一下它的前期工作】
                    1. 先生成 class-agnostic 3D proposals (3D instance segmentation)
                    2. 再用 open-vocabulary 2D object detectors 检测出 bounding boxes with class labels -> bbxox + label map
                    3. 3D propopsals 投影到2D，根据 bbox size+projection range+depth 选出对应 labels, 
                
                - Open3DIS (CVPR 2024)
                    - 2D-guided 3D Instance Proposal
                
                - OpenMask3D (NeurIPS 2023)
                    - use SAM to conduct open-vocabulary perception

            2. 第二块是基于在线动态分层缓存的自适应策略

            3. 最后一块研究内容是什么呢？
                - 2块内容就研究完了，显得有点少
                - 最后加一块应用验证？应用验证不能算单独的研究，只是研究的一部分
                - 调研一下新的论文，按照每年发表顺序
                    - ICML，2月
                    - ICCV，3月，奇数年
                    - ECCV，3月，偶数年
                    - MM, 4月
                    - NeurIPS，5月
                    - AAAI，8月
                    - ICLR，9月
                    - CVPR，11月

        - “缓存模型”
            - 只能是一块研究内容，分成缓存构建、更新、自适应三块不合适
                - 因为解决的是一个挑战，不是3个挑战
