请你为我实现一个对图像中目标检测结果的后处理脚本：

给定两个目录：images_root, labels_root

images_root中包含若干图像，图像格式为常见的图像格式；labels_root包含若干.txt格式的检测结果文件。检测结果文件与图像一一对应，名称仅在后缀名上不同。

每一个txt结果文件当中，格式如下：
```
4 0.6145519614219666 0.263671875 0.625983715057373 0.28358525037765503 0.5885801911354065 0.3694746494293213 0.5771484375 0.34956127405166626 0.5
4 0.3218342661857605 0.5272374749183655 0.34222498536109924 0.5627756118774414 0.2937750816345215 0.6739720702171326 0.27338436245918274 0.6384339332580566 0.26
4 0.5687585473060608 0.05067303031682968 0.5857345461845398 0.08038102835416794 0.5471952557563782 0.1684708297252655 0.5302192568778992 0.13876283168792725 0.8
```
有10列数字，中间由空格进行分隔。第一列数字是一个整数，代表了目标的类别。从第2列到第9列是目标检测框的四个点归一化的坐标。按照x1,y1,x2,y2,x3,y3,x4,y4的格式排列，四个点按照顺时针进行编号。第10列表示结果的置信度，区间是0到1。

现在假设整个图像集合按照场景分为若干组图像，每一个场景下的图像当中，目标之间的相对位置、目标的检测框方向不会有较大的偏差。

该后处理脚本会对所有图像检测结果进行聚类，对于包含结果较多的聚类，视为某一场景下的正常检测结果；而对于包含结果较为稀少的聚类，视为是因为漏检或者虚假而造成的异常检测结果。在我使用的语言当中，“结果”指的是检测算法对于某一张图像所给出的所有预测框、分类和置信度的集合，其中的某一个预测我称之为“结果当中的目标”或者“结果当中的检测框”。

脚本的大致流程如下：
1. 读取所有标签文件，提取检测框，计算中心点。
2. 进行聚类，找出异常结果和正常聚类。
3. 对于每个正常聚类，找到它的代表结果。
4. 对于每个异常结果，找到它匹配程度最高的代表结果。
5. 根据匹配结果，在异常结果中补全漏检的框。
6. 根据匹配结果，在异常结果中抑制虚警的框。（可选）
7. 保存处理后的标签文件。
8. 输出可视化

接下来，我将给出更为具体的执行逻辑：

# 1. 读取所有标签文件，提取检测框，计算中心点
读取所有的标签文件，将其进行编号，设编号后的区间是$[1, N]$. 对于编号为$i$的检测结果，记它的所有检测框构成列表$B_i=\{b^{(i)}_1, \dots, b^{(i)}_{n_i}\}$. 其中$b^{(i)}_j$是一个八元组，包含了检测框四个角点的坐标值。对于每一个检测框，以它的角点平均坐标作为中心点，构造中心点列表$P_i=\{p^{(i)}_1, \dots, p^{(i)}_{n_i}\}$，其中$p^{(i)}_j$是一个二元组，表示中心点的$(x,y)$坐标.

# 2. 进行聚类，找出异常结果和正常聚类
对所有检测结果进行聚类，聚类算法请自行选择，聚类参数通过命令行指定，请你自行给出默认值。结果之间的距离度量，通过中心点之间的二分匹配计算，具体逻辑如下：
1. 对于检测结果$P_i$和$P_j$，计算两个中心点列表当中，双方中心点两两之间的欧氏距离
2. 以中心点之间的欧氏距离作为路径权重，构建二分图$G(P_i\cup P_j, V)$
3. 对二分图$G$进行匹配，要求匹配后保留的路径，权重之和最小化，记最小化后的权重之和为$d_{i,j}$
4. $d_{i,j}$就是检测结果$P_i$和$P_j$之间的距离

聚类完成后，每一个聚类当中的检测结果视为同一场景下的正确检测结果；而离群点视为因漏检或虚警而产生的错误结果.

# 3. 对于每个正常聚类，找到它的代表结果
对于每一个聚类，将它当中距离聚类中心最近的一个检测结果，视为其代表，用于和异常结果进行匹配。

# 4. 对于每个异常结果，找到它匹配程度最高的代表结果
对于每一个异常结果$P^\text{a}_i$，计算它和所有聚类代表$P^\text{n}=\{P^\text{n}_1,\dots,P^\text{n}_k\}$之间的距离. 距离的计算方式与步骤2相同. 

假设异常结果$P^\text{a}_i$和$P^\text{n}_j$之间距离最短，则认为$P^\text{a}_i$是场景$j$当中的图像由于漏检或虚警产生的错误结果。匹配后的二分图$G(P^\text{a}_i\cup P^\text{n}_j, V)$保留用于步骤5和步骤6.

# 5. 根据匹配结果，在异常结果中补全漏检的框：
对于二分图$G(P^\text{a}_i\cup P^\text{n}_j, V)$进行剪枝，将路径距离大于阈值t_d的连接删去. 阈值t_d通过命令行指定，默认值是0.2（归一化坐标）.

找到$P^\text{n}_j$当中未能与$P^\text{a}_i$有连接的中心点，将其视为漏检目标中心点。假设$P^\text{n}_j[u]$是漏检目标的中心点。

重新打开images_root目录下，对应于$P^\text{a}_i$的图像，以$P^\text{n}_j[u]$为中心点，裁剪出宽度和高度为图像宽高八分之一的区域，记为X。另一方面，打开对应于$P^\text{n}_j$的图像，找到检测框$B^\text{n}_j[u]$，将框中的区域裁剪下来，通过补充0值扩充为水平矩形区域，记为K。

然后以K作为卷积核，对X进行卷积匹配。对于每一个卷积匹配的位置，提前将X的局部区域和K进行范数归一化，相当于计算余弦相似度。输出K与X局部的余弦相似度特征图。卷积的stride=1，padding根据K的实际尺寸设置，使得卷积后的特征图尺寸与卷积前一致。卷积操作通过pytorch库实现。

以余弦相似度特征图当中激活值最大的像素点，作为中心点，生成一个尺寸、方向和$B^\text{n}_j[u]$相同的检测框，并补充到$(P^\text{a}_i, B^\text{a}_i)$的检测结果当中。新生成的检测框，其置信度由一个基础的置信度base_conf叠加上余弦相似度，如果超过1则进行裁剪。base_conf是可调整的参数，默认是0.5.

# 6. 根据匹配结果，在异常结果中抑制虚警的框

找到步骤5当中，剪枝后的二分图里，未能与$P^\text{n}_j$匹配的中心点$P^\text{a}_i[v]$，视为虚警的目标。用一个命令行参数suppress决定是否将它消除。

# 7. 保存处理后的标签文件
将处理后的检测结果，以和输入时相同的格式，输出到目录processed_root下

# 8. 输出可视化
将每一个聚类当中的图像结果，连同其检测框，可视化输出到processed_root\vis\cluster_id当中，而对于归类到该聚类的异常结果可视化，也输出到这个目录，名称后缀带上"-anomaly"以进行区分. 这个功能利用一个命令行参数--vis控制是否开启，默认关闭

---

综上所述，需要在执行时指明的参数包括：1. 聚类算法的相关参数 2. 中心点欧氏距离的阈值t_d 3. 补充的检测框基础置信度base_conf 4. 相关的文件路径：images_root, labels_root, processed_labels_root 5. 控制是否执行消除虚警结果的参数--suppress 6. 控制是否可视化的参数--vis

这6个参数均通过命令行参数传入。

对于占用内存较大，可能导致溢出的部分，可以采用时间换空间的优化方式。其余部分尽可能优化时间复杂度。给出完整的实现，要求能够直接运行。