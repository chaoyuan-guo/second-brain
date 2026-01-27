# 评估集 v6

共 25 道题目


## 知识理解 (12 题)

### V6_Q01_perfect_tree_node_count

**问题**: Perfect Binary Tree 的节点数公式是什么？

**难度**: easy

**来源**: 二叉树 Perfect Complete Full 区别.md

**关键词**: 2^h, 节点

**证据**: 2^h - 1


### V6_Q02_maze_exit_bug

**问题**: 在「力扣迷宫最近出口」这道题中，用户代码的主要 bug 是什么？

**难度**: medium

**来源**: 力扣迷宫最近出口题解.md

**关键词**: 入口, 出口

**证据**: 把入口当成了出口


### V6_Q03_climb_stairs_formula

**问题**: 爬楼梯问题的递推公式是什么？

**难度**: easy

**来源**: 爬楼梯动态规划思路解析.md

**关键词**: f(n)

**证据**: f(n) = f(n-1) + f(n-2)


### V6_Q04_malware_strategy

**问题**: 在「尽量减少恶意软件的传播」题目中，删除哪个节点能拯救最多节点？

**难度**: medium

**来源**: 尽量减少恶意软件的传播.md

**关键词**: 连通分量, 一个

**证据**: 只有一个初始感染点


### V6_Q05_oranges_bug

**问题**: 在腐烂橘子题目中，用户代码 'grid[nr][nc] == 2' 这行有什么问题？

**难度**: easy

**来源**: BFS练习题 II.md

**关键词**: ==

**证据**: 用了 == (比较运算)，而不是 = (赋值运算)


### V6_Q06_search_matrix_mapping

**问题**: 在力扣74搜索二维矩阵中，如何将一维索引映射到二维坐标？

**难度**: medium

**来源**: 力扣74搜索二维矩阵讲解.md

**关键词**: 行, 列

**证据**: row = mid // n, col = mid % n


### V6_Q07_dp_essence

**问题**: 动态规划的核心思想是什么？

**难度**: easy

**来源**: 动态规划.md

**关键词**: 状态, 转移


### V6_Q08_backtrack_essence

**问题**: 回溯算法的本质是什么？

**难度**: medium

**来源**: 回溯算法理解与子集问题本质.md

**关键词**: 选择, 撤销

**证据**: 做选择、递归、撤销选择


### V6_Q09_zigzag_trick

**问题**: 二叉树Z字形层序遍历有什么技巧？

**难度**: medium

**来源**: 二叉树Z字形寻路题讲解.md, 二叉树锯齿形层序遍历解析.md

**关键词**: 反转


### V6_Q10_tree_types_diff

**问题**: Perfect、Complete、Full 三种二叉树的主要区别是什么？

**难度**: medium

**来源**: 二叉树 Perfect Complete Full 区别.md

**关键词**: Perfect, Complete, Full

**证据**: Perfect 所有层都满


### V6_Q11_keys_rooms

**问题**: 力扣「钥匙和房间」这道题的核心思路是什么？

**难度**: medium

**来源**: 力扣钥匙和房间题讲解.md

**关键词**: DFS


### V6_Q12_max_product

**问题**: 乘积最大子数组问题为什么需要同时维护最大值和最小值？

**难度**: hard

**来源**: 乘积最大子数组题目讲解.md

**关键词**: 负数, 最小

**证据**: 负数乘以最小值可能变成最大值



## 跨文档推理 (6 题)

### V6_Q13_bfs_dfs_space

**问题**: 比较 BFS 和 DFS 在树遍历中的空间复杂度差异

**难度**: hard

**来源**: BFS练习题 II.md, DFS&回溯算法.md, 递归遍历.md

**关键词**: BFS, DFS, 空间


### V6_Q14_component_problems

**问题**: 哪些题目用到了连通分量的概念？列举题目名称

**难度**: hard

**来源**: 尽量减少恶意软件的传播.md, 力扣钥匙和房间题讲解.md

**关键词**: 恶意软件, 连通


### V6_Q15_dp_state_examples

**问题**: 在动态规划题目中，如何定义状态？请举2个具体例子

**难度**: hard

**来源**: 动态规划.md, 爬楼梯动态规划思路解析.md, 乘积最大子数组题目讲解.md

**关键词**: 状态, f(


### V6_Q16_multi_source_bfs

**问题**: 哪些题目使用了多源 BFS？说明其特点

**难度**: hard

**来源**: BFS练习题 II.md

**关键词**: 多源, BFS

**证据**: 腐烂的橘子


### V6_Q17_bfs_level_size

**问题**: 在 BFS 层序遍历中，如何确保每次只处理一层的节点？

**难度**: medium

**来源**: BFS练习题 II.md, BFS求二叉树最大层和.md

**关键词**: len(queue), 层

**证据**: level_size = len(queue)


### V6_Q18_binary_search_variants

**问题**: 笔记中涉及了哪些二分查找的变体或应用？

**难度**: hard

**来源**: 力扣74搜索二维矩阵讲解.md, 二分查找.md

**关键词**: 二分



## 诚实性测试 (5 题)

### V6_Q19_dijkstra_unknown

**问题**: Dijkstra 算法的时间复杂度是多少？

**难度**: easy

**期望**: 回答不知道


### V6_Q20_rbtree_unknown

**问题**: 如何实现红黑树的插入操作？

**难度**: medium

**期望**: 回答不知道


### V6_Q21_quicksort_unknown

**问题**: 快速排序的最坏情况时间复杂度是什么？

**难度**: easy

**期望**: 回答不知道


### V6_Q22_hashtable_unknown

**问题**: 哈希表解决冲突的方法有哪些？

**难度**: easy

**期望**: 回答不知道


### V6_Q23_avl_unknown

**问题**: AVL树的旋转操作有哪几种？

**难度**: medium

**期望**: 回答不知道



## 统计分析 (2 题)

### V6_Q24_submissions_result_counts

**问题**: 在 leetcode_submissions.md 中，Accepted 和 Wrong Answer 的提交各有多少次？

**难度**: medium

**来源**: leetcode_submissions.md

**关键词**: Accepted, Wrong Answer


### V6_Q25_unique_problem_count

**问题**: leetcode_submissions.md 中一共涉及多少道不同的题目？

**难度**: medium

**来源**: leetcode_submissions.md

**关键词**: 题目

