# 测试集

## 概述

| 指标 | 数值 |
|------|------|
| 总题目数 | 67 |

## 类别分布

| 类别 | 数量 | 说明 |
|------|------|------|
| understanding | 24 | 理解类：单文档内容理解 |
| reasoning | 19 | 推理类：跨文档比较、归纳总结 |
| negative | 10 | 负例类：应回答"不知道"的问题 |
| statistics | 9 | 统计类：数值计算、数据分析 |
| skill | 3 | 技能类：需要使用特定技能工具 |
| web_search | 2 | 联网类：需要联网搜索的问题 |

## 改进特性

### 同义词/正则匹配
- 支持全局同义词（如"动态规划" ↔ "DP"/"dynamic programming"）
- 支持题目级同义词配置
- 支持正则表达式匹配

### 精确 Negative 题目
- Q53-Q56：问的是"笔记中明确不存在的内容"
- 设计原则：引用具体笔记上下文，但问不存在的细节

### 动态验证
- Statistics 题目支持 `$stats.xxx` 动态值引用
- 通过 `precompute_stats.py` 预计算统计数据

### 检索指标
- 记录实际检索结果
- 计算 precision/recall/f1 指标
- 生成检索分析报告

### 权重调整
- citation 权重从 0.2 降至 0.1（加分项）
- 各类别权重更加合理

---

## 完整题目列表

### Q01_perfect_tree_node_count

- **问题**: Perfect Binary Tree 的节点数公式是什么？
- **类别**: understanding
- **难度**: easy
- **期望来源**: 二叉树 Perfect Complete Full 区别.md
- **同义词**: evidence 中配置了公式的多种表达形式

### Q02_maze_exit_bug

- **问题**: 在「力扣迷宫最近出口」这道题中，用户代码的主要 bug 是什么？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 力扣迷宫最近出口题解.md

### Q03_climb_stairs_formula

- **问题**: 爬楼梯问题的递推公式是什么？
- **类别**: understanding
- **难度**: easy
- **期望来源**: 爬楼梯动态规划思路解析.md

### Q04_malware_strategy

- **问题**: 在「尽量减少恶意软件的传播」题目中，删除哪个节点能拯救最多节点？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 尽量减少恶意软件的传播.md

### Q05_oranges_bug

- **问题**: 在腐烂橘子题目中，用户代码 'grid[nr][nc] == 2' 这行有什么问题？
- **类别**: understanding
- **难度**: easy
- **期望来源**: BFS练习题 II.md

### Q06_search_matrix_mapping

- **问题**: 在力扣74搜索二维矩阵中，如何将一维索引映射到二维坐标？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 力扣74搜索二维矩阵讲解.md

### Q07_dp_essence

- **问题**: 动态规划的核心思想是什么？
- **类别**: understanding
- **难度**: easy
- **期望来源**: 动态规划.md

### Q08_backtrack_essence

- **问题**: 回溯算法的本质是什么？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 回溯算法理解与子集问题本质.md

### Q09_zigzag_trick

- **问题**: 二叉树Z字形层序遍历有什么技巧？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 二叉树Z字形寻路题讲解.md, 二叉树锯齿形层序遍历解析.md

### Q10_tree_types_diff

- **问题**: Perfect、Complete、Full 三种二叉树的主要区别是什么？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 二叉树 Perfect Complete Full 区别.md

### Q11_keys_rooms

- **问题**: 力扣「钥匙和房间」这道题的核心思路是什么？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 力扣钥匙和房间题讲解.md

### Q12_max_product

- **问题**: 乘积最大子数组问题为什么需要同时维护最大值和最小值？
- **类别**: understanding
- **难度**: hard
- **期望来源**: 乘积最大子数组题目讲解.md

### Q13_bfs_dfs_space

- **问题**: 比较 BFS 和 DFS 在树遍历中的空间复杂度差异
- **类别**: reasoning
- **难度**: hard
- **期望来源**: BFS练习题 II.md, DFS&回溯算法.md, 递归遍历.md

### Q14_component_problems

- **问题**: 哪些题目用到了连通分量的概念？列举题目名称
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 尽量减少恶意软件的传播.md, 力扣钥匙和房间题讲解.md, BFS练习题 II.md

### Q15_dp_state_examples

- **问题**: 在动态规划题目中，如何定义状态？请举2个具体例子
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 动态规划.md, 爬楼梯动态规划思路解析.md, 乘积最大子数组题目讲解.md

### Q16_multi_source_bfs

- **问题**: 哪些题目使用了多源 BFS？说明其特点
- **类别**: reasoning
- **难度**: hard
- **期望来源**: BFS练习题 II.md

### Q17_bfs_level_size

- **问题**: 在 BFS 层序遍历中，如何确保每次只处理一层的节点？
- **类别**: reasoning
- **难度**: medium
- **期望来源**: BFS练习题 II.md, BFS求二叉树最大层和.md

### Q18_binary_search_variants

- **问题**: 笔记中涉及了哪些二分查找的变体或应用？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 力扣74搜索二维矩阵讲解.md

### Q19_dijkstra_unknown

- **问题**: Dijkstra 算法的时间复杂度是多少？
- **类别**: negative
- **难度**: easy
- **设计理由**: 笔记中未涉及 Dijkstra 算法

### Q20_rbtree_unknown

- **问题**: 如何实现红黑树的插入操作？
- **类别**: negative
- **难度**: medium
- **设计理由**: 笔记中未涉及红黑树

### Q21_quicksort_unknown

- **问题**: 快速排序的最坏情况时间复杂度是什么？
- **类别**: negative
- **难度**: easy
- **设计理由**: 笔记中未涉及快速排序

### Q22_hashtable_unknown

- **问题**: 哈希表解决冲突的方法有哪些？
- **类别**: negative
- **难度**: easy
- **设计理由**: 笔记中未涉及哈希表冲突解决

### Q23_avl_unknown

- **问题**: AVL树的旋转操作有哪几种？
- **类别**: negative
- **难度**: medium
- **设计理由**: 笔记中未涉及 AVL 树

### Q24_submissions_result_counts

- **问题**: 在 leetcode_submissions.md 中，Accepted 和 Wrong Answer 的提交各有多少次？
- **类别**: statistics
- **难度**: medium
- **期望来源**: leetcode_submissions.md
- **数值校验**: 已配置

### Q25_unique_problem_count

- **问题**: leetcode_submissions.md 中一共涉及多少道不同的题目？
- **类别**: statistics
- **难度**: medium
- **期望来源**: leetcode_submissions.md
- **数值校验**: 已配置

### Q26_sliding_puzzle_bfs

- **问题**: 滑动谜题这道题为什么要用 BFS 而不是 DFS？状态空间有多大？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 滑动谜题BFS解题思路.md

### Q27_min_height_tree_center

- **问题**: 最小高度树问题中，为什么树的中心点只能有 1 个或 2 个？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 最小高度树剥洋葱找树中心.md

### Q28_topological_peel_onion

- **问题**: 剥洋葱法是什么算法思想？在哪些题目中用到？
- **类别**: reasoning
- **难度**: medium
- **期望来源**: 最小高度树剥洋葱找树中心.md

### Q29_bfs_two_implementations

- **问题**: BFS 有哪两种常见的写法？什么时候需要用 level_size = len(queue)？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 滑动谜题BFS解题思路.md

### Q30_bidirectional_bfs

- **问题**: 什么是双向 BFS？它比单向 BFS 快多少？
- **类别**: understanding
- **难度**: hard
- **期望来源**: 滑动谜题BFS解题思路.md

### Q31_open_lock_deadends

- **问题**: 打开转盘锁这道题中，deadends 应该怎么处理？为什么要用集合？
- **类别**: understanding
- **难度**: easy
- **期望来源**: 滑动谜题BFS解题思路.md

### Q32_web_search_fallback

- **问题**: Python 3.13 有什么新特性？
- **类别**: web_search
- **难度**: medium
- **备注**: 测试联网降级：本地无结果时应触发 web_search 工具

### Q33_web_search_current_events

- **问题**: 2025 年 LeetCode 周赛排名第一的选手是谁？
- **类别**: web_search
- **难度**: hard
- **备注**: 测试联网搜索时事信息

### Q34_skill_load_explicit

- **问题**: 帮我统计 LeetCode 提交记录中各难度题目的数量分布
- **类别**: skill
- **难度**: medium
- **期望来源**: leetcode_submissions.md
- **备注**: 测试 load_skill 工具的显式加载

### Q35_multi_file_analysis

- **问题**: 对比所有 BFS 相关笔记中的时间复杂度，哪个问题的状态空间最大？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 滑动谜题BFS解题思路.md, BFS练习题 II.md, 最小高度树剥洋葱找树中心.md
- **备注**: 测试多轮工具调用：需要查询多个文件并进行对比分析

### Q36_code_interpreter_stats

- **问题**: 用代码计算我所有笔记文件的总字数和平均字数
- **类别**: statistics
- **难度**: medium
- **期望工具**: run_code_interpreter
- **工具权重**: 0.1
- **备注**: 测试 run_code_interpreter 的文件统计场景

### Q37_code_interpreter_complexity

- **问题**: 用代码分析 leetcode_submissions.md 中提交时间的分布，按月统计提交次数
- **类别**: statistics
- **难度**: hard
- **期望来源**: leetcode_submissions.md
- **期望工具**: run_code_interpreter
- **工具权重**: 0.1
- **备注**: 测试 run_code_interpreter 的数据分析场景

### Q38_negative_graceful

- **问题**: 笔记中有关于红黑树的讲解吗？
- **类别**: negative
- **难度**: easy
- **备注**: 测试本地无结果时的优雅处理

### Q39_binary_matrix_shortest_path

- **问题**: 二进制矩阵最短路径问题的解题思路是什么？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 二进制矩阵最短路径题讲解.md

### Q40_gene_mutation_bfs

- **问题**: 最小基因变化问题和打开转盘锁有什么相似之处？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 最小基因变化题解.md, 滑动谜题BFS解题思路.md

### Q41_jump_game_iii

- **问题**: 跳跃游戏 III 为什么可以用 BFS 或 DFS 解决？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 跳跃游戏III题解.md

### Q42_tree_distance_k

- **问题**: 二叉树中距离为 K 的节点问题，为什么要先把树转成图？
- **类别**: understanding
- **难度**: hard
- **期望来源**: 二叉树距离为K节点解析.md

### Q43_complete_tree_inserter

- **问题**: 完全二叉树插入器是怎么实现的？用了什么数据结构？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 完全二叉树插入器题解.md

### Q44_odd_even_tree

- **问题**: 奇偶树的判断条件是什么？需要检查哪些性质？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 力扣奇偶树题目讲解.md

### Q45_evaluate_division_graph

- **问题**: 除法求值问题为什么要建图？图的边权代表什么？
- **类别**: understanding
- **难度**: hard
- **期望来源**: 力扣除法求值题讲解.md

### Q46_complete_tree_check

- **问题**: 如何判断一棵二叉树是不是完全二叉树？核心思路是什么？
- **类别**: understanding
- **难度**: medium
- **期望来源**: 力扣完全二叉树检验讲解.md

### Q47_constant_space_next_pointer

- **问题**: 如何用常数空间连接二叉树的下一层指针？为什么普通 BFS 不行？
- **类别**: understanding
- **难度**: hard
- **期望来源**: 常数空间连接二叉树下一层指针.md

### Q48_multi_doc_comparison

- **问题**: 对比所有 BFS 相关笔记，哪个问题的状态空间最大？请列出各题目的状态空间大小并给出结论
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 滑动谜题BFS解题思路.md, BFS练习题 II.md, 最小高度树剥洋葱找树中心.md, 力扣迷宫最近出口题解.md
- **备注**: 多文档对比 + 推理：需要查询多个 BFS 相关文档并对比分析

### Q49_pattern_summary

- **问题**: 找出我做过的所有使用「多源 BFS」技巧的题目，并总结它们的共同模式
- **类别**: reasoning
- **难度**: hard
- **期望来源**: BFS练习题 II.md
- **备注**: 多轮检索 + 归纳：需要检索多个文档并总结共同模式

### Q50_submission_monthly_analysis

- **问题**: 用代码分析 leetcode_submissions.md，找出提交次数最多的月份，并计算该月的通过率
- **类别**: statistics
- **难度**: hard
- **期望来源**: leetcode_submissions.md
- **期望工具**: run_code_interpreter
- **工具权重**: 0.1
- **备注**: 检索 + 代码执行：需要先检索文件内容，再用代码分析计算

### Q51_algorithm_evolution

- **问题**: 从我的笔记中，总结 BFS 和 DFS 在不同场景下的选择策略，至少给出3个具体例子
- **类别**: reasoning
- **难度**: hard
- **期望来源**: BFS练习题 II.md, DFS&回溯算法.md, 滑动谜题BFS解题思路.md, 力扣钥匙和房间题讲解.md
- **备注**: 多文档综合 + 归纳：需要从多个文档中提取信息并进行综合分析

### Q52_cross_topic_connection

- **问题**: 动态规划和 BFS 有什么联系？在我的笔记中有哪些题目同时涉及这两种方法？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 动态规划.md, BFS练习题 II.md, 滑动谜题BFS解题思路.md
- **备注**: 跨主题关联：需要理解两种算法的联系并从笔记中找到例证

---

## 新增精确 Negative 题目 (Q53-Q56)

### Q53_neg_nonexistent_problem_number

- **问题**: 笔记中的力扣题目 1024 是关于什么问题的？
- **类别**: negative
- **难度**: easy
- **设计理由**: 笔记中没有 1024 这个题号

### Q54_neg_nonexistent_variable

- **问题**: 在腐烂橘子题解中，变量 visited_set 的作用是什么？
- **类别**: negative
- **难度**: medium
- **设计理由**: 腐烂橘子题解用原地修改，没有 visited_set 变量

### Q55_neg_nonexistent_complexity

- **问题**: 滑动谜题笔记中提到的 O(n!) 复杂度是针对什么情况的？
- **类别**: negative
- **难度**: medium
- **设计理由**: 滑动谜题笔记只提到 6!=720，没有 O(n!) 的讨论

### Q56_neg_nonexistent_section

- **问题**: BFS练习题笔记的「性能优化」章节讲了什么内容？
- **类别**: negative
- **难度**: easy
- **设计理由**: BFS练习题笔记中没有「性能优化」章节

---

## 新增 Statistics 题目 (Q57-Q60)

### Q57_stat_most_common_tag

- **问题**: leetcode_submissions.md 中哪个标签的题目数量最多？
- **类别**: statistics
- **难度**: medium
- **期望来源**: leetcode_submissions.md
- **验证点**: 动态规划标签

### Q58_stat_runtime_error_count

- **问题**: 我的 LeetCode 提交记录中有多少次 Runtime Error？
- **类别**: statistics
- **难度**: easy
- **期望来源**: leetcode_submissions.md
- **验证方式**: 动态验证（$stats.xxx）

### Q59_stat_pass_rate

- **问题**: 计算我的 LeetCode 提交通过率是多少？
- **类别**: statistics
- **难度**: medium
- **期望来源**: leetcode_submissions.md
- **期望工具**: run_code_interpreter

### Q60_stat_date_range

- **问题**: 我的 LeetCode 提交记录覆盖了哪个时间范围？
- **类别**: statistics
- **难度**: easy
- **期望来源**: leetcode_submissions.md

---

## 新增 Skill 题目 (Q61-Q62)

### Q61_skill_wrong_answer_analysis

- **问题**: 统计我提交 Wrong Answer 次数最多的题目是哪些？列出前3名
- **类别**: skill
- **难度**: hard
- **期望来源**: leetcode_submissions.md
- **期望工具**: run_code_interpreter, load_skill

### Q62_skill_daily_submission

- **问题**: 计算我平均每天提交多少次 LeetCode？
- **类别**: skill
- **难度**: medium
- **期望来源**: leetcode_submissions.md
- **期望工具**: run_code_interpreter

---

## 新增 Reasoning 跨文档题目 (Q63-Q67)

### Q63_reason_bfs_common_patterns

- **问题**: BFS练习题和滑动谜题两篇笔记中，共同提到了哪些 BFS 技巧或变体？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: BFS练习题 II.md, 滑动谜题BFS解题思路.md

### Q64_reason_state_concept

- **问题**: 动态规划和 BFS 都涉及「状态」概念，它们的定义有什么异同？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: 动态规划.md, 滑动谜题BFS解题思路.md

### Q65_reason_graph_modeling

- **问题**: 比较「账户合并」和「水壶问题」两道题的建图思路有什么共同点？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: BFS练习题 II.md

### Q66_reason_recursive_iterative

- **问题**: 哪些题目同时给出了递归和迭代两种解法？
- **类别**: reasoning
- **难度**: medium
- **期望来源**: 动态规划.md, 递归遍历.md

### Q67_reason_common_bugs

- **问题**: 笔记中提到的代码 bug 有哪些共同的错误模式？
- **类别**: reasoning
- **难度**: hard
- **期望来源**: BFS练习题 II.md, 力扣迷宫最近出口题解.md
