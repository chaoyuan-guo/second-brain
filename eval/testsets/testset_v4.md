# Agentic RAG 评估测试集 v4

说明：本版本强调“用户误解纠偏/冲突识别/严格引用”和“统计类计算能力”。统计题必须基于表格记录计算，不可凭空估算。

类型说明：
- explicit_source：Query 明确指定某一笔记/文件作为来源。
- implicit_unique：Query 未指定具体来源，但在笔记库中只存在单一可命中的原文来源。
- multi_source：Query 不指定具体来源，且允许命中多份笔记中的任意一个来源。

---

## A. 用户误解纠偏 / 冲突 / 引用能力

### V4_Q01_perfect_tree_node_count_conflict Perfect 二叉树节点数冲突

- **Query**：请在《二叉树 Perfect Complete Full 区别.md》中逐字引用两句关于 Perfect 二叉树节点总数的表述（每句只引用包含节点总数公式的短句），并用一句话说明这两句存在口径不一致（说明时不要使用引号）。
- **来源**：`data/notes/my_markdowns/二叉树 Perfect Complete Full 区别.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：medium
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：strict
- **期望证据（必须）**：
  1. “高度为 h 时节点数一定是 \(2^{h+1}-1\)”
  2. “如果树的高度是 `h`，那么节点总数一定是 `2^h - 1`。”
- **通过判定**：两条证据必须逐字出现在答案中，且明确说明存在不一致/口径差异。

### V4_Q02_maze_exit_bug 迷宫最近出口入口误判

- **Query**：请在《力扣迷宫最近出口题解.md》中逐字引用用户提问“下面我的代码哪里出了问题？”这句，以及助手指出的核心错误短句，并用一句话说明正确的出口判定应排除入口。
- **来源**：`data/notes/my_markdowns/力扣迷宫最近出口题解.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：medium
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：strict
- **期望证据（必须）**：
  1. “下面我的代码哪里出了问题？”
  2. “把入口当成了出口”
- **通过判定**：两条证据必须逐字出现在答案中，且说明入口不能作为出口。

### V4_Q03_complete_tree_complexity 完全二叉树索引法复杂度误解

- **Query**：请在《力扣完全二叉树检验讲解.md》中逐字引用包含“索引法”的用户提问与助手给出的结论短句，并补充一句说明时间复杂度层级。
- **来源**：`data/notes/my_markdowns/力扣完全二叉树检验讲解.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：medium
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：strict
- **期望证据（必须）**：
  1. “索引法的复杂度会更低吗”
  2. “不会，本质上两种方法**时间复杂度和空间复杂度是一个级别的**。”
- **通过判定**：两条证据必须逐字出现在答案中，且说明时间复杂度为 O(n) 量级。

### V4_Q04_dp_sort_confusion 信封问题同宽降序理由

- **Query**：请在《动态规划.md》中逐字引用用户疑问“还是没懂为什么在 w 相同时 h 需要降序”这句，以及包含“LIS 误判”为核心的助手解释句，并用一句话总结为什么同宽要按高度降序。
- **来源**：`data/notes/my_markdowns/动态规划.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：medium
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：strict
- **期望证据（必须）**：
  1. “还是没懂为什么在 w 相同时 h 需要降序”
  2. “如果你按 `w` 升序、`h` 也升序排序，那么当 `w` 相同的时候，`h` 递增会被 LIS 误判为可嵌套（但宽度没变，不能嵌套）。”
- **通过判定**：两条证据必须逐字出现在答案中，且说明同宽降序是为避免同宽被 LIS 误选。

### V4_Q05_bfs_level_size_confusion BFS level_size 误解

- **Query**：请在《滑动谜题BFS解题思路.md》中逐字引用用户关于 level_size 的提问，以及包含“(状态, 步数)”的助手解释句，并用一句话说明原因。
- **来源**：`data/notes/my_markdowns/滑动谜题BFS解题思路.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：medium
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any_of
- **期望证据（必须）**：
  1. “这两个其实是 **两种 BFS 写法**，这题用的是其中一种，所以不需要 `level_size = len(queue)`。”
  2. “**只要你在队列里存的是 `(状态, 步数)`，就不需要 `level_size`；”
  3. “queue.append((start, 0))  # (状态, 到达该状态的步数)”
  4. “队列里放的是 `(state, steps)`。”
- **通过判定**：逐字引用任一证据句，且解释原因是队列中已携带步数。

### V4_Q06_search_matrix_mapping 二维矩阵索引映射公式

- **Query**：请在《力扣74搜索二维矩阵讲解.md》中逐字引用把一维索引映射回二维坐标的两条公式。
- **来源**：`data/notes/my_markdowns/力扣74搜索二维矩阵讲解.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：easy
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：strict
- **期望证据（必须）**：
  1. “row = mid // n”
  2. “col = mid % n”
- **通过判定**：两条公式必须逐字出现在答案中。

### V4_Q07_climb_stairs_formula 爬楼梯递推公式

- **Query**：请在《爬楼梯动态规划思路解析.md》中逐字引用递推公式（含空格与符号），并用一句话说明它表示的含义。
- **来源**：`data/notes/my_markdowns/爬楼梯动态规划思路解析.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：easy
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：strict
- **期望证据（必须）**：
  1. “f(n) = f(n - 1) + f(n - 2)”
- **通过判定**：公式必须逐字出现，且说明它表示到第 n 阶的爬法/方法数递推关系。

### V4_Q08_malware_components_quote 恶意软件连通分量关系

- **Query**：请在《尽量减少恶意软件的传播.md》中逐字引用关于连通分量关系的短句（只引用该短句本身，不要加冒号或解释），并用一句话说明它对删除策略的影响。
- **来源**：`data/notes/my_markdowns/尽量减少恶意软件的传播.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：easy
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：strict
- **期望证据（必须）**：
  1. “图的连通分量之间互不影响”
- **通过判定**：引用该短句并说明删除策略可分量独立处理。

### V4_Q09_lps_state_definition_multi LPS 状态定义多来源引用

- **Query**：请逐字引用原文里对“dp[i][j] 表示 s[i..j] 的最长回文子序列长度”的描述，并说明这是哪类 DP。
- **来源**：
  - `data/notes/my_markdowns/动态规划.md`
  - `data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：multi_doc
- **case_type**：multi_source
- **难度**：medium
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any_of（命中任一来源的证据句即可）
- **期望证据（任选其一）**：
  1. “dp[i][j] 表示 s[i..j] 的最长回文子序列长度。”
- **通过判定**：逐字引用该句，并说明它属于区间 DP。

### V4_Q10_division_reverse_edge_multi 除法求值反向边权重引用

- **Query**：请逐字引用原文里对反向边权重 1/k 的描述，并用一句话说明其含义。
- **来源**：
  - `data/notes/my_markdowns/力扣除法求值题讲解.md`
  - `data/notes/my_markdowns/BFS练习题 II.md`
  - `data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：multi_doc
- **case_type**：multi_source
- **难度**：medium
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any_of（命中任一来源的证据句即可）
- **期望证据（任选其一）**：
  1. “`b -> a` 权重为 `1/k`”
  2. “`b -> a` 权重 1/k”
  3. “b→a 权重 1/k”
- **通过判定**：逐字引用任一证据句，并说明反向边表示可逆比例关系。

---

## B. 统计与聚合能力（leetcode_submissions.md）

### V4_Q11_submissions_result_counts 提交结果总统计

- **Query**：请统计 leetcode_submissions.md 中所有提交的结果分布，输出 JSON：{"total":?,"accepted":?,"wrong_answer":?,"runtime_error":?,"mle":?,"tle":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：medium
- **允许未知**：否
- **期望要点（必须）**：
  1. total=1058
  2. accepted=572
  3. wrong_answer=240
  4. runtime_error=223
  5. mle=12
  6. tle=11
- **通过判定**：JSON 字段完整且数值准确。

### V4_Q12_submissions_rates Accepted 与 Runtime Error 占比

- **Query**：请基于 leetcode_submissions.md 统计 Accepted 与 Runtime Error 占总提交的百分比（保留两位小数），输出 JSON：{"accepted_rate":"xx.xx%","runtime_error_rate":"yy.yy%"}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. accepted_rate=54.06%
  2. runtime_error_rate=21.08%
- **通过判定**：百分比保留两位小数且数值准确。

### V4_Q13_unique_problem_count 题目数量统计

- **Query**：统计 leetcode_submissions.md 中包含的不同题目数量（题目标题格式为 “## 题目名 (`slug`)”），并给出总提交数，输出 JSON：{"unique_problems":?,"total_submissions":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：medium
- **允许未知**：否
- **期望要点（必须）**：
  1. unique_problems=220
  2. total_submissions=1058
- **通过判定**：JSON 字段完整且数值准确。

### V4_Q14_top_submission_problem 提交次数最多的题目

- **Query**：在 leetcode_submissions.md 中，找出提交次数最多的题目及其提交次数（题目标题格式为 “## 题目名 (`slug`)”），输出 JSON：{"title":"...","submissions":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. title="岛屿数量"
  2. submissions=28
- **通过判定**：JSON 字段完整且数值准确。

### V4_Q15_submissions_on_date 指定日期提交统计

- **Query**：统计 leetcode_submissions.md 中 2026-01-13 这一天的总提交数与 Accepted 数量，输出 JSON：{"date":"2026-01-13","total":?,"accepted":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. total=8
  2. accepted=5
- **通过判定**：JSON 字段完整且数值准确。

### V4_Q16_accepted_runtime_median Accepted 耗时中位数

- **Query**：统计 leetcode_submissions.md 中所有 Accepted 提交的耗时中位数（单位 ms，保留 1 位小数），输出 JSON：{"median_ms":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. median_ms=43.5
- **通过判定**：JSON 字段完整且数值准确。

### V4_Q17_accepted_runtime_avg Accepted 耗时平均值

- **Query**：统计 leetcode_submissions.md 中所有 Accepted 提交的平均耗时（单位 ms，保留两位小数），输出 JSON：{"avg_ms":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. avg_ms=184.47
- **通过判定**：JSON 字段完整且数值准确。

### V4_Q18_accepted_memory_extremes Accepted 内存极值

- **Query**：统计 leetcode_submissions.md 中所有 Accepted 提交的内存最小值与最大值（单位 MB，保留 1 位小数），输出 JSON：{"min_mb":?,"max_mb":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. min_mb=16.8
  2. max_mb=80.1
- **通过判定**：JSON 字段完整且数值准确。

### V4_Q19_accepted_runtime_zero Accepted 零耗时次数

- **Query**：统计 leetcode_submissions.md 中 Accepted 且耗时为 0 ms 的提交次数，输出 JSON：{"runtime_zero_ms":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. runtime_zero_ms=106
- **通过判定**：JSON 字段完整且数值准确。

### V4_Q20_no_accepted_problem 无 Accepted 的题目

- **Query**：在 leetcode_submissions.md 中找出唯一没有 Accepted 的题目，并给出其结果分布，输出 JSON：{"title":"...","accepted":0,"runtime_error":?,"tle":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **case_type**：explicit_source
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. title="排序数组"
  2. accepted=0
  3. runtime_error=1
  4. tle=1
- **通过判定**：JSON 字段完整且数值准确。
