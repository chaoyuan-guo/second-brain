# Agentic RAG 评估测试集 v3

说明：本版本强调“原文引用一致性”和“统计类问题的可计算性”。部分题目要求逐字引用原文（请用引号），并在回答中标注来源。


---

## 主测试集（单文档）

### V3_Q01_perfect_complete_full_quotes Perfect/Complete/Full 关键定义原文引用

- **Query**：请逐字引用原文中对 Perfect、Complete、Full 三种二叉树的关键定义短句（每种各引用一句），并在最后用一句话总结三者关系。
- **来源**：`data/notes/my_markdowns/二叉树 Perfect Complete Full 区别.md`
- **类型**：single_doc
- **难度**：medium
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any（允许引用来源中的任意原文片段，不要求命中特定证据句）
- **期望证据（必须）**：
  1. “每一层都被完全填满”
  2. “最后一层的节点都尽量靠左连续”
  3. “每个节点要么没有孩子要么正好有两个孩子”
- **通过判定**：三条证据必须逐字出现在答案中（允许换行），并能对应到来源。

### V3_Q02_complete_tree_bfs_rule 完全二叉树 BFS 空节点规则引用

- **Query**：用层序遍历判断完全二叉树时，原文对“遇到空节点后的规则”是怎么说的？请逐字引用并解释一句。
- **来源**：`data/notes/my_markdowns/力扣完全二叉树检验讲解.md`
- **类型**：single_doc
- **难度**：easy
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any（允许引用来源中的任意原文片段，不要求命中特定证据句）
- **期望证据（必须）**：
  1. “如果之后又遇到非空节点，说明中间有缺口”
- **通过判定**：必须逐字引用该短句，并给出一句解释。

### V3_Q03_search_matrix_mapping 搜索二维矩阵索引映射引用

- **Query**：请逐字引用原文中把一维索引映射回二维坐标的两条公式。
- **来源**：`data/notes/my_markdowns/力扣74搜索二维矩阵讲解.md`
- **类型**：single_doc
- **难度**：easy
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any（允许引用来源中的任意原文片段，不要求命中特定证据句）
- **期望证据（必须）**：
  1. “row = mid // n”
  2. “col = mid % n”
- **通过判定**：两条公式必须逐字出现在答案中。

### V3_Q04_climb_stairs_formula 爬楼梯递推公式引用

- **Query**：请逐字引用该笔记中的递推公式（含空格与符号）并说明它表达的含义。
- **来源**：`data/notes/my_markdowns/爬楼梯动态规划思路解析.md`
- **类型**：single_doc
- **难度**：easy
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any（允许引用来源中的任意原文片段，不要求命中特定证据句）
- **期望证据（必须）**：
  1. “f(n) = f(n - 1) + f(n - 2)”
- **通过判定**：公式必须逐字出现，且需给出含义解释。

### V3_Q05_malware_components_quote 恶意软件传播连通分量原文引用

- **Query**：请逐字引用原文里关于连通分量之间关系的那句话，并说明它如何影响删除策略。
- **来源**：`data/notes/my_markdowns/尽量减少恶意软件的传播.md`
- **类型**：single_doc
- **难度**：easy
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any（允许引用来源中的任意原文片段，不要求命中特定证据句）
- **期望证据（必须）**：
  1. “图的连通分量之间互不影响”
- **通过判定**：必须逐字引用并解释其对策略的影响。

### V3_Q06_evaluate_division_reverse_edge 除法求值反向边权重引用

- **Query**：请逐字引用原文中对反向边权重的描述（包含 1/k），并说明为什么需要这条反向边。
- **来源**：`data/notes/my_markdowns/力扣除法求值题讲解.md`
- **类型**：single_doc
- **难度**：easy
- **允许未知**：否
- **引用要求**：必须有引号的逐字引用
- **证据策略**：any（允许引用来源中的任意原文片段，不要求命中特定证据句）
- **期望证据（必须）**：
  1. “`b -> a` 权重为 `1/k`”
- **通过判定**：必须逐字引用并解释反向边的用途。

### V3_Q07_submissions_result_counts 提交结果总统计

- **Query**：请统计 leetcode_submissions.md 中所有提交的结果分布，输出 JSON：{"total":?, "accepted":?, "wrong_answer":?, "runtime_error":?, "mle":?, "tle":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
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

### V3_Q08_submissions_rates Accepted 与 Runtime Error 占比

- **Query**：请基于 leetcode_submissions.md 统计 Accepted 与 Runtime Error 占总提交的百分比（保留两位小数），输出 JSON：{"accepted_rate":"xx.xx%", "runtime_error_rate":"yy.yy%"}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. accepted_rate=54.06%
  2. runtime_error_rate=21.08%
- **通过判定**：百分比保留两位小数且数值准确。

### V3_Q09_lps_accept_re 最长回文子序列提交统计

- **Query**：在 leetcode_submissions.md 中，统计“最长回文子序列”提交里 Accepted 和 Runtime Error 分别是多少？请给出数字。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **难度**：medium
- **允许未知**：否
- **期望要点（必须）**：
  1. Accepted=3
  2. Runtime Error=2
- **通过判定**：数字正确且语义清晰。

### V3_Q10_circular_subarray_wa 环形子数组最大和 WA 统计

- **Query**：在 leetcode_submissions.md 中，“环形子数组的最大和”提交的 Wrong Answer 和 Accepted 分别有多少次？
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **难度**：medium
- **允许未知**：否
- **期望要点（必须）**：
  1. Wrong Answer=2
  2. Accepted=2
- **通过判定**：数字正确且语义清晰。

### V3_Q11_accepted_runtime_extremes Accepted 耗时最小/最大值

- **Query**：统计 leetcode_submissions.md 中所有 Accepted 提交的耗时最小值和最大值（单位 ms），输出 JSON：{"min_ms":?, "max_ms":?}。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **难度**：hard
- **允许未知**：否
- **期望要点（必须）**：
  1. min_ms=0
  2. max_ms=3220
- **通过判定**：JSON 字段完整且数值准确。

### V3_Q12_non_ac_total 非 Accepted 总量

- **Query**：在 leetcode_submissions.md 中，所有非 Accepted 的提交总数是多少？请给出数字并简要说明计算过程。
- **来源**：`data/notes/my_markdowns/leetcode_submissions.md`
- **类型**：single_doc
- **难度**：medium
- **允许未知**：否
- **期望要点（必须）**：
  1. 非 Accepted 总数=486
- **通过判定**：数字正确，并说明计算来源。
