import test from "node:test";
import assert from "node:assert/strict";

import { extractEvidenceAnchor } from "./native-evidence-snippet.js";

test("extractEvidenceAnchor removes conversation wrappers and reanchors snippet", () => {
  const content = [
    "-",
    "## 🤖 Assistant",
    "",
    "path.pop() 的作用是撤销上一步选择，恢复到当前层的干净状态。",
    "",
    "## 🧑‍💻 User",
  ].join("\n");

  const result = extractEvidenceAnchor(content, 140);

  assert.equal(result.snippet, "path.pop() 的作用是撤销上一步选择，恢复到当前层的干净状态。");
  assert.ok(result.charOffset > 140);
});

test("extractEvidenceAnchor preserves code blocks and avoids mid-sentence cuts", () => {
  const content = [
    "```python",
    "if grid[nr][nc] == 1:",
    "    grid[nr][nc] = 2",
    "    fresh -= 1",
    "```",
    "",
    "这里真正的 bug 是把赋值写成了比较，导致 fresh 会被重复扣减。",
  ].join("\n");

  const result = extractEvidenceAnchor(content, 3200);

  assert.match(result.snippet, /grid\[nr\]\[nc\] = 2/);
  assert.ok(result.charOffset >= 3200);
  assert.ok(result.snippet.length <= 240);
});

test("extractEvidenceAnchor keeps stable offsets for plain text evidence", () => {
  const content = "提交 690994503 的 bug 有两个：n 未定义；return dp[n][n-1] 越界。";

  const first = extractEvidenceAnchor(content, 900);
  const second = extractEvidenceAnchor(content, 900);

  assert.deepEqual(second, first);
});

test("extractEvidenceAnchor skips summary metadata in submission notes", () => {
  const content = [
    "# 近一年 LeetCode 提交汇总",
    "",
    "- 统计时间范围：2025-01-13 起至 2026-01-13 统计",
    "- 总提交数：1058",
    "",
    "## 最长回文子序列",
    "",
    "### 提交记录",
    "| 提交ID | 时间 | 结果 |",
    "| --- | --- | --- |",
    "| 690994503 | 2026-01-13 13:12:31 CST | Runtime Error |",
    "",
    "#### 提交 690994503 · Runtime Error · 2026-01-13 13:12:31 CST · python3",
    "",
    "```python",
    "class Solution:",
    "    def longestPalindromeSubseq(self, s: str) -> int:",
    "        dp = [[0] * n for _ in range(n)]",
    "        return dp[n][n - 1]",
    "```",
  ].join("\n");

  const result = extractEvidenceAnchor(content, 21);

  assert.match(result.snippet, /longestPalindromeSubseq/);
  assert.doesNotMatch(result.snippet, /统计时间范围/);
  assert.ok(result.charOffset > 21);
});

test("extractEvidenceAnchor prefers failed submission block over notebook summary", () => {
  const content = [
    "## 最长回文子序列 (`longest-palindromic-subsequence`)",
    "",
    "### 题目笔记",
    "#### 笔记 1 · 更新于 2026-01-13 13:04:20 CST",
    "",
    "```markdown",
    "用区间 DP。",
    "```",
    "",
    "### 未通过提交代码",
    "#### 提交 690994503 · Runtime Error · 2026-01-13 13:12:31 CST · python3",
    "",
    "```python",
    "class Solution:",
    "    def longestPalindromeSubseq(self, s: str) -> int:",
    "        dp = [[0] * n for _ in range(n)]",
    "        return dp[n][n - 1]",
    "```",
    "",
    "#### 提交 690994572 · Runtime Error · 2026-01-13 13:12:55 CST · python3",
    "",
    "```python",
    "class Solution:",
    "    def longestPalindromeSubseq(self, s: str) -> int:",
    "        n = len(s)",
    "        return dp[n][n - 1]",
    "```",
  ].join("\n");

  const result = extractEvidenceAnchor(content, 0);

  assert.match(result.snippet, /690994503/);
  assert.doesNotMatch(result.snippet, /用区间 DP/);
});

test("extractEvidenceAnchor jumps from stale partial code to the latest full later submission block", () => {
  const content = [
    " int:",
    "        dp = [amount + 1] * (amount + 1)",
    "        dp[0] = 0",
    "        for i in range(amount + 1):",
    "            for coin in coins:",
    "                if i >= coin:",
    "                    dp[i] = min(dp[i - coin], dp[i])",
    "        return dp[amount] if dp[amount] != (amount + 1) else -1",
    "```",
    "",
    "#### 提交 689182118 · Runtime Error · 2026-01-05 15:05:33 CST · python3",
    "",
    "```python",
    "class Solution:",
    "    def coinChange(self, coins: List[int], amount: int) -> int:",
    "        dp = [amount + 1] * (n + 1)",
    "```",
    "",
    "#### 提交 689424976 · Wrong Answer · 2026-01-06 14:42:34 CST · python3",
    "",
    "```python",
    "class Solution:",
    "    def coinChange(self, coins: List[int], amount: int) -> int:",
    "        dp = [[amount + 1] * (amount + 1) for _ in range(n + 1)]",
    "        return -1 if dp[n][amount] == (amount + 1) else dp[n][amount]",
    "```",
    "",
    "#### 提交 689425407 · Wrong Answer · 2026-01-06 14:44:03 CST · python3",
    "",
    "```python",
    "class Solution:",
    "    def coinChange(self, coins: List[int], amount: int) -> int:",
    "        dp = [[amount + 1] * (amount + 1) for _ in range(n + 1)]",
  ].join("\n");

  const result = extractEvidenceAnchor(content, 48800, {
    startedMidLine: true,
    currentHeading: "提交 688034687 · Wrong Answer · 2025-12-30 10:47:29 CST · python3",
  });

  assert.match(result.snippet, /689424976/);
  assert.doesNotMatch(result.snippet, /689182118/);
  assert.ok(result.charOffset > 48800);
});

test("extractEvidenceAnchor keeps the current submission when read starts inside its code body", () => {
  const content = [
    "        n = len(s)",
    "        dp = [[0] * n for _ in range(n)]",
    "        for i in range(n):",
    "            dp[i][i] = 1",
    "        return dp[n][n - 1]",
    "```",
    "",
    "#### 提交 690994572 · Runtime Error · 2026-01-13 13:12:55 CST · python3",
    "",
    "```python",
    "class Solution:",
    "    def longestPalindromeSubseq(self, s: str) -> int:",
    "        n = len(s)",
  ].join("\n");

  const result = extractEvidenceAnchor(content, 1497, {
    currentHeading: "提交 690994503 · Runtime Error · 2026-01-13 13:12:31 CST · python3",
    currentHeadingDistance: 448,
  });

  assert.doesNotMatch(result.snippet, /690994572/);
  assert.match(result.snippet, /dp = \[\[0\] \* n for _ in range\(n\)\]/);
});
