# 近一年 LeetCode 提交汇总

- 统计时间范围：2025-01-13 22:17:01 CST 起至 2026-01-13 23:19:51 CST 统计
- 总提交数：1058

## 最长回文子序列 (`longest-palindromic-subsequence`)

- 题目链接：https://leetcode.cn/problems/longest-palindromic-subsequence/
- 难度：Medium
- 标签：字符串, 动态规划
- 总提交次数：5
- 最近提交时间：2026-01-13 13:13:04 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-13 13:04:20 CST

```markdown
用区间 DP。
dp[i][j] 表示 s[i..j] 的最长回文子序列长度。
若 s[i]==s[j]，两端可以配对，dp[i][j]=dp[i+1][j-1]+2；否则只能舍弃一端，dp[i][j]=max(dp[i+1][j], dp[i][j-1])。

初始化 dp[i][i]=1。

按 i 从后往前、j 从前往后填表保证依赖已计算。
时间 O(n^2)，空间 O(n^2)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688523909 | 2026-01-02 10:45:06 CST | python3 | Accepted | 759 ms | 32.7 MB |
| 690711245 | 2026-01-12 10:58:35 CST | python3 | Accepted | 583 ms | 34.9 MB |
| 690994503 | 2026-01-13 13:12:31 CST | python3 | Runtime Error | N/A | N/A |
| 690994572 | 2026-01-13 13:12:55 CST | python3 | Runtime Error | N/A | N/A |
| 690994600 | 2026-01-13 13:13:04 CST | python3 | Accepted | 539 ms | 35.2 MB |

### 未通过提交代码
#### 提交 690994503 · Runtime Error · 2026-01-13 13:12:31 CST · python3

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[n][n - 1]
```

#### 提交 690994572 · Runtime Error · 2026-01-13 13:12:55 CST · python3

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[n][n - 1]
```


## 乘积最大子数组 (`maximum-product-subarray`)

- 题目链接：https://leetcode.cn/problems/maximum-product-subarray/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：3
- 最近提交时间：2026-01-13 12:23:53 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-12 14:30:13 CST

```markdown
每一步更新这两个值时，新的最大值可能来自「上一步的最大值 * 当前数」，「上一步的最小值 * 当前数」，或者「当前数本身」这三者中的最大者。最小值同理。

* 状态：以 i 结尾的最优值（连续子数组 DP 的标准姿势）
* 选择：当前位置 num 要么“自成一段”，要么“接到之前那段后面”
* 为什么要 min_end：乘积遇到负数会翻转，之前很小的负数乘以负数可能变成很大的正数
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685090653 | 2025-12-16 12:16:06 CST | python3 | Accepted | 11 ms | 18.2 MB |
| 690765622 | 2026-01-12 14:37:22 CST | python3 | Accepted | 3 ms | 19.5 MB |
| 690987275 | 2026-01-13 12:23:53 CST | python3 | Accepted | 3 ms | 19.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 环形子数组的最大和 (`maximum-sum-circular-subarray`)

- 题目链接：https://leetcode.cn/problems/maximum-sum-circular-subarray/
- 难度：Medium
- 标签：队列, 数组, 分治, 动态规划, 单调队列
- 总提交次数：4
- 最近提交时间：2026-01-13 12:14:45 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-13 12:02:49 CST

```markdown
环形最大子数组分两类：不跨首尾就是 Kadane 求最大子数组和；跨首尾等价于把数组中间一段最小子数组删掉，所以是 total - minSubarray。最终答案取 max(maxKadane, total - minKadane)。
若全为负，minKadane 会等于 total，total-min 为 0 表示空子数组，不合法，此时直接返回 maxKadane。
时间 O(n)，空间 O(1)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 690775439 | 2026-01-12 15:08:36 CST | python3 | Wrong Answer | N/A | N/A |
| 690775961 | 2026-01-12 15:10:03 CST | python3 | Accepted | 67 ms | 23.9 MB |
| 690985018 | 2026-01-13 12:13:45 CST | python3 | Wrong Answer | N/A | N/A |
| 690985161 | 2026-01-13 12:14:45 CST | python3 | Accepted | 57 ms | 23.9 MB |

### 未通过提交代码
#### 提交 690775439 · Wrong Answer · 2026-01-12 15:08:36 CST · python3

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        total = nums[0]
        max_so_far = nums[0]
        max_sum = nums[0]
        min_so_far = nums[0]
        min_sum = nums[0]
        for num in nums:
            total += num
            max_so_far = max(num, num + max_so_far)
            max_sum = max(max_sum, max_so_far)
            min_so_far = min(num, num + min_so_far)
            min_sum = min(min_sum, min_so_far)
        if max_sum < 0:
            return max_sum
        return max(max_sum, total - min_sum)
```

#### 提交 690985018 · Wrong Answer · 2026-01-13 12:13:45 CST · python3

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        total = nums[0]
        max_so_far = nums[0]
        max_sum = nums[0]
        min_so_far = nums[0]
        min_sum = nums[0]
        for num in nums[1:]:
            total += num
            max_so_far = max(num, num + max_so_far)
            max_sum = max(max_sum, max_so_far)
            min_so_far = min(num, num + min_so_far)
            min_sum = min(min_sum, min_so_far)
        if max_sum < 0:
            return max_sum
        return max(max_sum, total - min_so_far)
```


## 多边形三角剖分的最低得分 (`minimum-score-triangulation-of-polygon`)

- 题目链接：https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：1
- 最近提交时间：2026-01-13 11:58:36 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-13 11:58:33 CST

```markdown
一段区间要被切分成若干部分，总代价等于“子区间代价 + 当前切分产生的代价”

状态（State）：dp[i][j] = 对子多边形顶点 i..j 的最小剖分得分
选择（Choice）：选择哪个 k 作为三角形 (i, k, j) 的第三个点（等价于“最后一刀”）
转移（Transition）：dp[i][j] = min(dp[i][k] + dp[k][j] + A[i]*A[k]*A[j])
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 690982337 | 2026-01-13 11:58:36 CST | python3 | Accepted | 27 ms | 19.3 MB |

### 未通过提交代码
(所有提交均已通过)

## 戳气球 (`burst-balloons`)

- 题目链接：https://leetcode.cn/problems/burst-balloons/
- 难度：Hard
- 标签：数组, 动态规划
- 总提交次数：1
- 最近提交时间：2026-01-13 10:41:32 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-13 10:46:41 CST

```markdown
两侧相邻会随戳破变化，难在动态相邻关系。
用“最后戳破”思路：在区间 (l,r) 内枚举最后戳的气球 k，则最后一次收益只与 l、k、r 有关，左右两段 (l,k)、(k,r) 互不影响。
设 dp[l][r] 表示开区间 (l,r) 内全部戳完的最大金币数，转移 dp[l][r]=max(dp[l][k]+dp[k][r]+vals[l]*vals[k]*vals[r])。两端补 1，按区间长度递增填表。
时间复杂度 O(n^3)，空间复杂度O(n^2)。

预处理：vals = [1] + nums + [1]
状态：dp[l][r]：戳完开区间 (l, r) 内所有气球的最大金币
选择：最后一个戳破的气球位置 k（l < k < r）
转移：dp[l][r] = max(dp[l][k] + dp[k][r] + vals[l] * vals[k] * vals[r])
填表顺序：区间长度从小到大（确保子区间先算好）

开区间 (l, r) 的本质作用是：
把 l 和 r 固定为“本次子任务的左右**边界墙**”。
* 这两堵墙一定存在（在当前子问题里不被戳破）。
* 这样当你戳破区间内最后一个气球 k 时，它的邻居百分之百就是 l 和 r。
* 从而让计算变得确定、独立：score = val[l] * val[k] * val[r]。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 690958340 | 2026-01-13 10:41:32 CST | python3 | Accepted | 1744 ms | 21.7 MB |

### 未通过提交代码
(所有提交均已通过)

## 最长回文子串 (`longest-palindromic-substring`)

- 题目链接：https://leetcode.cn/problems/longest-palindromic-substring/
- 难度：Medium
- 标签：双指针, 字符串, 动态规划
- 总提交次数：13
- 最近提交时间：2026-01-12 14:01:17 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-30 15:33:27 CST

```markdown
中心扩展法，分奇数和偶数两种情况
循环结束时，left 和 right 都多走了一步，实际的回文子串范围应该是从 left + 1 到 right - 1，这时子串的长度是 right - left - 1

时间复杂度是 O(n²)，空间复杂度是 O(1)
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681284721 | 2025-11-28 13:59:07 CST | python3 | Runtime Error | N/A | N/A |
| 681284772 | 2025-11-28 13:59:20 CST | python3 | Accepted | 267 ms | 17.4 MB |
| 681285628 | 2025-11-28 14:03:53 CST | python3 | Wrong Answer | N/A | N/A |
| 681286360 | 2025-11-28 14:07:26 CST | python3 | Accepted | 263 ms | 17.5 MB |
| 681671141 | 2025-11-30 15:32:41 CST | python3 | Accepted | 291 ms | 17.4 MB |
| 681906993 | 2025-12-01 16:16:16 CST | python3 | Runtime Error | N/A | N/A |
| 681907036 | 2025-12-01 16:16:22 CST | python3 | Accepted | 291 ms | 17.7 MB |
| 685559037 | 2025-12-18 11:37:04 CST | python3 | Accepted | 211 ms | 17 MB |
| 685795105 | 2025-12-19 13:47:41 CST | python3 | Runtime Error | N/A | N/A |
| 685795126 | 2025-12-19 13:47:49 CST | python3 | Runtime Error | N/A | N/A |
| 685795165 | 2025-12-19 13:48:08 CST | python3 | Accepted | 207 ms | 17.2 MB |
| 690756338 | 2026-01-12 14:00:31 CST | python3 | Runtime Error | N/A | N/A |
| 690756534 | 2026-01-12 14:01:17 CST | python3 | Accepted | 207 ms | 19.3 MB |

### 未通过提交代码
#### 提交 681284721 · Runtime Error · 2025-11-28 13:59:07 CST · python3

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        start = end = 0
        # 中心扩展法，分奇数和偶数两种情况
        
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            # 循环结束时，left 和 right 都多走了一步，实际的回文子串范围应该是从 left + 1 到 right - 1，这时子串的长度是 right - left - 1
            return right - left - 1
        
        for i in range(len(s)):
            len1 = expand_around_center(i, i) # 偶数的情况
            len2 = expand_around_center(i, i+1) # 奇数的情况
            max_len = man(len1, len2)
            if max_len > (end - start + 1):
                start = i - (max_len - 1) // 2 # 因为 i 靠左，同时覆盖奇数和偶数的情况
                end = i + max_len // 2
        
        return s[start:end+1]
```

#### 提交 681285628 · Wrong Answer · 2025-11-28 14:03:53 CST · python3

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        start = end = 0
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        for i in range(len(s)):
            len_1 = expand_around_center(i, i)
            len_2 = expand_around_center(i, i+1)
            max_len = max(len_1, len_2)
            if max_len > (start - end + 1):
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        
        return s[start:end+1]
```

#### 提交 681906993 · Runtime Error · 2025-12-01 16:16:16 CST · python3

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] = s[right]:
                left -= 1
                right += 1
            return right - left - 1
        start = end = 0
        for i in range(len(s)):
            len_1 = expand_around_center(i, i)  # 奇数的场景
            len_2 = expand_around_center(i, i + 1)  # 偶数的场景
            max_len = max(len_1, len_2)
            if max_len > end - start + 1:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        return s[start : end + 1]
```

#### 提交 685795105 · Runtime Error · 2025-12-19 13:47:41 CST · python3

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        def expand_around_center(left, right):
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        start = end = 0
        for i in range(s):
            len_1 = expand_around_center(i, i)
            len_2 = expand_around_center(i, i + 1)
            max_len = max(len_1, len_2)
            if max_len > right - left + 1:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        return s[start : end + 1]
```

#### 提交 685795126 · Runtime Error · 2025-12-19 13:47:49 CST · python3

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        def expand_around_center(left, right):
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        start = end = 0
        for i in range(n):
            len_1 = expand_around_center(i, i)
            len_2 = expand_around_center(i, i + 1)
            max_len = max(len_1, len_2)
            if max_len > right - left + 1:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        return s[start : end + 1]
```

#### 提交 690756338 · Runtime Error · 2026-01-12 14:00:31 CST · python3

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        def expand_arount_center(left, right):
            while left >= 0 and right < n and s[left] == s[right]:
                left += 1
                right -= 1
            return right - left - 1
        start = end = 0
        for i in range(n):
            len_1 = expand_arount_center(i, i)
            len_2 = expand_arount_center(i, i + 1)
            max_len = max(len_1, len_2)
            if max_len > end - start + 1:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        return s[start : end + 1]
```


## 交错字符串 (`interleaving-string`)

- 题目链接：https://leetcode.cn/problems/interleaving-string/
- 难度：Medium
- 标签：字符串, 动态规划
- 总提交次数：2
- 最近提交时间：2026-01-12 13:37:53 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 690746726 | 2026-01-12 13:06:00 CST | python3 | Wrong Answer | N/A | N/A |
| 690751806 | 2026-01-12 13:37:53 CST | python3 | Accepted | 53 ms | 19.4 MB |

### 未通过提交代码
#### 提交 690746726 · Wrong Answer · 2026-01-12 13:06:00 CST · python3

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)
        if m + n != len(s3):
            return False
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                k = i + j - 1
                if i > 0 and s3[k] == s1[i - 1] and dp[i - 1][j]:
                    dp[i][j] = True
                if j > 0 and s3[k] == s2[j - 1] and dp[i][j - 1]:
                    dp[i][j] = True
        return dp[m][n]
```


## 让字符串成为回文串的最少插入次数 (`minimum-insertion-steps-to-make-a-string-palindrome`)

- 题目链接：https://leetcode.cn/problems/minimum-insertion-steps-to-make-a-string-palindrome/
- 难度：Hard
- 标签：字符串, 动态规划
- 总提交次数：2
- 最近提交时间：2026-01-12 11:04:54 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-02 11:07:02 CST

```markdown
和 516 题打通：
最少插入次数 = n - 最长回文子序列长度（LPS） 因为你最多能“保留”的回文骨架就是 LPS，其余字符都得通过插入来补齐对称。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688527691 | 2026-01-02 11:11:57 CST | python3 | Accepted | 327 ms | 18.5 MB |
| 690713229 | 2026-01-12 11:04:54 CST | python3 | Accepted | 187 ms | 20.9 MB |

### 未通过提交代码
(所有提交均已通过)

## 两个字符串的最小ASCII删除和 (`minimum-ascii-delete-sum-for-two-strings`)

- 题目链接：https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings/
- 难度：Medium
- 标签：字符串, 动态规划
- 总提交次数：5
- 最近提交时间：2026-01-12 10:41:32 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-02 09:53:12 CST

```markdown
dp[i][j] 表示让 s1 前 i 个和 s2 前 j 个相同的最小删除 ASCII 和。初始化 dp[i][0] 是删光 s1[:i] 的代价，dp[0][j] 是删光 s2[:j] 的代价。转移：若末尾字符相等，dp[i][j]=dp[i-1][j-1]；否则要么删 s1 的末尾要么删 s2 的末尾，取更小：min(dp[i-1][j]+ord(s1[i-1]), dp[i][j-1]+ord(s2[j-1]))。答案 dp[m][n]，复杂度 O(mn)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688518909 | 2026-01-02 10:02:54 CST | python3 | Runtime Error | N/A | N/A |
| 688518924 | 2026-01-02 10:03:06 CST | python3 | Wrong Answer | N/A | N/A |
| 688519000 | 2026-01-02 10:03:57 CST | python3 | Accepted | 439 ms | 21.1 MB |
| 690705580 | 2026-01-12 10:39:03 CST | python3 | Wrong Answer | N/A | N/A |
| 690706247 | 2026-01-12 10:41:32 CST | python3 | Accepted | 267 ms | 23.3 MB |

### 未通过提交代码
#### 提交 688518909 · Runtime Error · 2026-01-02 10:02:54 CST · python3

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
        for j in range(n + 1):
            dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])
        
        for i in range(m + 1):
            for j in in range(n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    del_s1 = dp[i - 1][j] + ord(s1[i - 1])
                    del_s2 = dp[i][j - 1] + ord(s2[j - 1])
                    dp[i][j] = min(del_s1, del_s2)
        
        return dp[m][n]
```

#### 提交 688518924 · Wrong Answer · 2026-01-02 10:03:06 CST · python3

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
        for j in range(n + 1):
            dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])
        
        for i in range(m + 1):
            for j in range(n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    del_s1 = dp[i - 1][j] + ord(s1[i - 1])
                    del_s2 = dp[i][j - 1] + ord(s2[j - 1])
                    dp[i][j] = min(del_s1, del_s2)
        
        return dp[m][n]
```

#### 提交 690705580 · Wrong Answer · 2026-01-12 10:39:03 CST · python3

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = ord(s1[i - 1])
        for j in range(1, n + 1):
            dp[0][j] = ord(s2[j - 1])
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    del_s1 = dp[i - 1][j] + ord(s1[i - 1])
                    del_s2 = dp[i][j - 1] + ord(s2[j - 1])
                    dp[i][j] = min(del_s1, del_s2)
        return dp[m][n]
```


## 两个字符串的删除操作 (`delete-operation-for-two-strings`)

- 题目链接：https://leetcode.cn/problems/delete-operation-for-two-strings/
- 难度：Medium
- 标签：字符串, 动态规划
- 总提交次数：9
- 最近提交时间：2026-01-12 09:55:39 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685328370 | 2025-12-17 11:28:30 CST | python3 | Wrong Answer | N/A | N/A |
| 685328628 | 2025-12-17 11:29:23 CST | python3 | Wrong Answer | N/A | N/A |
| 685328807 | 2025-12-17 11:30:08 CST | python3 | Wrong Answer | N/A | N/A |
| 685329263 | 2025-12-17 11:31:49 CST | python3 | Accepted | 137 ms | 19.4 MB |
| 688515358 | 2026-01-02 09:10:11 CST | python3 | Accepted | 139 ms | 19 MB |
| 690694803 | 2026-01-12 09:53:50 CST | python3 | Wrong Answer | N/A | N/A |
| 690694867 | 2026-01-12 09:54:10 CST | python3 | Runtime Error | N/A | N/A |
| 690694897 | 2026-01-12 09:54:20 CST | python3 | Wrong Answer | N/A | N/A |
| 690695170 | 2026-01-12 09:55:39 CST | python3 | Accepted | 79 ms | 21 MB |

### 未通过提交代码
#### 提交 685328370 · Wrong Answer · 2025-12-17 11:28:30 CST · python3

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] == dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        longest_common_len = dp[m][n]
        return m + n - 2 * longest_common_len
```

#### 提交 685328628 · Wrong Answer · 2025-12-17 11:29:23 CST · python3

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] == dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        longest_common_len = dp[m][n]
        return m + n - 2 * longest_common_len
```

#### 提交 685328807 · Wrong Answer · 2025-12-17 11:30:08 CST · python3

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] == dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        longest_common_len = dp[m][n]
        return m + n - 2 * longest_common_len
```

#### 提交 690694803 · Wrong Answer · 2026-01-12 09:53:50 CST · python3

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if word1[i - 1] == word2[i - 2]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        max_common_len = dp[m][n]
        return m + n - 2 * max_common_len
```

#### 提交 690694867 · Runtime Error · 2026-01-12 09:54:10 CST · python3

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[i - 2]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        max_common_len = dp[m][n]
        return m + n - 2 * max_common_len
```

#### 提交 690694897 · Wrong Answer · 2026-01-12 09:54:20 CST · python3

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        max_common_len = dp[m][n]
        return m + n - 2 * max_common_len
```


## 最长公共子序列 (`longest-common-subsequence`)

- 题目链接：https://leetcode.cn/problems/longest-common-subsequence/
- 难度：Medium
- 标签：字符串, 动态规划
- 总提交次数：8
- 最近提交时间：2026-01-12 09:47:25 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-17 11:15:59 CST

```markdown
这类最长公共子序列问题非常适合用动态规划，原因有两个关键点。第一，它有明显的最优子结构：前 i、j 个字符的最长公共子序列，可以完全由更短前缀的最优结果推出来，比如相等时是 dp[i-1][j-1]+1，不相等时是 max(dp[i-1][j], dp[i][j-1])。第二，它有大量重叠子问题，不同的匹配路径会反复遇到相同的前缀组合 (i, j)。如果用递归暴力会是指数级，而用 DP 把每个状态的结果保存起来，每个子问题只算一次，整体复杂度就降到 O(m*n)，既正确又高效。

动态规划题的适用场景：最优子结构、重叠子问题

dp[i][j] 表示 text1 的前 i 个字符与 text2 的前 j 个字符的最长公共子序列
多一行一列是为了解决空字符的问题，避免越界
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682062863 | 2025-12-02 09:53:23 CST | python3 | Wrong Answer | N/A | N/A |
| 682063112 | 2025-12-02 09:54:29 CST | python3 | Accepted | 499 ms | 42 MB |
| 685325987 | 2025-12-17 11:19:58 CST | python3 | Accepted | 503 ms | 41.9 MB |
| 685533196 | 2025-12-18 10:00:23 CST | python3 | Accepted | 526 ms | 41.6 MB |
| 685810059 | 2025-12-19 14:56:32 CST | python3 | Wrong Answer | N/A | N/A |
| 685810610 | 2025-12-19 14:58:47 CST | python3 | Accepted | 507 ms | 41.4 MB |
| 688515050 | 2026-01-02 09:02:20 CST | python3 | Accepted | 491 ms | 41.4 MB |
| 690693605 | 2026-01-12 09:47:25 CST | python3 | Accepted | 351 ms | 43.7 MB |

### 未通过提交代码
#### 提交 682062863 · Wrong Answer · 2025-12-02 09:53:23 CST · python3

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
```

#### 提交 685810059 · Wrong Answer · 2025-12-19 14:56:32 CST · python3

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[m - 1] == text2[n - 1]:
                    dp[i][j] = dp[i - 1 ][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
```


## 最大子数组和 (`maximum-subarray`)

- 题目链接：https://leetcode.cn/problems/maximum-subarray/
- 难度：Medium
- 标签：数组, 分治, 动态规划
- 总提交次数：7
- 最近提交时间：2026-01-11 21:34:21 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-02 08:18:05 CST

```markdown
典型的「一维动态规划」+「贪心」结合题

核心思路是维护一个current_sum。遍历数组时，如果current_sum变成了负数，说明前面的前缀对后续没有贡献（贪心），我就直接抛弃它，从当前数字重新开始累加；否则就继续累加。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682359771 | 2025-12-03 12:50:35 CST | python3 | Runtime Error | N/A | N/A |
| 682359796 | 2025-12-03 12:50:52 CST | python3 | Accepted | 56 ms | 31.9 MB |
| 685091160 | 2025-12-16 12:20:07 CST | python3 | Accepted | 59 ms | 31.9 MB |
| 688513688 | 2026-01-02 08:13:58 CST | python3 | Accepted | 79 ms | 29.3 MB |
| 688513749 | 2026-01-02 08:16:38 CST | python3 | Accepted | 51 ms | 29.3 MB |
| 690633288 | 2026-01-11 21:33:57 CST | python3 | Wrong Answer | N/A | N/A |
| 690633402 | 2026-01-11 21:34:21 CST | python3 | Accepted | 32 ms | 31 MB |

### 未通过提交代码
#### 提交 682359771 · Runtime Error · 2025-12-03 12:50:35 CST · python3

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        current_sum = nums[0]
        for num in num[1:]:
            if current_sum < 0:
                current_sum = num
            else:
                current_sum += num
            max_sum = max(max_sum, current_sum)
        return max_sum
```

#### 提交 690633288 · Wrong Answer · 2026-01-11 21:33:57 CST · python3

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        current_sum = nums[0]
        for num in nums[1:]:
            if current_sum < 0:
                current_sum = num
            else:
                current_sum += num
                max_sum = max(max_sum, current_sum)
        return max_sum
```


## 编辑距离 (`edit-distance`)

- 题目链接：https://leetcode.cn/problems/edit-distance/
- 难度：Medium
- 标签：字符串, 动态规划
- 总提交次数：6
- 最近提交时间：2026-01-10 17:29:42 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-31 14:59:48 CST

```markdown
“这道题是经典的二维动态规划问题。
我定义 dp[i][j] 为 word1 前 i 个字符转换到 word2 前 j 个字符的最小步数。
首先初始化边界，也就是其中一个字符串为空时，步数就是另一个字符串的长度。
然后进行状态转移：
如果当前两个字符相等，则不需要操作，dp[i][j] 直接等于左上角的 dp[i-1][j-1]；
如果不相等，则考虑增、删、改三种操作，取其代价最小者加 1，也就是取左、上、左上三个位置的最小值加 1。
最终 dp[m][n] 就是答案。时间复杂度和空间复杂度都是 O(m \times n)。”

状态里的删除、插入、替换，本质上是在看我们如何缩小问题的规模：
* 如果是 dp[i-1][j]，说明我们忽略了 word1 的第 i 个字符，这就等同于删除。
* 如果是 dp[i][j-1]，说明我们先搞定了 word2 的第 j 个字符（假设 word1 加上了这个字符），这就等同于插入。
* 如果是 dp[i-1][j-1]，说明我们同时处理了两个字符串的末尾字符，这就等同于替换（如果不相等的话）。”

网格 DP 的一个固定套路：只要状态是“前 i 个 / 前 j 个”，就要多开一行一列，专门表示空串。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 679406845 | 2025-11-20 11:12:01 CST | python3 | Runtime Error | N/A | N/A |
| 679406918 | 2025-11-20 11:12:15 CST | python3 | Accepted | 71 ms | 20.9 MB |
| 682002018 | 2025-12-01 22:00:19 CST | python3 | Runtime Error | N/A | N/A |
| 682002098 | 2025-12-01 22:00:35 CST | python3 | Accepted | 72 ms | 20.9 MB |
| 688277717 | 2025-12-31 13:52:50 CST | python3 | Accepted | 75 ms | 20.3 MB |
| 690376130 | 2026-01-10 17:29:42 CST | python3 | Accepted | 47 ms | 22.3 MB |

### 未通过提交代码
#### 提交 679406845 · Runtime Error · 2025-11-20 11:12:01 CST · python3

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        if not m * n:
            return m + n
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # 初始化边界条件
        # 第一列，word2是空串，操作删除word1的每一个字符
        for i in range(m + 1):
            dp[i][0] = i
        # 第一行，word1是空串，只能操作插入
        for j in range(n + 1):
            dp[0][j] = j
        
        # 开始填表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # 字符不同，取 min(左上, 上, 左) + 1
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[m][n]
```

#### 提交 682002018 · Runtime Error · 2025-12-01 22:00:19 CST · python3

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        if not m * n:
            return m + n
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # 第一列
        for i in range(m + 1):
            dp[i][0] = i
        # 第一行
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] = word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        
        return dp[m][n]
```


## 和为目标值的最长子序列的长度 (`length-of-the-longest-subsequence-that-sums-to-target`)

- 题目链接：https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：4
- 最近提交时间：2026-01-09 14:06:38 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-07 16:29:52 CST

```markdown
0-1 背包最值问题。
dp[i][j] 表示用前 i 个数凑出和 j 的最大长度，不可达记为负无穷。
转移是选或不选第 i 个数：dp[i][j]=max(dp[i-1][j], dp[i-1][j-num]+1)。初始化 dp[0][0]=0。最后看 dp[n][target]，不可达返回 -1。复杂度 O(n*target)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 689713537 | 2026-01-07 16:36:57 CST | python3 | Runtime Error | N/A | N/A |
| 689714021 | 2026-01-07 16:38:24 CST | python3 | Runtime Error | N/A | N/A |
| 689715076 | 2026-01-07 16:41:29 CST | python3 | Accepted | 2461 ms | 28.2 MB |
| 690138972 | 2026-01-09 14:06:38 CST | python3 | Accepted | 2555 ms | 28.4 MB |

### 未通过提交代码
#### 提交 689713537 · Runtime Error · 2026-01-07 16:36:57 CST · python3

```python
class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [[float('-inf') * (target + 1) for _ in range(n + 1)]]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for x in range(target + 1):
                dp[i][x] = dp[i - 1][x]
                if x >= num:
                    dp[i][x] = max(dp[i][x], dp[i - 1][x - num] + 1)
        return -1 if dp[n][target] == float('-inf') else dp[n][target]
```

#### 提交 689714021 · Runtime Error · 2026-01-07 16:38:24 CST · python3

```python
class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [[float('-inf') * (target + 1) for _ in range(n + 1)]]
        dp[0][0] = 0
        for i in range(1, n + 1):
            num = nums[i - 1]
            for x in range(target + 1):
                dp[i][x] = dp[i - 1][x]
                if x >= num:
                    dp[i][x] = max(dp[i][x], dp[i - 1][x - num] + 1)
        return -1 if dp[n][target] == float('-inf') else dp[n][target]
```


## 一和零 (`ones-and-zeroes`)

- 题目链接：https://leetcode.cn/problems/ones-and-zeroes/
- 难度：Medium
- 标签：数组, 字符串, 动态规划
- 总提交次数：14
- 最近提交时间：2026-01-09 14:00:24 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-04 09:33:10 CST

```markdown
条件反射（看到就想到）
* 每个字符串 选 / 不选（最多选一次） ⇒ 0-1 背包
* 约束有两个：0 的个数 ≤ m，1 的个数 ≤ n ⇒ 二维背包容量
* 目标：选出最多字符串数量 ⇒ dp 取 max


状态：dp[i][j]：在“最多 i 个 0、最多 j 个 1”的限制下，能选到的字符串最大数量
选择：
* 不选：dp[i][j] 不变
* 选：前提是 i >= zero_cnt 且 j >= one_cnt，则来自 dp[i-zero][j-one] + 1
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688851086 | 2026-01-04 08:55:49 CST | python3 | Wrong Answer | N/A | N/A |
| 688851147 | 2026-01-04 08:56:25 CST | python3 | Accepted | 2427 ms | 71.7 MB |
| 688856919 | 2026-01-04 09:46:02 CST | python3 | Accepted | 1587 ms | 17.5 MB |
| 688955122 | 2026-01-04 16:07:28 CST | python3 | Runtime Error | N/A | N/A |
| 688955397 | 2026-01-04 16:08:13 CST | python3 | Runtime Error | N/A | N/A |
| 688955957 | 2026-01-04 16:09:45 CST | python3 | Wrong Answer | N/A | N/A |
| 688956262 | 2026-01-04 16:10:46 CST | python3 | Wrong Answer | N/A | N/A |
| 688956719 | 2026-01-04 16:12:16 CST | python3 | Accepted | 2379 ms | 71.5 MB |
| 689166360 | 2026-01-05 14:14:04 CST | python3 | Runtime Error | N/A | N/A |
| 689166516 | 2026-01-05 14:14:48 CST | python3 | Wrong Answer | N/A | N/A |
| 689166957 | 2026-01-05 14:16:25 CST | python3 | Wrong Answer | N/A | N/A |
| 689169043 | 2026-01-05 14:23:53 CST | python3 | Accepted | 2700 ms | 70.5 MB |
| 690137308 | 2026-01-09 13:58:24 CST | python3 | Wrong Answer | N/A | N/A |
| 690137706 | 2026-01-09 14:00:24 CST | python3 | Accepted | 2079 ms | 73.4 MB |

### 未通过提交代码
#### 提交 688851086 · Wrong Answer · 2026-01-04 08:55:49 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        # 只使用前 k 个字符串时，最多可以凑出的字符串的数量
        dp =[[[0] * (n + 1) for _ in range(m + 1)] for _ in range(k + 1)]
        for t in range(1, k + 1):
            s = strs[t - 1]
            zero_count = s.count('0')
            one_count = s.count('1')
            for i in range(m + 1):
                for j in range(n + 1):
                    best = dp[t - 1][i][j]  
                    if i >= zero_count and j >= one_count:
                        best = max(best, dp[t - 1][i - zero_count][j - one_count])
                    dp[t][i][j] = best
        return dp[k][m][n]
```

#### 提交 688955122 · Runtime Error · 2026-01-04 16:07:28 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        dp = [[[0] * (k + 1) for _ in range (n + 1)] for _ in range(m + 1)]
        dp[0][0][0] = 1
        for s in range(1, k + 1):
            curr = strs[s - 1]
            zero_cnt = curr.count('0')
            one_cnt = curr.count('1')
            for i in range(m + 1):
                for j in range(n + 1):
                    best = dp[s - 1][i][j]
                    if m >= zero_cnt and n >= one_cnt:
                        best = max(best, dp[s - 1][m - zero_cnt][n - one_cnt] + 1)
        return dp[k][m][n]
```

#### 提交 688955397 · Runtime Error · 2026-01-04 16:08:13 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        dp = [[[0] * (k + 1) for _ in range (n + 1)] for _ in range(m + 1)]
        dp[0][0][0] = 1
        for s in range(1, k + 1):
            curr = strs[s - 1]
            zero_cnt = curr.count('0')
            one_cnt = curr.count('1')
            for i in range(m + 1):
                for j in range(n + 1):
                    best = dp[s - 1][i][j]
                    if i >= zero_cnt and j >= one_cnt:
                        best = max(best, dp[s - 1][i - zero_cnt][j - one_cnt] + 1)
        return dp[k][m][n]
```

#### 提交 688955957 · Wrong Answer · 2026-01-04 16:09:45 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        dp = [[[0] * (n + 1) for _ in range (m + 1)] for _ in range(k + 1)]
        dp[0][0][0] = 1
        for s in range(1, k + 1):
            curr = strs[s - 1]
            zero_cnt = curr.count('0')
            one_cnt = curr.count('1')
            for i in range(m + 1):
                for j in range(n + 1):
                    best = dp[s - 1][i][j]
                    if i >= zero_cnt and j >= one_cnt:
                        best = max(best, dp[s - 1][i - zero_cnt][j - one_cnt] + 1)
        return dp[k][m][n]
```

#### 提交 688956262 · Wrong Answer · 2026-01-04 16:10:46 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        dp = [[[0] * (n + 1) for _ in range (m + 1)] for _ in range(k + 1)]
        # dp[0][0][0] = 1
        for s in range(1, k + 1):
            curr = strs[s - 1]
            zero_cnt = curr.count('0')
            one_cnt = curr.count('1')
            for i in range(m + 1):
                for j in range(n + 1):
                    best = dp[s - 1][i][j]
                    if i >= zero_cnt and j >= one_cnt:
                        best = max(best, dp[s - 1][i - zero_cnt][j - one_cnt] + 1)
        return dp[k][m][n]
```

#### 提交 689166360 · Runtime Error · 2026-01-05 14:14:04 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(k + 1)]
        for t in range(k + 1):
            curr = strs[t - 1]
            zero_cnt = curr.count('0')
            one_cnt = curr.count('1')
            for i in range(i + 1):
                for j in range(j + 1):
                    best = dp[t - 1][i][j]
                    if i >= zero_cnt and j >= zero_cnt:
                        best = max(best, dp[t - 1][i - zero_cnt][j - one_cnt] + 1)
                    dp[t][i][j] = best
        return dp[k][m][n]
```

#### 提交 689166516 · Wrong Answer · 2026-01-05 14:14:48 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(k + 1)]
        for t in range(k + 1):
            curr = strs[t - 1]
            zero_cnt = curr.count('0')
            one_cnt = curr.count('1')
            for i in range(m + 1):
                for j in range(n + 1):
                    best = dp[t - 1][i][j]
                    if i >= zero_cnt and j >= zero_cnt:
                        best = max(best, dp[t - 1][i - zero_cnt][j - one_cnt] + 1)
                    dp[t][i][j] = best
        return dp[k][m][n]
```

#### 提交 689166957 · Wrong Answer · 2026-01-05 14:16:25 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(k + 1)]
        for t in range(1, k + 1):
            curr = strs[t - 1]
            zero_cnt = curr.count('0')
            one_cnt = curr.count('1')
            for i in range(m + 1):
                for j in range(n + 1):
                    best = dp[t - 1][i][j]
                    if i >= zero_cnt and j >= zero_cnt:
                        best = max(best, dp[t - 1][i - zero_cnt][j - one_cnt] + 1)
                    dp[t][i][j] = best
        return dp[k][m][n]
```

#### 提交 690137308 · Wrong Answer · 2026-01-09 13:58:24 CST · python3

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        k = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(k + 1)]
        for t in range(1, k + 1):
            curr = strs[t - 1]
            zero_cnt = curr.count('0')
            one_cnt = curr.count('1')
            for i in range(m + 1):
                for j in range(n + 1):
                    dp[t][i][j] = dp[t - 1][i][j]
                    if i >= zero_cnt and j >= one_cnt:
                        dp[t][i][j] = max(dp[t][i][j], dp[t - 1][i - zero_cnt][j - one_cnt])
        return dp[k][m][n]
```


## 最低票价 (`minimum-cost-for-tickets`)

- 题目链接：https://leetcode.cn/problems/minimum-cost-for-tickets/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：9
- 最近提交时间：2026-01-09 13:46:19 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-06 08:31:15 CST

```markdown
状态：dp[i] = 覆盖到第 i 天（包含 i）的最低成本
选择（只在出行日需要做选择）：今天买哪种票
* 买 1 天游：从 dp[i-1] 转移
* 买 7 天游：从 dp[i-7] 转移
* 买 30 天游：从 dp[i-30] 转移
关键点：非出行日不需要买票，直接继承昨天最优

时间：O(last_day)，last_day ≤ 365（题目典型约束下很稳）
空间：O(last_day)
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 689354608 | 2026-01-06 09:37:47 CST | python3 | Runtime Error | N/A | N/A |
| 689354712 | 2026-01-06 09:38:18 CST | python3 | Runtime Error | N/A | N/A |
| 689354777 | 2026-01-06 09:38:43 CST | python3 | Accepted | 5 ms | 19.3 MB |
| 689695397 | 2026-01-07 15:55:54 CST | python3 | Runtime Error | N/A | N/A |
| 689695454 | 2026-01-07 15:56:08 CST | python3 | Wrong Answer | N/A | N/A |
| 689695675 | 2026-01-07 15:56:47 CST | python3 | Wrong Answer | N/A | N/A |
| 689696298 | 2026-01-07 15:58:40 CST | python3 | Accepted | 0 ms | 19.2 MB |
| 690134990 | 2026-01-09 13:46:04 CST | python3 | Runtime Error | N/A | N/A |
| 690135038 | 2026-01-09 13:46:19 CST | python3 | Accepted | 0 ms | 18.9 MB |

### 未通过提交代码
#### 提交 689354608 · Runtime Error · 2026-01-06 09:37:47 CST · python3

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        travel_days = set(days)
        last_day = travel_days[-1]
        dp = [0] * (last_day + 1)
        for day in range(1, last_day + 1):
            if day not in travel_days:
                dp[day] = dp[day - 1]
                continue
            cost_1 = dp[day - 1] + costs[0]
            cost_7 = dp[max(0, day - 7)] + cost[1]
            cost_30 = dp[max(0, day - 30)] + cost[2]
            dp[day] = min(cost_1, cost_7, cost_30)
        return dp[last_day]
```

#### 提交 689354712 · Runtime Error · 2026-01-06 09:38:18 CST · python3

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        travel_days = set(days)
        last_day = days[-1]
        dp = [0] * (last_day + 1)
        for day in range(1, last_day + 1):
            if day not in travel_days:
                dp[day] = dp[day - 1]
                continue
            cost_1 = dp[day - 1] + costs[0]
            cost_7 = dp[max(0, day - 7)] + cost[1]
            cost_30 = dp[max(0, day - 30)] + cost[2]
            dp[day] = min(cost_1, cost_7, cost_30)
        return dp[last_day]
```

#### 提交 689695397 · Runtime Error · 2026-01-07 15:55:54 CST · python3

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        travel_days = set(days)
        last_day = travel_days[-1]
        dp = [0] * (last_day + 1)
        for day in range(1, last_day + 1):
            if day not in travel_days:
                dp[day] = dp[day - 1]
                continue
            cost_1 = dp[day - 1] + 2
            cost_7 = dp[max(0, day - 7)] + 7
            cost_15 = dp[max(0, day - 15)] + 15
            dp[day] = min(cost_1, cost_7, cost_15)
        return dp[last_day]
```

#### 提交 689695454 · Wrong Answer · 2026-01-07 15:56:08 CST · python3

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        travel_days = set(days)
        last_day = days[-1]
        dp = [0] * (last_day + 1)
        for day in range(1, last_day + 1):
            if day not in travel_days:
                dp[day] = dp[day - 1]
                continue
            cost_1 = dp[day - 1] + 2
            cost_7 = dp[max(0, day - 7)] + 7
            cost_15 = dp[max(0, day - 15)] + 15
            dp[day] = min(cost_1, cost_7, cost_15)
        return dp[last_day]
```

#### 提交 689695675 · Wrong Answer · 2026-01-07 15:56:47 CST · python3

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        travel_days = set(days)
        last_day = days[-1]
        dp = [0] * (last_day + 1)
        for day in range(1, last_day + 1):
            if day not in travel_days:
                dp[day] = dp[day - 1]
                continue
            cost_1 = dp[day - 1] + 2
            cost_7 = dp[max(0, day - 7)] + 7
            cost_15 = dp[max(0, day - 30)] + 15
            dp[day] = min(cost_1, cost_7, cost_15)
        return dp[last_day]
```

#### 提交 690134990 · Runtime Error · 2026-01-09 13:46:04 CST · python3

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        travel_days = set(days)
        last_day = days[-1]
        dp = [0] * (last_day + 1)
        for day in range(1, last_day + 1):
            if day not in travel_days:
                dp[day]= dp[day - 1]
                continue
            cost_1 = dp[day - 1] + cost[0]
            cost_7 = dp[(max(0, day - 7))] + cost[1]
            cost_30 = dp[(max(0, day - 30))] + cost[2]
            dp[day] = min(cost_1, cost_7, cost_30)
        return dp[last_day]
```


## 零钱兑换 (`coin-change`)

- 题目链接：https://leetcode.cn/problems/coin-change/
- 难度：Medium
- 标签：广度优先搜索, 数组, 动态规划
- 总提交次数：23
- 最近提交时间：2026-01-09 12:42:53 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-18 10:14:17 CST

```markdown
一句话结论
* 这题天然满足“最优子结构 + 子问题重叠 + 无后效性”，贪心不可靠，搜索会指数爆炸；DP 能把问题分解并缓存子结果，把指数级压到 O(金额 × 硬币种类)。

核心三点（本质原因）
* 最优子结构：凑成金额 i 的最优解，等价于“选最后一枚硬币 coin，再加上凑成 i-coin 的最优解”。否则可以替换出更优方案，矛盾。这直接给出转移：dp[i] = min(dp[i-coin] + 1)。
* 子问题重叠：不同金额的计算会反复用到相同的子金额（如 i-1、i-2…），不用 DP 就会重复计算很多次，复杂度飙升。
* 无后效性：dp[i] 只和数值 i 有关，不依赖具体选取顺序，且代价是可加的（每加一枚硬币都 +1）。所以一维 DP 就能正确收敛。

面试时的经验法则
* 当问题是最值优化，能“去掉最后一步/取第一步”得到同型更小子问题，且状态只依赖规模指标（金额、长度、容量），就优先考虑 DP。
* 若还能把状态按某个单调维度排序（本题按金额递增），那基本就是 DP 的舒适区。

30 秒口述要点
* 这题适合 DP，因为有明确的最优子结构：最后一枚硬币是 coin 时，答案等于 dp[i-coin]+1。
* 存在大量子问题重叠，直接搜索会指数爆炸，DP 用数组缓存把复杂度降到 O(N×A)。
* 状态无后效：dp[i] 只依赖金额 i，而非硬币顺序；按金额递增算，相当于在一个无环图上做最短路。
* 贪心不可靠，DP 是稳妥且易于实现的正确做法。

**注意不要丢掉 base case 的情况：dp[0] = 0**
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 676814884 | 2025-11-09 10:48:35 CST | python3 | Accepted | 755 ms | 18.1 MB |
| 677068809 | 2025-11-10 14:10:57 CST | python3 | Accepted | 771 ms | 17.9 MB |
| 685535748 | 2025-12-18 10:12:29 CST | python3 | Wrong Answer | N/A | N/A |
| 685535973 | 2025-12-18 10:13:26 CST | python3 | Accepted | 735 ms | 17.3 MB |
| 685805869 | 2025-12-19 14:39:34 CST | python3 | Wrong Answer | N/A | N/A |
| 685806051 | 2025-12-19 14:40:15 CST | python3 | Accepted | 759 ms | 17.7 MB |
| 688034687 | 2025-12-30 10:47:29 CST | python3 | Wrong Answer | N/A | N/A |
| 688034761 | 2025-12-30 10:47:48 CST | python3 | Accepted | 803 ms | 17.6 MB |
| 688555794 | 2026-01-02 14:05:23 CST | python3 | Accepted | 825 ms | 17.3 MB |
| 689182118 | 2026-01-05 15:05:33 CST | python3 | Runtime Error | N/A | N/A |
| 689182236 | 2026-01-05 15:05:52 CST | python3 | Runtime Error | N/A | N/A |
| 689182947 | 2026-01-05 15:08:00 CST | python3 | Accepted | 895 ms | 17.5 MB |
| 689388574 | 2026-01-06 11:34:34 CST | python3 | Wrong Answer | N/A | N/A |
| 689388748 | 2026-01-06 11:35:08 CST | python3 | Accepted | 403 ms | 19.3 MB |
| 689424976 | 2026-01-06 14:42:34 CST | python3 | Wrong Answer | N/A | N/A |
| 689425407 | 2026-01-06 14:44:03 CST | python3 | Wrong Answer | N/A | N/A |
| 689425621 | 2026-01-06 14:44:45 CST | python3 | Accepted | 695 ms | 21.1 MB |
| 689426380 | 2026-01-06 14:47:12 CST | python3 | Wrong Answer | N/A | N/A |
| 689428293 | 2026-01-06 14:53:06 CST | python3 | Accepted | 691 ms | 21 MB |
| 689704752 | 2026-01-07 16:12:12 CST | python3 | Wrong Answer | N/A | N/A |
| 689705689 | 2026-01-07 16:14:49 CST | python3 | Accepted | 711 ms | 21.2 MB |
| 690126090 | 2026-01-09 12:41:58 CST | python3 | Wrong Answer | N/A | N/A |
| 690126195 | 2026-01-09 12:42:53 CST | python3 | Accepted | 695 ms | 20.8 MB |

### 未通过提交代码
#### 提交 685535748 · Wrong Answer · 2025-12-18 10:12:29 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        for i in range(amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != (amount + 1) else -1
```

#### 提交 685805869 · Wrong Answer · 2025-12-19 14:39:34 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for i in range(amount + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = dp[i - coin] + 1
        return dp[amount] if dp[amount] != (amount + 1) else -1
```

#### 提交 688034687 · Wrong Answer · 2025-12-30 10:47:29 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for i in range(amount + 1):
            for coin in coins:
                if i >= coin:
                    dp[i] = min(dp[i - coin], dp[i])
        return dp[amount] if dp[amount] != (amount + 1) else -1
```

#### 提交 689182118 · Runtime Error · 2026-01-05 15:05:33 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [amount + 1] * (n + 1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[coin], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != (amount + 1) else -1
```

#### 提交 689182236 · Runtime Error · 2026-01-05 15:05:52 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [amount + 1] * (n + 1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != (amount + 1) else -1
```

#### 提交 689388574 · Wrong Answer · 2026-01-06 11:34:34 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != (amount + 1) else -1
```

#### 提交 689424976 · Wrong Answer · 2026-01-06 14:42:34 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [[amount + 1] * (amount + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            coin = coins[i - 1]
            for j in range(amount + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= coin:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - coin])
        return -1 if dp[n][amount] == (amount + 1) else dp[n][amount]
```

#### 提交 689425407 · Wrong Answer · 2026-01-06 14:44:03 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [[amount + 1] * (amount + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            coin = coins[i - 1]
            for j in range(amount + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= coin:
                    dp[i][j] = min(dp[i][j], dp[i][j - coin])
        return -1 if dp[n][amount] == (amount + 1) else dp[n][amount]
```

#### 提交 689426380 · Wrong Answer · 2026-01-06 14:47:12 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [[amount + 1] * (amount + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            coin = coins[i - 1]
            for j in range(coin, amount + 1):
                # dp[i][j] = dp[i - 1][j]
                # if j >= coin:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - coin] + 1)
        return -1 if dp[n][amount] == (amount + 1) else dp[n][amount]
```

#### 提交 689704752 · Wrong Answer · 2026-01-07 16:12:12 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [[amount + 1] * (amount + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            coin = coins[i - 1]
            for x in range(amount + 1):
                dp[i][x] = dp[i - 1][x]
                if x >= coin:
                    dp[i][x] = min(dp[i - 1][x], dp[i - 1][x - coin])
        return dp[n][amount]
```

#### 提交 690126090 · Wrong Answer · 2026-01-09 12:41:58 CST · python3

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        dp = [[amount + 1] * (amount + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            coin = coins[i - 1]
            for j in range(amount + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= coin:
                    dp[i][j] = min(dp[i][j], dp[i][j - coin] + 1)
        return dp[n][amount]
```


## 掷骰子等于目标和的方法数 (`number-of-dice-rolls-with-target-sum`)

- 题目链接：https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/
- 难度：Medium
- 标签：动态规划
- 总提交次数：7
- 最近提交时间：2026-01-09 11:44:19 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-06 16:14:16 CST

```markdown
* 状态：dp[i][s] 表示用 i 个 k 面骰子掷出点数和为 s 的方案数
* 选择：最后一个骰子掷出点数 x（1..k）
* 转移：dp[i][s] += dp[i-1][s-x]（要求 s >= x）
* 初始化：dp[0][0] = 1
* 答案：dp[n][target]
* 全程对 1e9+7 取模
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 689459442 | 2026-01-06 16:21:16 CST | python3 | Accepted | 319 ms | 19.5 MB |
| 689473829 | 2026-01-06 17:01:23 CST | python3 | Runtime Error | N/A | N/A |
| 689473957 | 2026-01-06 17:01:47 CST | python3 | Runtime Error | N/A | N/A |
| 689474067 | 2026-01-06 17:02:07 CST | python3 | Wrong Answer | N/A | N/A |
| 689474273 | 2026-01-06 17:02:44 CST | python3 | Accepted | 203 ms | 19.2 MB |
| 690117288 | 2026-01-09 11:43:16 CST | python3 | Wrong Answer | N/A | N/A |
| 690117546 | 2026-01-09 11:44:19 CST | python3 | Accepted | 274 ms | 19.5 MB |

### 未通过提交代码
#### 提交 689473829 · Runtime Error · 2026-01-06 17:01:23 CST · python3

```python
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        mod = 10 ** 9 + 7
        prev = [0] * (target + 1)  # 表示前 i-1 个骰子凑出 j 的方案数
        prev[0] = 1
        for _ in range(n + 1):
            curr = [0] * (n + 1)
            for j in range(1, target + 1):
                total = 0
                for x in range(1, min(k, j) + 1):
                    total += prev[j - x]
                prev[j] = total % mod
            prev = curr
        return prev[target]
```

#### 提交 689473957 · Runtime Error · 2026-01-06 17:01:47 CST · python3

```python
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        mod = 10 ** 9 + 7
        prev = [0] * (target + 1)  # 表示前 i-1 个骰子凑出 j 的方案数
        prev[0] = 1
        for _ in range(n + 1):
            curr = [0] * (n + 1)
            for j in range(1, target + 1):
                total = 0
                for x in range(1, min(k, j) + 1):
                    total += prev[j - x]
                curr[j] = total % mod
            prev = curr
        return prev[target]
```

#### 提交 689474067 · Wrong Answer · 2026-01-06 17:02:07 CST · python3

```python
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        mod = 10 ** 9 + 7
        prev = [0] * (target + 1)  # 表示前 i-1 个骰子凑出 j 的方案数
        prev[0] = 1
        for _ in range(n + 1):
            curr = [0] * (target + 1)
            for j in range(1, target + 1):
                total = 0
                for x in range(1, min(k, j) + 1):
                    total += prev[j - x]
                curr[j] = total % mod
            prev = curr
        return prev[target]
```

#### 提交 690117288 · Wrong Answer · 2026-01-09 11:43:16 CST · python3

```python
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        mod = 10 ** 9 + 7
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(target + 1):
                total = 0
                for x in range(min(j, k) + 1):
                    total += dp[i - 1][j - x]
                dp[i][j] = total % mod
        return dp[n][target]
```


## 组合总和 Ⅳ (`combination-sum-iv`)

- 题目链接：https://leetcode.cn/problems/combination-sum-iv/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：4
- 最近提交时间：2026-01-09 11:17:08 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-05 15:42:09 CST

```markdown
dp[t] 表示凑出和为 t 的排列数。dp[0]=1 作为空序列。外层遍历 t 从 1 到 target，内层遍历每个 num：如果 t>=num，则 dp[t] += dp[t-num]，表示最后一步选 num，前面部分有 dp[t-num] 种排列。因为要计顺序，所以必须 t 外层，这样不同顺序会在不同的“最后一步”路径上累计出来。

时间 O(target*len(nums))，空间 O(target)。

状态：dp[t] = 和为 t 的排列数量
选择：最后一个数选哪个 num
转移：dp[t] += dp[t - num]（若 t >= num）
初始化：dp[0] = 1（凑出 0 的“空排列”是 1 种）

用一个非常“口语化但严格”的方式理解：按“最后一步选了什么”来分类计数。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 689193569 | 2026-01-05 15:39:15 CST | python3 | Accepted | 55 ms | 19.2 MB |
| 689443001 | 2026-01-06 15:36:12 CST | python3 | Accepted | 77 ms | 19 MB |
| 690109005 | 2026-01-09 11:14:24 CST | python3 | Wrong Answer | N/A | N/A |
| 690109806 | 2026-01-09 11:17:08 CST | python3 | Accepted | 67 ms | 19.1 MB |

### 未通过提交代码
#### 提交 690109005 · Wrong Answer · 2026-01-09 11:14:24 CST · python3

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(target + 1):
                if j >= num:
                    dp[j] += dp[j - num]
        return dp[target]
```


## 将一个数字表示成幂的和的方案数 (`ways-to-express-an-integer-as-sum-of-powers`)

- 题目链接：https://leetcode.cn/problems/ways-to-express-an-integer-as-sum-of-powers/
- 难度：Medium
- 标签：动态规划
- 总提交次数：10
- 最近提交时间：2026-01-09 10:51:31 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 689843700 | 2026-01-08 09:29:33 CST | python3 | Runtime Error | N/A | N/A |
| 689844008 | 2026-01-08 09:31:54 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 689844337 | 2026-01-08 09:34:10 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 689844474 | 2026-01-08 09:35:09 CST | python3 | Wrong Answer | N/A | N/A |
| 689844643 | 2026-01-08 09:36:22 CST | python3 | Wrong Answer | N/A | N/A |
| 689845429 | 2026-01-08 09:41:25 CST | python3 | Accepted | 739 ms | 20.9 MB |
| 690101886 | 2026-01-09 10:49:06 CST | python3 | Wrong Answer | N/A | N/A |
| 690102092 | 2026-01-09 10:49:50 CST | python3 | Wrong Answer | N/A | N/A |
| 690102491 | 2026-01-09 10:51:25 CST | python3 | Runtime Error | N/A | N/A |
| 690102532 | 2026-01-09 10:51:31 CST | python3 | Accepted | 1043 ms | 22.1 MB |

### 未通过提交代码
#### 提交 689843700 · Runtime Error · 2026-01-08 09:29:33 CST · python3

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        powers = []
        mode =  10 ** 9 + 1
        val = 1
        while val <= n:
            powers.append(val)
            val = val ** x
        k = len(powers)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            num = powers[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] = (dp[i]][j] + dp[i - 1][j - num]) % mode
        return dp[k][n]
```

#### 提交 689844008 · Memory Limit Exceeded · 2026-01-08 09:31:54 CST · python3

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        powers = []
        mode = 10 ** 9 + 1
        val = 1
        while val <= n:
            powers.append(val)
            val = val ** x
        k = len(powers)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            num = powers[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - num])
                dp[i][j] %= mode
        return dp[k][n]
```

#### 提交 689844337 · Memory Limit Exceeded · 2026-01-08 09:34:10 CST · python3

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        powers = []
        mode = 10 ** 9 + 1
        val = 1
        while val <= n:
            powers.append(val)
            val = val ** x
        k = len(powers)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            num = powers[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - num]) % mode
        return dp[k][n]
```

#### 提交 689844474 · Wrong Answer · 2026-01-08 09:35:09 CST · python3

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        powers = []
        mode = 10 ** 9 + 1
        val = 1
        while val <= n:
            powers.append(val)
            val += 1
            val = val ** x
        k = len(powers)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            num = powers[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - num]) % mode
        return dp[k][n]
```

#### 提交 689844643 · Wrong Answer · 2026-01-08 09:36:22 CST · python3

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        powers = []
        mode = 10 ** 9 + 7
        val = 1
        while val <= n:
            powers.append(val)
            val += 1
            val = val ** x
        k = len(powers)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            num = powers[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - num]) % mode
        return dp[k][n]
```

#### 提交 690101886 · Wrong Answer · 2026-01-09 10:49:06 CST · python3

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        powers = []
        base = 1
        while True:
            value = base ** x
            if value > n:
                break
            powers.append(value)
            base += 1
        k = len(powers)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            num = powers[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += dp[i - 1][j - num]
        return dp[k][n]
```

#### 提交 690102092 · Wrong Answer · 2026-01-09 10:49:50 CST · python3

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        powers = []
        base = 1
        while True:
            value = base ** x
            if value > n:
                break
            powers.append(value)
            base += 1
        mod = 10 ** 9 + 7
        k = len(powers)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            num = powers[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += (dp[i - 1][j - num]) % mod
        return dp[k][n]
```

#### 提交 690102491 · Runtime Error · 2026-01-09 10:51:25 CST · python3

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        powers = []
        base = 1
        while True:
            value = base ** x
            if value > n:
                break
            powers.append(value)
            base += 1
        mod = 10 ** 9 + 7
        k = len(powers)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(1, k + 1):
            num = powers[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += dp[i - 1][j - num]
                dp[i][j] % = mod
        return dp[k][n]
```


## 目标和 (`target-sum`)

- 题目链接：https://leetcode.cn/problems/target-sum/
- 难度：Medium
- 标签：数组, 动态规划, 回溯
- 总提交次数：11
- 最近提交时间：2026-01-09 10:44:44 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-05 13:53:37 CST

```markdown
看到：
* 每个元素 只能用一次
* 选择 + 或 -
* 问 方案数

立刻想到：
* 用代数把 “+/-” 变成 “选子集”
* 变成 0-1 背包计数版：dp 里存“方案数”，转移用加法

令 P 为取正号的元素和，N 为取负号的元素和，则 P-N=target 且 P+N=total，可得 P=(total+target)/2。于是问题变成：从数组选一些数凑出 need=P 的方案数。
用 0-1 背包计数：dp[i][j] 表示前 i 个数凑出 j 的方案数，dp[0][0]=1，转移 dp[i][j]=dp[i-1][j]+dp[i-1][j-num]。

时间：O(n * need)
空间：O(n * need)（二维版；可优化到 O(need) )
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688677956 | 2026-01-03 08:57:14 CST | python3 | Runtime Error | N/A | N/A |
| 688677989 | 2026-01-03 08:58:07 CST | python3 | Runtime Error | N/A | N/A |
| 688678035 | 2026-01-03 08:59:16 CST | python3 | Accepted | 39 ms | 17.1 MB |
| 688949676 | 2026-01-04 15:51:40 CST | python3 | Runtime Error | N/A | N/A |
| 688949719 | 2026-01-04 15:51:47 CST | python3 | Runtime Error | N/A | N/A |
| 688950081 | 2026-01-04 15:52:50 CST | python3 | Accepted | 43 ms | 17.1 MB |
| 689162895 | 2026-01-05 13:59:31 CST | python3 | Accepted | 39 ms | 17.3 MB |
| 689750010 | 2026-01-07 19:01:29 CST | python3 | Wrong Answer | N/A | N/A |
| 689750237 | 2026-01-07 19:02:45 CST | python3 | Accepted | 51 ms | 19.1 MB |
| 690100624 | 2026-01-09 10:44:29 CST | python3 | Runtime Error | N/A | N/A |
| 690100695 | 2026-01-09 10:44:44 CST | python3 | Accepted | 39 ms | 19.1 MB |

### 未通过提交代码
#### 提交 688677956 · Runtime Error · 2026-01-03 08:57:14 CST · python3

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if target > total:
            return 0
        if (total + target) % 2 == 1:
            return 0
        n = len(nums)
        need = (total + target) // 2
        dp  = [[0] * (need + 1) for _ in range(n)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(0, need + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += dp[i -1][j - num]
        return dp[n][need]
```

#### 提交 688677989 · Runtime Error · 2026-01-03 08:58:07 CST · python3

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if target > total:
            return 0
        if (total + target) % 2 == 1:
            return 0
        n = len(nums)
        need = (total + target) // 2
        dp  = [[0] * (need + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(0, need + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += dp[i - 1][j - num]
        return dp[n][need]
```

#### 提交 688949676 · Runtime Error · 2026-01-04 15:51:40 CST · python3

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        # P 为取正的所有数，N 为取负的所有数
        # P + N = total， P - N = target，所以 P = (total + target) / 2
        # 转化成背包问题：用前 i 个数凑出和为 P 的方案数
        if (total + target) % 2 = 1:
            return 0
        need = (total + target) // 2
        n = len(nums)
        dp = [[0] * (need + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(need + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += dp[i - 1][j - num]
        return dp[n][need]
```

#### 提交 688949719 · Runtime Error · 2026-01-04 15:51:47 CST · python3

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        # P 为取正的所有数，N 为取负的所有数
        # P + N = total， P - N = target，所以 P = (total + target) / 2
        # 转化成背包问题：用前 i 个数凑出和为 P 的方案数
        if (total + target) % 2 == 1:
            return 0
        need = (total + target) // 2
        n = len(nums)
        dp = [[0] * (need + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(need + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += dp[i - 1][j - num]
        return dp[n][need]
```

#### 提交 689750010 · Wrong Answer · 2026-01-07 19:01:29 CST · python3

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > total:
            return 0
        need = (total + target) // 2
        n = len(nums)
        dp = [[0] * (need + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(need + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += dp[i - 1][j - num]
        return dp[n][need]
```

#### 提交 690100624 · Runtime Error · 2026-01-09 10:44:29 CST · python3

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > total:
            return 0
        if (total + target) % 2 == 1:
            return 0
        need = (target + total) // 2
        dp = [[0] * (need + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(need + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= num:
                    dp[i][j] += dp[i - 1][j - num]
        return dp[n][need]
```


## 单词拆分 (`word-break`)

- 题目链接：https://leetcode.cn/problems/word-break/
- 难度：Medium
- 标签：字典树, 记忆化搜索, 数组, 哈希表, 字符串, 动态规划
- 总提交次数：9
- 最近提交时间：2026-01-09 10:33:00 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-05 16:06:02 CST

```markdown
为什么用动态规划做（本质）
* 把“是否能拆分”转为“前缀可达性”：如果 s[:j] 可拆分且 s[j:i] 在字典中，则 s[:i] 也可拆分。
* 存在最优子结构：要判断 s[:i] 是否可拆分，只需要看某个合法切分点 j 是否把问题化简为 s[:j]。
* 用 dp[i] 表示 s[:i] 是否可拆分，dp[0] 表示空串可拆分（True），答案是 dp[n]。

定义 dp[i] 表示 s 的前 i 个字符能否被拆分。dp[0]=True。对每个位置 i，枚举最后一个单词的长度 L（或枚举切分点 j），如果 dp[i-L] 为真且 s[i-L:i] 在字典中，则 dp[i]=True。最后返回 dp[n]。用 set 加速查词，并用字典最大单词长度剪枝。时间 O(n * maxLen)（或 O(n^2)），空间 O(n)。

状态：dp[i]：前缀 s[:i] 是否可达
选择：最后一步选哪个单词（等价于选一个切分点 j）
转移：dp[i] |= dp[j] and s[j:i] in dict
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 676188492 | 2025-11-06 09:33:52 CST | python3 | Runtime Error | N/A | N/A |
| 676188629 | 2025-11-06 09:34:37 CST | python3 | Accepted | 3 ms | 17.5 MB |
| 676477579 | 2025-11-07 14:01:12 CST | python3 | Accepted | 8 ms | 17.4 MB |
| 676478040 | 2025-11-07 14:03:33 CST | python3 | Accepted | 3 ms | 17.7 MB |
| 677328098 | 2025-11-11 14:22:15 CST | python3 | Runtime Error | N/A | N/A |
| 677328149 | 2025-11-11 14:22:28 CST | python3 | Accepted | 3 ms | 17.4 MB |
| 689208654 | 2026-01-05 16:22:09 CST | python3 | Runtime Error | N/A | N/A |
| 689208694 | 2026-01-05 16:22:17 CST | python3 | Accepted | 0 ms | 19.2 MB |
| 690097706 | 2026-01-09 10:33:00 CST | python3 | Accepted | 4 ms | 19 MB |

### 未通过提交代码
#### 提交 676188492 · Runtime Error · 2025-11-06 09:33:52 CST · python3

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)
        n = len(s)
        dp = [False] * (n+1)
        dp[0] = True
        for i range(1, n+1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[n]
```

#### 提交 677328098 · Runtime Error · 2025-11-11 14:22:15 CST · python3

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s):
        dp = [False] * (n+1)
        # 空串可拆分
        dp[0] = True
        max_len = max([len(w) for w in wordDict])
        for i in range(1, n+1):
            start = max(0, i-max_len)
            # 枚举切分点
            for j in range (start, i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
        return dp[n]
```

#### 提交 689208654 · Runtime Error · 2026-01-05 16:22:09 CST · python3

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        word_set = set(wordDict)
        max_len = max([len(w) for w in word_set]
        for i in range(1, n + 1):
            start = max(0, i - max_len)
            for j in range(start, i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[n]
```


## 执行操作可获得的最大总奖励 I (`maximum-total-reward-using-operations-i`)

- 题目链接：https://leetcode.cn/problems/maximum-total-reward-using-operations-i/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：6
- 最近提交时间：2026-01-09 10:08:09 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-04 14:39:10 CST

```markdown
题型直觉（看到就该想到）
判断是不是 0-1背包，你只问一句：
同一个奖励值 v 能不能被选两次？
这题规则要求选 v 时必须 v > S。一旦你选了 v，新的和变成 S' = S + v，必然 S' >= v，以后就不可能再满足 v > S'，所以 v 永远不可能再被选第二次。因此（至少在“去重”后）每个 v 都是“最多用一次” ⇒ 0-1 背包味道就出来了。

另外一个关键观察（帮你建立框架感）：
* 一旦你选了某个值 v，总和会变成 >= v，所以同一个数值不可能再选第二次；因此可以先对 rewardValues 去重。

先对 rewardValues 去重排序，因为任意可行选择序列必然严格递增。设 dp[i][s] 表示用前 i 个值能否得到总和 s。转移：不选则继承 dp[i-1][s]；若选当前 v，则必须保证选之前的和 prev=s-v 小于 v，同时 prev 可达，即 dp[i-1][prev] 为真。总和上界是 2M-1，所以状态空间是 O(n*M)，最后取 dp[n] 里最大的可达 s。

* 时间：O(n * M)（更准确是 O(n * 2M)）
* 空间：O(n * M)

为什么要排序：由于规则要求每次选的奖励必须大于当前总和，所以一旦选了一个较大的值，所有更小的值都会被永久阻塞。因此任何可行方案的选取顺序必然是严格递增的，我们可以先对 rewardValues 去重并排序，再做 DP 而无需考虑排列顺序。

为什么上界是 2M-1？
设你最终选到的最大值是 M（数组最大奖励）。要选 M 的那一步，之前的和一定是 S < M，选完后总和：
S + M < 2M，总和是整数 ⇒ S + M <= 2M - 1
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688898064 | 2026-01-04 12:02:45 CST | python3 | Wrong Answer | N/A | N/A |
| 688898232 | 2026-01-04 12:03:41 CST | python3 | Accepted | 991 ms | 78 MB |
| 688938906 | 2026-01-04 15:18:43 CST | python3 | Wrong Answer | N/A | N/A |
| 688940015 | 2026-01-04 15:21:57 CST | python3 | Accepted | 1125 ms | 77.9 MB |
| 689133665 | 2026-01-05 11:20:23 CST | python3 | Accepted | 1065 ms | 77.9 MB |
| 690092013 | 2026-01-09 10:08:09 CST | python3 | Accepted | 1090 ms | 80.1 MB |

### 未通过提交代码
#### 提交 688898064 · Wrong Answer · 2026-01-04 12:02:45 CST · python3

```python
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        values = sorted(set(rewardValues))
        n = len(values)
        max_value = values[-1]
        limit = values[-1]
        dp = [[False] * (limit + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for i in range(1, n + 1):
            v = values[i - 1]
            for s in range(0, limit + 1):
                dp[i][s] = dp[i - 1][s]
                prev = s - v
                if prev >= 0 and prev < v and dp[i - 1][prev]:
                    dp[i][s] = True
        for s in range(limit, -1, -1):
            if dp[n][s]:
                return s
        return 0
```

#### 提交 688938906 · Wrong Answer · 2026-01-04 15:18:43 CST · python3

```python
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        values = sorted(set(rewardValues))
        n = len(values)
        max_value = values[-1]
        limit = 2 * max_value - 1
        # dp[i][j] 表示用前 i 个元素能否凑出和为 j 的总奖励
        dp = [[False] * (limit + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for i in range(n + 1):
            value = values[i - 1]
            for j in range(limit + 1):
                dp[i][j] = dp[i - 1][j]
                prev = j - value
                if prev >= 0 and prev < value and dp[i - 1][prev]:
                    dp[i][j] = True
        for best in range(limit, -1, -1):
            if dp[n][best]:
                return best
```


## 最后一块石头的重量 II (`last-stone-weight-ii`)

- 题目链接：https://leetcode.cn/problems/last-stone-weight-ii/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：12
- 最近提交时间：2026-01-07 18:01:42 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-05 10:44:39 CST

```markdown
“两两粉碎、剩余最小”这种题，最终等价于：把所有石头分成两堆 A/B，最后剩下的是 |sum(A) - sum(B)|
所以目标变成：让两堆尽量接近 ⇒ 找一个子集和 best 尽量接近 total/2，答案就是 total - 2best

这题等价于把石头分成两堆，最后重量是两堆和的差。令 total 为总和，只要找一堆的和 best 尽量接近 total/2，答案是 total-2best。
用 0-1 背包可行性 DP：dp[i][j] 表示前 i 个石头能否凑出 j，转移为选或不选当前石头。最后在 j<=total/2 中找最大的可达 j 作为 best。

注意 total // 2 只是我们希望找到的一堆的和尽量接近总和的一半的上界，它不一定能凑出来
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688680098 | 2026-01-03 09:41:16 CST | python3 | Wrong Answer | N/A | N/A |
| 688680155 | 2026-01-03 09:42:09 CST | python3 | Wrong Answer | N/A | N/A |
| 688680186 | 2026-01-03 09:42:28 CST | python3 | Wrong Answer | N/A | N/A |
| 688680236 | 2026-01-03 09:43:14 CST | python3 | Accepted | 7 ms | 17.3 MB |
| 688749224 | 2026-01-03 16:47:42 CST | python3 | Wrong Answer | N/A | N/A |
| 688753427 | 2026-01-03 17:05:39 CST | python3 | Accepted | 11 ms | 17.2 MB |
| 688920552 | 2026-01-04 14:21:29 CST | python3 | Accepted | 10 ms | 17.6 MB |
| 689130386 | 2026-01-05 11:10:01 CST | python3 | Wrong Answer | N/A | N/A |
| 689130497 | 2026-01-05 11:10:22 CST | python3 | Accepted | 7 ms | 17.2 MB |
| 689738988 | 2026-01-07 17:59:20 CST | python3 | Wrong Answer | N/A | N/A |
| 689739154 | 2026-01-07 18:00:01 CST | python3 | Wrong Answer | N/A | N/A |
| 689739542 | 2026-01-07 18:01:42 CST | python3 | Accepted | 11 ms | 19.4 MB |

### 未通过提交代码
#### 提交 688680098 · Wrong Answer · 2026-01-03 09:41:16 CST · python3

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        target = total // 2
        n = len(stones)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = True
        for i in range(1, n + 1):
            stone = stones[i - 1]
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - stone]
        for best in range(target, -1, -1):
            if dp[n][best]:
                return total - 2 * best
        return 0
```

#### 提交 688680155 · Wrong Answer · 2026-01-03 09:42:09 CST · python3

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        target = total // 2
        n = len(stones)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = True
        for i in range(1, n + 1):
            stone = stones[i - 1]
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j] or (j >= total and dp[i - 1][j - stone])
        for best in range(target, -1, -1):
            if dp[n][best]:
                return total - 2 * best
        return 0
```

#### 提交 688680186 · Wrong Answer · 2026-01-03 09:42:28 CST · python3

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        target = total // 2
        n = len(stones)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = True
        for i in range(1, n + 1):
            stone = stones[i - 1]
            for j in range(1, target + 1):
                dp[i][j] = dp[i - 1][j] or (j >= total and dp[i - 1][j - stone])
        for best in range(target, -1, -1):
            if dp[n][best]:
                return total - 2 * best
        return 0
```

#### 提交 688749224 · Wrong Answer · 2026-01-03 16:47:42 CST · python3

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        target = total // 2
        n = len(stones)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = True
        for i in range(1, n + 1):
            stone = stones[i - 1]
            for j in range(1, target + 1):
                dp[i][j] = dp[i - 1][j] or (j >= stone and dp[i - 1][j - stone])
        # for best in range(target, -1, -1):
        #     if dp[n][best]:
        #         return total - 2 * best
        return total - 2 * target
        # return 0
```

#### 提交 689130386 · Wrong Answer · 2026-01-05 11:10:01 CST · python3

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        target = total // 2
        n = len(stones)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for i in range(1, n + 1):
            stone = stones[i - 1]
            for j in range(target + 1):
                dp[i][j] = dp[i - 1][j] or (j >= stone and dp[i - 1][j - stone])
        
        for best in range(target, -1, -1):
            if dp[n][best]:
                return best
```

#### 提交 689738988 · Wrong Answer · 2026-01-07 17:59:20 CST · python3

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        target = total // 2
        n = len(stones)
        dp  = [[False] * (target + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            stone = stones[i - 1]
            for x in range(target + 1):
                dp[i][x] = dp[i - 1][x] or (x >= stone and dp[i - 1][x - stone])
        for best in range(target, -1, -1):
            if dp[n][best]:
                return best
```

#### 提交 689739154 · Wrong Answer · 2026-01-07 18:00:01 CST · python3

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        target = total // 2
        n = len(stones)
        dp  = [[False] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for i in range(1, n + 1):
            stone = stones[i - 1]
            for x in range(target + 1):
                dp[i][x] = dp[i - 1][x] or (x >= stone and dp[i - 1][x - stone])
        for best in range(target, -1, -1):
            if dp[n][best]:
                return best
```


## 零钱兑换 II (`coin-change-ii`)

- 题目链接：https://leetcode.cn/problems/coin-change-ii/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：8
- 最近提交时间：2026-01-07 16:20:29 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-06 13:53:17 CST

```markdown
二维的写法跟 416 题目有点像，却别只在于能否重复使用

一维的解法：
dp[j] 表示用当前已处理的硬币种类，凑出金额 j 的组合数。初始化 dp[0]=1（凑出 0 只有一种：什么都不选）。遍历每个 coin，j 从 coin 到 amount 正序：dp[j] += dp[j-coin]，表示在凑出 j-coin 的每种方案后再加一枚 coin。coin 放外层保证按“硬币种类”递增构造，从而只计组合不计排列。复杂度 O(amount * len(coins))，空间 O(amount)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688559918 | 2026-01-02 14:32:59 CST | python3 | Accepted | 179 ms | 17.5 MB |
| 688564942 | 2026-01-02 15:01:19 CST | python3 | Accepted | 443 ms | 62 MB |
| 688943221 | 2026-01-04 15:31:48 CST | python3 | Accepted | 522 ms | 62 MB |
| 689136599 | 2026-01-05 11:29:51 CST | python3 | Accepted | 435 ms | 61.8 MB |
| 689414665 | 2026-01-06 14:03:30 CST | python3 | Accepted | 171 ms | 19.5 MB |
| 689421576 | 2026-01-06 14:30:49 CST | python3 | Accepted | 447 ms | 64.2 MB |
| 689707749 | 2026-01-07 16:20:16 CST | python3 | Runtime Error | N/A | N/A |
| 689707827 | 2026-01-07 16:20:29 CST | python3 | Accepted | 475 ms | 64.2 MB |

### 未通过提交代码
#### 提交 689707749 · Runtime Error · 2026-01-07 16:20:16 CST · python3

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        dp = [[0] * (amount + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            coin = coins[i - 1]:
            for x in range(amount + 1):
                dp[i][x] = dp[i - 1][x]
                if x >= coin:
                    dp[i][x] += dp[i][x - coin]
        return dp[n][amount]
```


## 完全平方数 (`perfect-squares`)

- 题目链接：https://leetcode.cn/problems/perfect-squares/
- 难度：Medium
- 标签：广度优先搜索, 数学, 动态规划
- 总提交次数：7
- 最近提交时间：2026-01-06 15:03:49 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-06 11:49:10 CST

```markdown
把所有不超过 n 的平方数当作硬币面值，比如 1、4、9… 
dp[x] 表示凑出 x 的最少平方数个数。初始化 dp[0]=0，其它为n + 1。
对每个平方数 sq（完全背包可重复），遍历 x 从 sq 到 n 正序：dp[x]=min(dp[x], dp[x-sq]+1)。最后返回 dp[n]。时间 O(n*sqrt(n))，空间 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 689188353 | 2026-01-05 15:23:55 CST | python3 | Accepted | 1492 ms | 19.4 MB |
| 689392598 | 2026-01-06 11:49:22 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 689393048 | 2026-01-06 11:51:14 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 689393141 | 2026-01-06 11:51:34 CST | python3 | Wrong Answer | N/A | N/A |
| 689393512 | 2026-01-06 11:53:06 CST | python3 | Accepted | 1437 ms | 19.3 MB |
| 689431164 | 2026-01-06 15:02:06 CST | python3 | Wrong Answer | N/A | N/A |
| 689431659 | 2026-01-06 15:03:49 CST | python3 | Accepted | 2757 ms | 27.8 MB |

### 未通过提交代码
#### 提交 689392598 · Memory Limit Exceeded · 2026-01-06 11:49:22 CST · python3

```python
class Solution:
    def numSquares(self, n: int) -> int:
        k = 1
        squares = []
        while k ^ 2 <= n:
            squares.append(k ^ 2)
            k + 1
        dp = [n + 1] * (n + 1)
        dp[0] = 0
        for sq in squares:
            for x in range(1, n + 1):
                dp[x] = min(dp[x], dp[x - sq] + 1)
        return dp[n]
```

#### 提交 689393048 · Memory Limit Exceeded · 2026-01-06 11:51:14 CST · python3

```python
class Solution:
    def numSquares(self, n: int) -> int:
        k = 1
        squares = []
        while k ** 2 <= n:
            squares.append(k ** 2)
            k + 1
        dp = [n + 1] * (n + 1)
        dp[0] = 0
        for sq in squares:
            for x in range(1, n + 1):
                dp[x] = min(dp[x], dp[x - sq] + 1)
        return dp[n]
```

#### 提交 689393141 · Wrong Answer · 2026-01-06 11:51:34 CST · python3

```python
class Solution:
    def numSquares(self, n: int) -> int:
        k = 1
        squares = []
        while k ** 2 <= n:
            squares.append(k ** 2)
            k += 1
        dp = [n + 1] * (n + 1)
        dp[0] = 0
        for sq in squares:
            for x in range(1, n + 1):
                dp[x] = min(dp[x], dp[x - sq] + 1)
        return dp[n]
```

#### 提交 689431164 · Wrong Answer · 2026-01-06 15:02:06 CST · python3

```python
class Solution:
    def numSquares(self, n: int) -> int:
        k = 1
        squares = []
        while k ** 2 <= n:
            squares.append(k ** 2)
            k += 1
        m = len(squares)
        dp = [[n + 1] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, m + 1):
            sq = squares[i - 1]
            for j in range(n + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= sq:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - sq] + 1)
        return dp[m][n]
```


## 分割等和子集 (`partition-equal-subset-sum`)

- 题目链接：https://leetcode.cn/problems/partition-equal-subset-sum/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：6
- 最近提交时间：2026-01-05 10:56:07 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-02 11:23:53 CST

```markdown
二维解法的思路：
        - 令 target = sum(nums) / 2，问题变成：能否选一些数凑出 target
        - dp[i][j] 表示：只使用前 i 个数（nums[0..i-1]），能否凑出和 j
        - 转移（0-1 背包，可行性）：
            dp[i][j] = dp[i-1][j] or (j >= num and dp[i-1][j-num])
				时间 O(ntarget)，空间 O(ntarget)。

本题所对应背包语言是：
i = 可用物品范围；j = 背包容量（这里是“和”）

对第 i 个数（num = nums[i-1]），只有两种选择：
* 不选这个数
* 选这个数（前提是放得下：j >= num）

这就是典型 0-1：每个数最多用一次。

一维的写法看了半天都没看懂，先不看了
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688450338 | 2026-01-01 17:50:45 CST | python3 | Accepted | 1179 ms | 32.1 MB |
| 688451649 | 2026-01-01 18:00:59 CST | python3 | Accepted | 1127 ms | 32.1 MB |
| 688530396 | 2026-01-02 11:30:11 CST | python3 | Accepted | 1111 ms | 32.2 MB |
| 688917782 | 2026-01-04 14:10:16 CST | python3 | Accepted | 1159 ms | 32.2 MB |
| 689125533 | 2026-01-05 10:55:42 CST | python3 | Wrong Answer | N/A | N/A |
| 689125662 | 2026-01-05 10:56:07 CST | python3 | Accepted | 1247 ms | 32.2 MB |

### 未通过提交代码
#### 提交 689125533 · Wrong Answer · 2026-01-05 10:55:42 CST · python3

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total =  sum(nums)
        if total % 2 == 1:
            return False
        target = total // 2
        n = len(nums)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(target + 1):
                dp[i][j] = dp[i - 1][j] or (j >= num and dp[i - 1][j - num])
        return dp[n][target]
```


## 打家劫舍 III (`house-robber-iii`)

- 题目链接：https://leetcode.cn/problems/house-robber-iii/
- 难度：Medium
- 标签：树, 深度优先搜索, 动态规划, 二叉树
- 总提交次数：2
- 最近提交时间：2026-01-03 11:25:11 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-03 11:19:11 CST

```markdown
对每个节点返回两个值：rob 表示偷当前节点的最大金额，skip 表示不偷当前节点的最大金额。若偷当前节点，则左右孩子都不能偷，所以 rob = val + left.skip + right.skip；若不偷当前节点，则左右孩子可偷可不偷，skip = max(left.rob,left.skip) + max(right.rob,right.skip)。后序遍历递归计算，答案是 max(root.rob, root.skip)。时间 O(n)，空间 O(h) 递归栈。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688692777 | 2026-01-03 11:24:50 CST | python3 | Runtime Error | N/A | N/A |
| 688692839 | 2026-01-03 11:25:11 CST | python3 | Accepted | 11 ms | 18.2 MB |

### 未通过提交代码
#### 提交 688692777 · Runtime Error · 2026-01-03 11:24:50 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if node is None:
                return 0, 0
            left_rob, left_skip = dfs(node.left)
            right_rob, right_skip = dfs(node.right)

            curr_rob = node.val + left_skip + right_skip
            curr_skip = max(left_rob, left_skip) + max(right_rob, right_skip)
            return curr_rob, curr_skip
        root_rob, root_skip = dfs(root)
        return max(root_rob, right_skip)
```


## 打家劫舍 II (`house-robber-ii`)

- 题目链接：https://leetcode.cn/problems/house-robber-ii/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：2
- 最近提交时间：2026-01-03 11:11:12 CST

### 题目笔记
#### 笔记 1 · 更新于 2026-01-03 11:01:27 CST

```markdown
因为房子成环，0 和 n-1 不能同时偷。把问题拆成两种互斥情况：不偷最后一间（只考虑 [0..n-2]）或不偷第一间（只考虑 [1..n-1]），两种情况都是 198 的线性打家劫舍。分别求最大值取 max 即可。时间 O(n)，空间 O(1)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688690618 | 2026-01-03 11:10:53 CST | python3 | Wrong Answer | N/A | N/A |
| 688690654 | 2026-01-03 11:11:12 CST | python3 | Accepted | 0 ms | 16.9 MB |

### 未通过提交代码
#### 提交 688690618 · Wrong Answer · 2026-01-03 11:10:53 CST · python3

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        best_1 = self._rob_linear(nums, 0, n - 2)
        best_2 = self._rob_linear(nums, 1, n - 1)
    
    def _rob_linear(self, nums, left, right):
        length = right - left + 1
        if length == 1:
            return nums[left]
        if length == 2:
            return max(nums[left], nums[left + 1])
        prev_2, prev_1 = nums[left], max(nums[left], nums[left + 1])
        for i in range(left + 2, right + 1):
            curr = max(prev_1, prev_2 + nums[i])
            prev_2, prev_1 = prev_1, curr
        return prev_1
```


## 打家劫舍 (`house-robber`)

- 题目链接：https://leetcode.cn/problems/house-robber/
- 难度：Medium
- 标签：数组, 动态规划
- 总提交次数：3
- 最近提交时间：2026-01-03 10:35:20 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-16 12:28:01 CST

```markdown
* 这是典型的动态规划题。设 dp[i] 为前 i 间房能偷到的最大金额。
* 对第 i 间房有两种选择：
	* 不偷：收益是 dp[i-1]
	* 偷：那前一间不能偷，收益是 dp[i-2] + nums[i]
* 所以转移方程是 dp[i] = max(dp[i-1], dp[i-2] + nums[i])。
* 用两个变量分别保存 dp[i-1] 和 dp[i-2] 就能做到 O(1) 空间，整体时间 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685096199 | 2025-12-16 12:37:11 CST | python3 | Accepted | 0 ms | 17.4 MB |
| 688685356 | 2026-01-03 10:34:58 CST | python3 | Runtime Error | N/A | N/A |
| 688685400 | 2026-01-03 10:35:20 CST | python3 | Accepted | 0 ms | 16.9 MB |

### 未通过提交代码
#### 提交 688685356 · Runtime Error · 2026-01-03 10:34:58 CST · python3

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        prev_2, prev_1 = nums[0], max(nums[0], nums[1])
        for i in range(2, n):
            curr = max(prev_1, prev_2 + nums[i])
            prev_2, prev_1 = prev_1, curr
        return curr
```


## 最小路径和 (`minimum-path-sum`)

- 题目链接：https://leetcode.cn/problems/minimum-path-sum/
- 难度：Medium
- 标签：数组, 动态规划, 矩阵
- 总提交次数：3
- 最近提交时间：2025-12-31 10:15:37 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685048867 | 2025-12-16 09:05:00 CST | python3 | Wrong Answer | N/A | N/A |
| 685049112 | 2025-12-16 09:07:26 CST | python3 | Accepted | 15 ms | 19.7 MB |
| 688240807 | 2025-12-31 10:15:37 CST | python3 | Accepted | 11 ms | 19.1 MB |

### 未通过提交代码
#### 提交 685048867 · Wrong Answer · 2025-12-16 09:05:00 CST · python3

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 0
        # 第一列，只能从上往下走
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        # 第一行，只能从左往右走
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j]= min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        
        return dp[m-1][n-1]
```


## 下降路径最小和 (`minimum-falling-path-sum`)

- 题目链接：https://leetcode.cn/problems/minimum-falling-path-sum/
- 难度：Medium
- 标签：数组, 动态规划, 矩阵
- 总提交次数：3
- 最近提交时间：2025-12-30 17:52:37 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-30 16:42:59 CST

```markdown
这是标准网格 DP。定义 dp[i][j] 为到达第 i 行第 j 列的最小路径和。转移来自上一行的三个方向：j、j-1、j+1（越界就忽略）。初始化第一行 dp 等于原矩阵第一行。最后答案是 dp 最后一行的最小值。时间 O(n^2)，用滚动数组可把空间降到 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688128988 | 2025-12-30 17:10:22 CST | python3 | Accepted | 24 ms | 17.9 MB |
| 688139892 | 2025-12-30 17:51:51 CST | python3 | Runtime Error | N/A | N/A |
| 688140053 | 2025-12-30 17:52:37 CST | python3 | Accepted | 23 ms | 17.7 MB |

### 未通过提交代码
#### 提交 688139892 · Runtime Error · 2025-12-30 17:51:51 CST · python3

```python
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        prev = matrix[0][:]
        n = len(matrix)
        for i in range(1, n):
            curr = [0] * n
            for j in range(n):
                best_above = prev[i - 1][j]
                if j - 1 >= 0:
                    best_above = min(best_above, prev[i - 1][j - 1])
                if j + 1 < n:
                    best_above = min(best_above, prev[i - 1][j + 1])
                curr[j] = matrix[i][j] + best_above
            prev = curr
        return min(prev)
```


## 俄罗斯套娃信封问题 (`russian-doll-envelopes`)

- 题目链接：https://leetcode.cn/problems/russian-doll-envelopes/
- 难度：Hard
- 标签：数组, 二分查找, 动态规划, 排序
- 总提交次数：4
- 最近提交时间：2025-12-30 15:13:03 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-30 15:08:15 CST

```markdown
同样长度的递增子序列，结尾越小越好：因为结尾小，后面越容易接上更大的数，潜力更大。这就是 tails 的含义。

这是二维严格递增问题，可以转成一维 LIS：先按宽度升序排序，宽度相同按高度降序，避免同宽信封被误选。然后取排序后的高度序列，对高度做 300 题同款的 patience sorting + 二分的 LIS，得到最大嵌套数量。时间复杂度 O(n log n)，空间 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688092168 | 2025-12-30 15:11:48 CST | python3 | Runtime Error | N/A | N/A |
| 688092392 | 2025-12-30 15:12:31 CST | python3 | Runtime Error | N/A | N/A |
| 688092487 | 2025-12-30 15:12:53 CST | python3 | Runtime Error | N/A | N/A |
| 688092535 | 2025-12-30 15:13:03 CST | python3 | Accepted | 160 ms | 50.6 MB |

### 未通过提交代码
#### 提交 688092168 · Runtime Error · 2025-12-30 15:11:48 CST · python3

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        envelopes.sort(lambda: x: (x[0], -x[1]))
        tails = []
        for _, h in envelopes:
            pos = bisect.bisect_left(tails, h)
            if pos == len(pos):
                tails.append(h)
            else:
                tails[pos] = h
        return len(tails)
```

#### 提交 688092392 · Runtime Error · 2025-12-30 15:12:31 CST · python3

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        envelopes.sort(lambda x: (x[0], -x[1]))
        tails = []
        for _, h in envelopes:
            pos = bisect.bisect_left(tails, h)
            if pos == len(pos):
                tails.append(h)
            else:
                tails[pos] = h
        return len(tails)
```

#### 提交 688092487 · Runtime Error · 2025-12-30 15:12:53 CST · python3

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        tails = []
        for _, h in envelopes:
            pos = bisect.bisect_left(tails, h)
            if pos == len(pos):
                tails.append(h)
            else:
                tails[pos] = h
        return len(tails)
```


## 最长递增子序列 (`longest-increasing-subsequence`)

- 题目链接：https://leetcode.cn/problems/longest-increasing-subsequence/
- 难度：Medium
- 标签：数组, 二分查找, 动态规划
- 总提交次数：8
- 最近提交时间：2025-12-30 13:47:04 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-30 13:39:57 CST

```markdown
dp[i] 表示以 nums[i] 结尾的最长递增序列的长度

不要写错状态转移方程：dp[i] = max(dp[i], dp[j] + 1)
错误的理解：dp[i] = max(dp[i], dp[i] + 1)

时间复杂度是 O(N^2)，空间复杂度是 O(N)。”


二分查找的解法（挺难理解的）：
维护一个数组 tails，tails[k] 表示长度为 k+1 的递增子序列能达到的最小结尾。遍历每个数 x：如果 x 比 tails 最后一个大，就能把最长长度加一；否则用二分找到第一个 >= x 的位置并替换成 x，相当于把同长度的结尾变小，让后面更容易接。最后 tails 的长度就是 LIS。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682069640 | 2025-12-02 10:20:03 CST | python3 | Accepted | 1855 ms | 18 MB |
| 685537070 | 2025-12-18 10:18:02 CST | python3 | Wrong Answer | N/A | N/A |
| 685537283 | 2025-12-18 10:18:54 CST | python3 | Accepted | 1978 ms | 17.1 MB |
| 685803653 | 2025-12-19 14:31:00 CST | python3 | Accepted | 2177 ms | 17.4 MB |
| 688042807 | 2025-12-30 11:10:20 CST | python3 | Accepted | 2064 ms | 17.2 MB |
| 688071276 | 2025-12-30 13:46:17 CST | python3 | Runtime Error | N/A | N/A |
| 688071315 | 2025-12-30 13:46:31 CST | python3 | Runtime Error | N/A | N/A |
| 688071418 | 2025-12-30 13:47:04 CST | python3 | Accepted | 3 ms | 17.7 MB |

### 未通过提交代码
#### 提交 685537070 · Wrong Answer · 2025-12-18 10:18:02 CST · python3

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i] 表示为以 nums[i] 结尾的最长递增子序列的长度
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[i] + 1)
        return max(dp)
```

#### 提交 688071276 · Runtime Error · 2025-12-30 13:46:17 CST · python3

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails  = []  # tails[k] 表示长度为 k+1 的递增子序列的最小结尾
        for x in nums:
            pos = bisect.bisect_left(nums, x)
            if pos = len(tails):
                tails.append(x)
            else:
                tails[pos] = x
        return len(tails)
```

#### 提交 688071315 · Runtime Error · 2025-12-30 13:46:31 CST · python3

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails  = []  # tails[k] 表示长度为 k+1 的递增子序列的最小结尾
        for x in nums:
            pos = bisect.bisect_left(nums, x)
            if pos == len(tails):
                tails.append(x)
            else:
                tails[pos] = x
        return len(tails)
```


## 斐波那契数 (`fibonacci-number`)

- 题目链接：https://leetcode.cn/problems/fibonacci-number/
- 难度：Easy
- 标签：递归, 记忆化搜索, 数学, 动态规划
- 总提交次数：1
- 最近提交时间：2025-12-30 10:06:39 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 688024382 | 2025-12-30 10:06:39 CST | python3 | Accepted | 0 ms | 16.9 MB |

### 未通过提交代码
(所有提交均已通过)

## 单词搜索 (`word-search`)

- 题目链接：https://leetcode.cn/problems/word-search/
- 难度：Medium
- 标签：深度优先搜索, 数组, 字符串, 回溯, 矩阵
- 总提交次数：2
- 最近提交时间：2025-12-29 15:31:02 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687872018 | 2025-12-29 15:29:08 CST | python3 | Runtime Error | N/A | N/A |
| 687872640 | 2025-12-29 15:31:02 CST | python3 | Accepted | 3220 ms | 17.1 MB |

### 未通过提交代码
#### 提交 687872018 · Runtime Error · 2025-12-29 15:29:08 CST · python3

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        L = len(word)
        def dfs(r, c, idx):
            if not 0 <= r < m or not 0 <= c < n:
                return False
            if board[r][c] != word[idx]:
                return False
            origin = board[r][c]
            board[r][c] = '#'
            found = (
                dfs(r + 1, c, idx + 1) 
                or dfs(r - 1, c, idx + 1) 
                or dfs(r, c + 1, idx + 1) 
                or dfs(r, c - 1, idx + 1)
            )
            board[r][c] = origin
            return found
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False
```


## 电话号码的字母组合 (`letter-combinations-of-a-phone-number`)

- 题目链接：https://leetcode.cn/problems/letter-combinations-of-a-phone-number/
- 难度：Medium
- 标签：哈希表, 字符串, 回溯
- 总提交次数：2
- 最近提交时间：2025-12-29 15:04:05 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687864284 | 2025-12-29 15:03:51 CST | python3 | Runtime Error | N/A | N/A |
| 687864341 | 2025-12-29 15:04:05 CST | python3 | Accepted | 0 ms | 17.1 MB |

### 未通过提交代码
#### 提交 687864284 · Runtime Error · 2025-12-29 15:03:51 CST · python3

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        mapping = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }
        n = len(digits)
        res = []
        path = []
        def backtrack(pos):
            if pos == n:
                res.append(''.join(path))
            curr_digit = digits[pos]
            for ch in mapping[curr_digit]:
                path.append(ch)
                backtrack(pos + 1)
                path.pop()
        backtrack(0)
        return res
```


## 格雷编码 (`gray-code`)

- 题目链接：https://leetcode.cn/problems/gray-code/
- 难度：Medium
- 标签：位运算, 数学, 回溯
- 总提交次数：2
- 最近提交时间：2025-12-29 14:39:33 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-29 14:11:13 CST

```markdown
1 << i 的效果是：只有第 i 位是 1，其它位都是 0

异或运算 ^ 的规律：
* 0 ^ 0 = 0
* 1 ^ 0 = 1 （跟 0 异或不变）
* 0 ^ 1 = 1
* 1 ^ 1 = 0 （跟 1 异或会翻转）

所以 curr ^ (1 << i) 的效果是：
第 i 位：原来是 0 就变成 1，原来是 1 就变成 0（被 1 异或，翻转）
其它位：跟 0 异或，保持不变
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687855499 | 2025-12-29 14:32:05 CST | python3 | Accepted | 59 ms | 36 MB |
| 687857501 | 2025-12-29 14:39:33 CST | python3 | Accepted | 62 ms | 36 MB |

### 未通过提交代码
(所有提交均已通过)

## 复原 IP 地址 (`restore-ip-addresses`)

- 题目链接：https://leetcode.cn/problems/restore-ip-addresses/
- 难度：Medium
- 标签：字符串, 回溯
- 总提交次数：15
- 最近提交时间：2025-12-29 12:52:55 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-28 18:12:42 CST

```markdown
* “这道题本质是把字符串切成 4 段，枚举所有切法并过滤出合法 IP。
* 每一段只有 1~3 位，深度固定 4，搜索树非常小；而且每一段是否合法（0~255、无前导零）可以局部检查、及时剪枝。
* 所以我用回溯/DFS，从左到右依次选择每一段的长度，遇到不合法就回退，最终枚举出所有可能的 IP 地址。这类‘字符串划分 + 枚举所有合法方案’的问题，用回溯是最自然、代码也最清晰的。”

凡是要求“所有组合/分割方案”的，只要数据规模不大，回溯法就是标准解。

“固定段数、每段长度有上限”的分割题（像 IP 地址） → 搜索树高度和分支数都是常数 → 回溯是常数复杂度。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681502094 | 2025-11-29 16:34:55 CST | python3 | Wrong Answer | N/A | N/A |
| 681502289 | 2025-11-29 16:35:55 CST | python3 | Wrong Answer | N/A | N/A |
| 681502367 | 2025-11-29 16:36:19 CST | python3 | Accepted | 3 ms | 17.7 MB |
| 681502389 | 2025-11-29 16:36:25 CST | python3 | Accepted | 0 ms | 17.8 MB |
| 681551995 | 2025-11-29 21:20:26 CST | python3 | Accepted | 0 ms | 17.8 MB |
| 681608604 | 2025-11-30 10:40:53 CST | python3 | Accepted | 3 ms | 17.5 MB |
| 681820899 | 2025-12-01 10:43:48 CST | python3 | Accepted | 0 ms | 17.7 MB |
| 683027249 | 2025-12-06 15:17:20 CST | python3 | Wrong Answer | N/A | N/A |
| 683027439 | 2025-12-06 15:18:08 CST | python3 | Wrong Answer | N/A | N/A |
| 683027883 | 2025-12-06 15:20:06 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 683414273 | 2025-12-08 15:08:37 CST | python3 | Wrong Answer | N/A | N/A |
| 683414673 | 2025-12-08 15:10:00 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 687697581 | 2025-12-28 18:07:42 CST | python3 | Wrong Answer | N/A | N/A |
| 687697906 | 2025-12-28 18:09:38 CST | python3 | Accepted | 3 ms | 16.9 MB |
| 687838456 | 2025-12-29 12:52:55 CST | python3 | Accepted | 0 ms | 17.1 MB |

### 未通过提交代码
#### 提交 681502094 · Wrong Answer · 2025-11-29 16:34:55 CST · python3

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def is_valid(segment):
            if len(segment) > 1 and segment == '0':
                return False
            if int(segment) > 255:
                return False
            return True

        path = []
        res = []
        
        def backtrack(start_index):
            if len(path) == 4:
                if start_index == len(s):
                    res.append(".".join(path))
                    return
            remaining_chars = len(s) - start_index
            remaining_segments = 4 - len(path)
            if remaining_chars < remaining_segments or remaining_chars > 3 * remaining_segments:
                return
            
            for length in range(1, 4):
                end_index = start_index + length
                if end_index > len(s):
                    break
                segment = s[start_index:end_index]
                if not is_valid(segment):
                    continue
                path.append(segment)
                backtrack(end_index)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 681502289 · Wrong Answer · 2025-11-29 16:35:55 CST · python3

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def is_valid(segment):
            if not segment:
                return False
            if len(segment) > 1 and segment == '0':
                return False
            if int(segment) > 255:
                return False
            return True

        path = []
        res = []
        
        def backtrack(start_index):
            if len(path) == 4:
                if start_index == len(s):
                    res.append(".".join(path))
                    return
            remaining_chars = len(s) - start_index
            remaining_segments = 4 - len(path)
            if remaining_chars < remaining_segments or remaining_chars > 3 * remaining_segments:
                return
            
            for length in range(1, 4):
                end_index = start_index + length
                if end_index > len(s):
                    break
                segment = s[start_index:end_index]
                if not is_valid(segment):
                    continue
                path.append(segment)
                backtrack(end_index)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 683027249 · Wrong Answer · 2025-12-06 15:17:20 CST · python3

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def is_valid(segment):
            if len(segment) > 0 and segment[0] == '0':
                return False
            if int(segment) > 255:
                return False
            return True
        res = []
        path = []
        n = len(s)
        def backtrack(start_index):
            if len(path) == 4 and start_index == n:
                res.append(path[:])
                return
            remaining_chars = n - start_index
            remaining_segments = 4 - len(path)
            if remaining_chars < remaining_segments or remaining_chars > 3 * remaining_segments:
                return
            for length in range(1, 4):
                end_index = start_index + length
                if end_index > n:
                    break
                segment = s[start_index : end_index]
                if not is_valid(segment):
                    continue
                path.append(segment)
                backtrack(end_index)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 683027439 · Wrong Answer · 2025-12-06 15:18:08 CST · python3

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def is_valid(segment):
            if len(segment) > 1 and segment[0] == '0':
                return False
            if int(segment) > 255:
                return False
            return True
        res = []
        path = []
        n = len(s)
        def backtrack(start_index):
            if len(path) == 4 and start_index == n:
                res.append(path[:])
                return
            remaining_chars = n - start_index
            remaining_segments = 4 - len(path)
            if remaining_chars < remaining_segments or remaining_chars > 3 * remaining_segments:
                return
            for length in range(1, 4):
                end_index = start_index + length
                if end_index > n:
                    break
                segment = s[start_index : end_index]
                if not is_valid(segment):
                    continue
                path.append(segment)
                backtrack(end_index)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 683414273 · Wrong Answer · 2025-12-08 15:08:37 CST · python3

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def is_valid(segment):
            if len(segment) > 1 and segment[0] == '0':
                return False
            if int(segment) > 255:
                return False
            return True
        res = []
        path = []
        n = len(s)
        def backtrack(start_index):
            if len(path) == 4 and start_index == n:
                res.append('.'.join(path))
                return
            remaining_chars = n - start_index
            remaining_segments = 4 - len(path)
            if remaining_chars < remaining_segments or remaining_chars > 3 * remaining_segments:
                return
            for length in range(1, 4):
                end_index = start_index + length
                if end_index > n:
                    break
                segment = s[start_index : end_index + 1]
                if not is_valid(segment):
                    continue
                path.append(segment)
                backtrack(end_index + 1)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 687697581 · Wrong Answer · 2025-12-28 18:07:42 CST · python3

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def is_valid(segment):
            if len(segment) > 1 and segment[0] == '0':
                return False
            if int(segment) > 255:
                return False
            return True
        res = []
        path = []
        n = len(s)
        def backtrack(start_index):
            if len(path) == 4 and start_index == n:
                res.append(''.join(path))
                return
            remaining_chars = n - start_index
            remaining_segments = 4 - len(path)
            if remaining_chars < remaining_segments or remaining_chars > 3 * remaining_segments:
                return
            for length in range(1, 4):
                end_index = start_index + length
                if end_index > n:
                    break
                segment = s[start_index : end_index]
                if not is_valid(segment):
                    continue
                path.append(segment)
                backtrack(end_index)
                path.pop()
        backtrack(0)
        return res
```


## 分割回文串 (`palindrome-partitioning`)

- 题目链接：https://leetcode.cn/problems/palindrome-partitioning/
- 难度：Medium
- 标签：字符串, 动态规划, 回溯
- 总提交次数：10
- 最近提交时间：2025-12-29 11:45:09 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-28 17:35:06 CST

```markdown
定义一个递归函数，维护当前的起始切割位置 startIndex。
在每一层递归中，从 startIndex 开始向后遍历，截取子串。如果当前子串是回文串，就把它加入路径列表，然后递归处理剩下的字符串；递归返回后，进行回溯（撤销选择），尝试下一个切割位置。
当 startIndex 走到字符串末尾时，说明找到了一组有效解。


时间复杂度是 O(N \cdot 2^N)。因为字符串有 N-1 个切分点，每个点可选切或不切，最坏情况下会产生 2^N 种方案；同时每次分割我们还需要 O(N) 的时间来生成子串和判断回文。
空间复杂度是 O(N)，主要取决于递归栈的最大深度，也就是字符串的长度。

不变式：调用 backtrack(i) 时，下标 [0, i-1] 的部分已经被切分完毕且不重不漏，接下来只负责从 i 开始切。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681617872 | 2025-11-30 11:05:16 CST | python3 | Accepted | 51 ms | 34 MB |
| 681817191 | 2025-12-01 10:30:09 CST | python3 | Wrong Answer | N/A | N/A |
| 681817363 | 2025-12-01 10:30:45 CST | python3 | Wrong Answer | N/A | N/A |
| 681817720 | 2025-12-01 10:32:01 CST | python3 | Accepted | 55 ms | 33.7 MB |
| 683022358 | 2025-12-06 14:54:38 CST | python3 | Accepted | 48 ms | 33.9 MB |
| 683424854 | 2025-12-08 15:42:01 CST | python3 | Wrong Answer | N/A | N/A |
| 683425791 | 2025-12-08 15:44:59 CST | python3 | Wrong Answer | N/A | N/A |
| 683426469 | 2025-12-08 15:46:59 CST | python3 | Accepted | 51 ms | 33.9 MB |
| 687693494 | 2025-12-28 17:44:56 CST | python3 | Accepted | 43 ms | 33.5 MB |
| 687828919 | 2025-12-29 11:45:09 CST | python3 | Accepted | 47 ms | 33.5 MB |

### 未通过提交代码
#### 提交 681817191 · Wrong Answer · 2025-12-01 10:30:09 CST · python3

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        def backtrack(start_index):
            if start_index == len(s):
                res.append(path[:])
            for i in range(start_index, len(s)):
                substring = s[start_index : i + 1]
                if not substring == substring[::-1]:
                    continue
                path.append(substring)
                backtrack(start_index + 1)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 681817363 · Wrong Answer · 2025-12-01 10:30:45 CST · python3

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        def backtrack(start_index):
            if start_index == len(s):
                res.append(path[:])
                return
            for i in range(start_index, len(s)):
                substring = s[start_index : i + 1]
                if not substring == substring[::-1]:
                    continue
                path.append(substring)
                backtrack(start_index + 1)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 683424854 · Wrong Answer · 2025-12-08 15:42:01 CST · python3

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        n = len(s)
        def backtrack(start_index):
            if start_index == n:
                res.append(list(path))
                return
            for end_index in range(start_index, n):
                sub_string = s[start_index : end_index + 1]
                if not sub_string == sub_string[::-1]:
                    continue
                path.append(sub_string)
                backtrack(start_index + 1)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 683425791 · Wrong Answer · 2025-12-08 15:44:59 CST · python3

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        n = len(s)
        def backtrack(start_index):
            if start_index >= n:
                res.append(list(path))
                return
            for end_index in range(start_index, n):
                sub_string = s[start_index : end_index + 1]
                if not sub_string == sub_string[::-1]:
                    continue
                path.append(sub_string)
                backtrack(start_index + 1)
                path.pop()
        backtrack(0)
        return res
```


## 组合总和 II (`combination-sum-ii`)

- 题目链接：https://leetcode.cn/problems/combination-sum-ii/
- 难度：Medium
- 标签：数组, 回溯
- 总提交次数：4
- 最近提交时间：2025-12-29 11:39:51 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-26 21:49:30 CST

```markdown
排序是 O(n log n)。回溯过程中，相当于在枚举所有子集并用 path_sum > target 剪枝。
最坏情况下（比如 target 很大、几乎不剪枝），所有子集数量是 2ⁿ 量级，每个可行解在加入结果时要复制一次路径，最多 O(n)。
所以时间复杂度可以看成 O(n·2ⁿ)，空间复杂度是递归栈和当前路径，都是 O(n)。

看到类似「组合/子集/选或不选」这类题，几乎可以形成条件反射：
* 每个元素“选/不选” → 2ⁿ 级别的搜索空间。
* 再看有没有：
	* 排序 + 去重？
	* 剪枝（如 sum > target）？
* 复杂度一般就说：指数级 O(2ⁿ) 或 O(n·2ⁿ)，空间 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687380121 | 2025-12-26 21:44:29 CST | python3 | Runtime Error | N/A | N/A |
| 687380200 | 2025-12-26 21:44:59 CST | python3 | Accepted | 27 ms | 17 MB |
| 687827326 | 2025-12-29 11:38:04 CST | python3 | Wrong Answer | N/A | N/A |
| 687827765 | 2025-12-29 11:39:51 CST | python3 | Accepted | 23 ms | 17.1 MB |

### 未通过提交代码
#### 提交 687380121 · Runtime Error · 2025-12-26 21:44:29 CST · python3

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        candidates.sort()
        self.path_sum = 0
        n = len(candidates)
        def backtrack(start_index):
            if self.path_sum == target:
                res.append(list(path))
                return
            if self.path_sum > target:
                return
            for i in range(start_index, n):
                if i > start_index and candidates[i] == candidates[i-1]:
                    continue
                path.append(candidates[i])
                path_sum += candidates[i]
                backtrack(i+1)
                path.pop()
                path_sum -= candidates[i]
        backtrack(0)
        return res
```

#### 提交 687827326 · Wrong Answer · 2025-12-29 11:38:04 CST · python3

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        path_sum = 0
        candidates.sort()
        n = len(candidates)
        def backtrack(start_index):
            nonlocal path_sum
            if path_sum == target:
                res.append(list(path))
                return
            for i in range(start_index, n):
                if i > start_index and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                path_sum += candidates[i]
                backtrack(i + 1)
                path_sum -= candidates[i]
                path.pop()
        backtrack(1)
        return res
```


## 非递减子序列 (`non-decreasing-subsequences`)

- 题目链接：https://leetcode.cn/problems/non-decreasing-subsequences/
- 难度：Medium
- 标签：位运算, 数组, 哈希表, 回溯
- 总提交次数：4
- 最近提交时间：2025-12-29 11:26:41 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-28 15:12:04 CST

```markdown
关于 used 数组的使用，注意和第 47 题区分

排列类（每个元素只能用一次）
→ used 是 bool 数组，放外面，跨层记录 index 是否被用。
组合 / 子集 / 子序列去重类
→ used 是 本层局部 set，只管“这一层用过哪些值”，每层重建。**
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687649044 | 2025-12-28 14:55:19 CST | python3 | Wrong Answer | N/A | N/A |
| 687650500 | 2025-12-28 15:02:35 CST | python3 | Wrong Answer | N/A | N/A |
| 687650727 | 2025-12-28 15:03:36 CST | python3 | Accepted | 19 ms | 22.7 MB |
| 687824502 | 2025-12-29 11:26:41 CST | python3 | Accepted | 15 ms | 22.6 MB |

### 未通过提交代码
#### 提交 687649044 · Wrong Answer · 2025-12-28 14:55:19 CST · python3

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        path = []
        res = []
        used = set()
        n = len(nums)
        def backtrack(start_index):
            if len(path) >= 2:
                res.append(list(path))
                return
            for i in range(start_index, n):
                if path and path[-1] > nums[i]:
                    continue
                if nums[i] in used:
                    continue
                used.add(nums[i])
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
                used.remove(nums[i])
        backtrack(0)
        return res
```

#### 提交 687650500 · Wrong Answer · 2025-12-28 15:02:35 CST · python3

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        path = []
        res = []
        
        n = len(nums)
        def backtrack(start_index):
            if len(path) >= 2:
                res.append(list(path))
            used = set()  # 每一层都应该有一个 set，防止相同元素的重复使用
            for i in range(start_index, n):
                if path and path[-1] > nums[i]:
                    continue
                if nums[i] in used:
                    continue
                used.add(nums[i])
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
                used.remove(nums[i])
        backtrack(0)
        return res
```


## 优美的排列 (`beautiful-arrangement`)

- 题目链接：https://leetcode.cn/problems/beautiful-arrangement/
- 难度：Medium
- 标签：位运算, 数组, 动态规划, 回溯, 状态压缩
- 总提交次数：2
- 最近提交时间：2025-12-29 11:05:26 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-29 10:56:00 CST

```markdown
题目要求把 1..n 排列成一个长度为 n 的排列，使得每个位置 i 上的数字 x 满足 x % i == 0 或 i % x == 0。

解法就是标准回溯构造排列：用一个 used 数组标记数字 1..n 是否已经用过，按位置 i 从 1 到 n 递归。

在位置 i，只尝试还没用过且满足整除条件的数字 x，做选择后标记 used[x]=True，递归到 i+1，回溯时再恢复。

当 i > n 时说明构造出一个合法排列，计数加一。

搜索树高度是 n，理论上最坏接近 O(n!)，但整除剪枝非常强，n≤15 完全可行，空间复杂度 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687818550 | 2025-12-29 11:05:14 CST | python3 | Runtime Error | N/A | N/A |
| 687818610 | 2025-12-29 11:05:26 CST | python3 | Accepted | 717 ms | 17 MB |

### 未通过提交代码
#### 提交 687818550 · Runtime Error · 2025-12-29 11:05:14 CST · python3

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        res = 0
        used = [False] * (n + 1)
        def backtrack(pos):
            if pos == n + 1:
                res += 1
                return
            for x in range(1, n + 1):
                if used[x]:
                    continue
                if x % pos != 0 and pos % x != 0:
                    continue
                used[x] = True
                backtrack(pos + 1)
                used[x] = False
        backtrack(1)
        return res
```


## 不同路径 III (`unique-paths-iii`)

- 题目链接：https://leetcode.cn/problems/unique-paths-iii/
- 难度：Hard
- 标签：位运算, 数组, 回溯, 矩阵
- 总提交次数：1
- 最近提交时间：2025-12-29 10:42:39 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-29 10:19:00 CST

```markdown
这题用 DFS 回溯做。

先遍历网格，找到起点的位置，并统计所有需要经过的非障碍格子数量 total（包括起点和终点）。

从起点开始 DFS，每走到一个格子就标记为已访问，并让 remain--。

如果走到终点且 remain == 0，说明刚好走完所有非障碍格子，是一条合法路径，计数加一。

四个方向递归搜索，走完后要回溯，把格子恢复成未访问状态。

DFS 树的高度是格子数 K，分支因子最多 3 左右，时间复杂度在最坏情况下接近 O(3^K)，空间复杂度是 O(K) 的递归栈。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687798042 | 2025-12-29 10:42:39 CST | python3 | Accepted | 18 ms | 17 MB |

### 未通过提交代码
(所有提交均已通过)

## 连续差相同的数字 (`numbers-with-same-consecutive-differences`)

- 题目链接：https://leetcode.cn/problems/numbers-with-same-consecutive-differences/
- 难度：Medium
- 标签：广度优先搜索, 回溯
- 总提交次数：2
- 最近提交时间：2025-12-28 14:42:05 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687646310 | 2025-12-28 14:41:59 CST | python3 | Runtime Error | N/A | N/A |
| 687646331 | 2025-12-28 14:42:05 CST | python3 | Accepted | 8 ms | 17.6 MB |

### 未通过提交代码
#### 提交 687646310 · Runtime Error · 2025-12-28 14:41:59 CST · python3

```python
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        res = []
        path_num = 0  # 当前路径组成的数字
        pos = 0  # 当前已经构造了多少位
        def backtrack():
            nonlocal pos, path_num
            if pos == n:
                res.append(path_num)
                return
            for x in range(10):
                if pos == 0 and x = 0:
                    continue
                if pos > 0 and abs(x - path_num % 10) != k:
                    continue
                path_num = path_num * 10 + x
                pos += 1
                backtrack()
                pos -= 1
                path_num //= 10
        backtrack()
        return res
```


## 飞地的数量 (`number-of-enclaves`)

- 题目链接：https://leetcode.cn/problems/number-of-enclaves/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 并查集, 数组, 矩阵
- 总提交次数：3
- 最近提交时间：2025-12-28 12:16:49 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687623822 | 2025-12-28 12:15:43 CST | python3 | Accepted | 131 ms | 28.2 MB |
| 687623856 | 2025-12-28 12:15:59 CST | python3 | Accepted | 131 ms | 28 MB |
| 687623996 | 2025-12-28 12:16:49 CST | python3 | Accepted | 131 ms | 27.9 MB |

### 未通过提交代码
(所有提交均已通过)

## 不同岛屿的数量 (`number-of-distinct-islands`)

- 题目链接：https://leetcode.cn/problems/number-of-distinct-islands/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 并查集, 哈希表, 哈希函数
- 总提交次数：1
- 最近提交时间：2025-12-28 11:52:06 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-28 11:43:41 CST

```markdown
shapes 里每个元素是“某块岛的相对坐标列表（转成 tuple）”，其中列表的第一个元素都是 (0,0)，也就是 DFS 起点（锚点）


先遍历整张网格，遇到一个还没访问的陆地 1，就以它为「锚点」，从这里做一次 DFS，把这一整块岛的所有格子坐标都记录下来，但记录的是相对坐标（行列坐标减去锚点坐标）。
这样同样形状、只是平移位置不同的岛，它们的相对坐标集合是一样的。
每块岛 DFS 完后，把这块岛的相对坐标列表转成元组塞到一个 set 里，最后 set 的大小就是不同岛屿形状的数量。
时间复杂度 O(mn)，空间复杂度 O(mn)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687619700 | 2025-12-28 11:52:06 CST | python3 | Accepted | 27 ms | 17.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 统计子岛屿 (`count-sub-islands`)

- 题目链接：https://leetcode.cn/problems/count-sub-islands/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 并查集, 数组, 矩阵
- 总提交次数：1
- 最近提交时间：2025-12-28 10:47:59 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-28 10:05:46 CST

```markdown
这道题的关键在于，如何快速判断子岛屿？

当岛屿 B 中所有陆地在岛屿 A 中也是陆地的时候，岛屿 B 是岛屿 A 的子岛。
反过来说，如果岛屿 B 中存在一片陆地，在岛屿 A 的对应位置是海水，那么岛屿 B 就不是岛屿 A 的子岛。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687602631 | 2025-12-28 10:47:59 CST | python3 | Accepted | 377 ms | 27.9 MB |

### 未通过提交代码
(所有提交均已通过)

## 岛屿的最大面积 (`max-area-of-island`)

- 题目链接：https://leetcode.cn/problems/max-area-of-island/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 并查集, 数组, 矩阵
- 总提交次数：1
- 最近提交时间：2025-12-28 09:54:33 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687593768 | 2025-12-28 09:54:33 CST | python3 | Accepted | 8 ms | 17.7 MB |

### 未通过提交代码
(所有提交均已通过)

## 统计封闭岛屿的数目 (`number-of-closed-islands`)

- 题目链接：https://leetcode.cn/problems/number-of-closed-islands/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 并查集, 数组, 矩阵
- 总提交次数：1
- 最近提交时间：2025-12-28 08:53:44 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687588717 | 2025-12-28 08:53:44 CST | python3 | Accepted | 15 ms | 17.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 岛屿数量 (`number-of-islands`)

- 题目链接：https://leetcode.cn/problems/number-of-islands/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 并查集, 数组, 矩阵
- 总提交次数：28
- 最近提交时间：2025-12-27 17:01:56 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-18 10:48:29 CST

```markdown
面试口述要点（30秒）
* 把网格当图，'1'为点、四向为边，本质是数连通分量。
* 遍历网格，遇到未访问的'1'，答案+1，并用 BFS/DFS 把它的整块连通区域扩展并标记掉。
* 每格最多访问一次，时间 O(mn)；用迭代 BFS/DFS 避免递归爆栈，原地标记省内存。并查集也行但更复杂，这题搜索更直接。

空间复杂度：O(mn) 最坏（整张表都是陆地时队列/栈可能很大）。通常可接受。

写 bfs 时 while 循环里总是忘记写dq.append((nr, nc))
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 677266625 | 2025-11-11 09:45:14 CST | python3 | Wrong Answer | N/A | N/A |
| 677266928 | 2025-11-11 09:46:47 CST | python3 | Accepted | 244 ms | 19.7 MB |
| 677268480 | 2025-11-11 09:54:02 CST | python3 | Accepted | 318 ms | 19.6 MB |
| 677268516 | 2025-11-11 09:54:13 CST | python3 | Accepted | 254 ms | 19.7 MB |
| 677268530 | 2025-11-11 09:54:20 CST | python3 | Accepted | 244 ms | 19.8 MB |
| 677273186 | 2025-11-11 10:14:33 CST | python3 | Runtime Error | N/A | N/A |
| 677273214 | 2025-11-11 10:14:40 CST | python3 | Runtime Error | N/A | N/A |
| 677273277 | 2025-11-11 10:14:55 CST | python3 | Accepted | 252 ms | 19.7 MB |
| 677323245 | 2025-11-11 14:01:28 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 677323945 | 2025-11-11 14:05:04 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 677324428 | 2025-11-11 14:07:05 CST | python3 | Accepted | 258 ms | 19.6 MB |
| 678258356 | 2025-11-15 14:44:31 CST | python3 | Runtime Error | N/A | N/A |
| 678258404 | 2025-11-15 14:44:48 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 678258934 | 2025-11-15 14:47:34 CST | python3 | Accepted | 246 ms | 19.7 MB |
| 680737405 | 2025-11-26 09:38:07 CST | python3 | Accepted | 271 ms | 19.9 MB |
| 682410854 | 2025-12-03 16:09:54 CST | python3 | Runtime Error | N/A | N/A |
| 682411068 | 2025-12-03 16:10:30 CST | python3 | Wrong Answer | N/A | N/A |
| 682411309 | 2025-12-03 16:11:12 CST | python3 | Wrong Answer | N/A | N/A |
| 682411715 | 2025-12-03 16:12:22 CST | python3 | Wrong Answer | N/A | N/A |
| 682411938 | 2025-12-03 16:13:06 CST | python3 | Accepted | 246 ms | 19.7 MB |
| 685544733 | 2025-12-18 10:46:56 CST | python3 | Wrong Answer | N/A | N/A |
| 685545081 | 2025-12-18 10:48:04 CST | python3 | Accepted | 238 ms | 19.6 MB |
| 685799380 | 2025-12-19 14:11:29 CST | python3 | Runtime Error | N/A | N/A |
| 685799460 | 2025-12-19 14:11:56 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 685799783 | 2025-12-19 14:13:21 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 685799817 | 2025-12-19 14:13:33 CST | python3 | Accepted | 285 ms | 19.3 MB |
| 685799863 | 2025-12-19 14:13:43 CST | python3 | Accepted | 249 ms | 19.5 MB |
| 687498231 | 2025-12-27 17:01:56 CST | python3 | Accepted | 252 ms | 19.3 MB |

### 未通过提交代码
#### 提交 677266625 · Wrong Answer · 2025-11-11 09:45:14 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        count = 0

        def bfs(sr, sc):
            # 入队前先置为 0
            grid[sr][sc] = 0
            dq = collections.deque()
            dq.append((sr, sc))
            while dq:
                r, c = dq.popleft()
                # 四个方向
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, r + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
                        dq.append((nr, nc))
            
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    count += 1
                    bfs(i, j)
        return count
```

#### 提交 677273186 · Runtime Error · 2025-11-11 10:14:33 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        count = 0

        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque()
            dq.append((sr, sc))
            while dq:
                r, c = dq.popleft()
                # 四个方向
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
                        dq.append((nr, nc))
        
        for i in rows:
            for j in cols:
                if grid[i]grid[j] == '1':
                    count += 1
                    bfs(i, j)
        
        return count
```

#### 提交 677273214 · Runtime Error · 2025-11-11 10:14:40 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        count = 0

        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque()
            dq.append((sr, sc))
            while dq:
                r, c = dq.popleft()
                # 四个方向
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
                        dq.append((nr, nc))
        
        for i in rows:
            for j in cols:
                if grid[i][j] == '1':
                    count += 1
                    bfs(i, j)
        
        return count
```

#### 提交 677323245 · Memory Limit Exceeded · 2025-11-11 14:01:28 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        counts = 0
        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque()
            dq.append((sr, sc))
            while dq:
                r, c = dq.popleft()
                # 四个方向
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        grid[nr][nc] == '0'
                        dq.append((nr, nc))
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    counts += 1
                    bfs(i, j)
        
        return counts
```

#### 提交 677323945 · Memory Limit Exceeded · 2025-11-11 14:05:04 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        counts = 0
        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque()
            dq.append((sr, sc))
            while dq:
                r, c = dq.popleft()
                # 四个方向
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        grid[nr][nc] == '0'
                        dq.append((nr, nc))
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    counts += 1
                    bfs(i, j)
        
        return counts
```

#### 提交 678258356 · Runtime Error · 2025-11-15 14:44:31 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        count = 0
        def dfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque()
            dq.append((sr, sc))
            while dq:
                r, c = dq.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        grid[nr][nc] == '0'
                        dq.append((nr, nc))
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    count += 1
                    bfs(i, j)
        return count
```

#### 提交 678258404 · Memory Limit Exceeded · 2025-11-15 14:44:48 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        count = 0
        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque()
            dq.append((sr, sc))
            while dq:
                r, c = dq.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        grid[nr][nc] == '0'
                        dq.append((nr, nc))
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    count += 1
                    bfs(i, j)
        return count
```

#### 提交 682410854 · Runtime Error · 2025-12-03 16:09:54 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        def bfs(sr, sc):
            grid[sr][sc] = 0
            dq = collections.deque((sr, sc))
            while dq:
                r, c = dq.popleft()
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 < nc < n and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    res += 1
                    bfs(i, j)
        
        return res
```

#### 提交 682411068 · Wrong Answer · 2025-12-03 16:10:30 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        def bfs(sr, sc):
            grid[sr][sc] = 0
            dq = collections.deque([(sr, sc)])
            while dq:
                r, c = dq.popleft()
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 < nc < n and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    res += 1
                    bfs(i, j)
        
        return res
```

#### 提交 682411309 · Wrong Answer · 2025-12-03 16:11:12 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        def bfs(sr, sc):
            grid[sr][sc] = 0
            dq = collections.deque([(sr, sc)])
            while dq:
                r, c = dq.popleft()
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 < nc < n and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
                        dq.append((nr, nc))
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    res += 1
                    bfs(i, j)
        
        return res
```

#### 提交 682411715 · Wrong Answer · 2025-12-03 16:12:22 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque([(sr, sc)])
            while dq:
                r, c = dq.popleft()
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 < nc < n and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
                        dq.append((nr, nc))
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    res += 1
                    bfs(i, j)
        
        return res
```

#### 提交 685544733 · Wrong Answer · 2025-12-18 10:46:56 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque([(sr, sc)])
            while dq:
                r, c = dq.popleft()
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    res += 1
                    bfs(i, j)
        
        return res
```

#### 提交 685799380 · Runtime Error · 2025-12-19 14:11:29 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque((sr, sc))
            while dq:
                r, c = dq.popleft()
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '1':
                        grid[nr][nc] == '0'
                        dq.append((nr, nc))
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    bfs(i, j)
                    res += 1
        
        return res
```

#### 提交 685799460 · Memory Limit Exceeded · 2025-12-19 14:11:56 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque([(sr, sc)])
            while dq:
                r, c = dq.popleft()
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '1':
                        grid[nr][nc] == '0'
                        dq.append((nr, nc))
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    bfs(i, j)
                    res += 1
        
        return res
```

#### 提交 685799783 · Memory Limit Exceeded · 2025-12-19 14:13:21 CST · python3

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        def bfs(sr, sc):
            grid[sr][sc] = '0'
            dq = collections.deque([(sr, sc)])
            while dq:
                r, c = dq.popleft()
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '1':
                        grid[nr][nc] == '0'
                        dq.append((nr, nc))
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    res += 1
                    bfs(i, j)
                    
        
        return res
```


## 组合总和 III (`combination-sum-iii`)

- 题目链接：https://leetcode.cn/problems/combination-sum-iii/
- 难度：Medium
- 标签：数组, 回溯
- 总提交次数：2
- 最近提交时间：2025-12-27 16:14:37 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687486746 | 2025-12-27 16:14:27 CST | python3 | Runtime Error | N/A | N/A |
| 687486787 | 2025-12-27 16:14:37 CST | python3 | Accepted | 0 ms | 16.9 MB |

### 未通过提交代码
#### 提交 687486746 · Runtime Error · 2025-12-27 16:14:27 CST · python3

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        path = []
        self.path_sum = 0
        res = []
        def backtrack(start_num):
            if self.path_sum == n and len(path) == k:
                res.append(list(path))
            if self.path_sum > n or len(path) > k:
                return
            for num in ranges(start_num, 10):
                path.append(num)
                self.path_sum += num
                backtrack(num+1)
                path.pop()
                self.path_sum -= num
        backtrack(1)
        return res
```


## 组合总和 (`combination-sum`)

- 题目链接：https://leetcode.cn/problems/combination-sum/
- 难度：Medium
- 标签：数组, 回溯
- 总提交次数：1
- 最近提交时间：2025-12-27 16:03:15 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687484075 | 2025-12-27 16:03:15 CST | python3 | Accepted | 11 ms | 17.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 全排列 II (`permutations-ii`)

- 题目链接：https://leetcode.cn/problems/permutations-ii/
- 难度：Medium
- 标签：数组, 回溯, 排序
- 总提交次数：11
- 最近提交时间：2025-12-27 13:34:22 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-27 13:16:20 CST

```markdown
剪枝逻辑：同一层的相同数字，只让第一个出现在分支里，后面的相同数字在这一层一律跳过。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680482420 | 2025-11-25 10:11:14 CST | python3 | Wrong Answer | N/A | N/A |
| 680482598 | 2025-11-25 10:11:53 CST | python3 | Wrong Answer | N/A | N/A |
| 680482851 | 2025-11-25 10:12:44 CST | python3 | Wrong Answer | N/A | N/A |
| 680483215 | 2025-11-25 10:14:08 CST | python3 | Accepted | 3 ms | 17.9 MB |
| 680523107 | 2025-11-25 12:57:08 CST | python3 | Accepted | 7 ms | 17.9 MB |
| 682997152 | 2025-12-06 11:51:31 CST | python3 | Wrong Answer | N/A | N/A |
| 682997258 | 2025-12-06 11:52:19 CST | python3 | Wrong Answer | N/A | N/A |
| 682997647 | 2025-12-06 11:55:04 CST | python3 | Accepted | 3 ms | 17.7 MB |
| 687450374 | 2025-12-27 13:22:02 CST | python3 | Accepted | 3 ms | 17.6 MB |
| 687450839 | 2025-12-27 13:25:47 CST | python3 | Accepted | 15 ms | 17.5 MB |
| 687451921 | 2025-12-27 13:34:22 CST | python3 | Accepted | 3 ms | 17.8 MB |

### 未通过提交代码
#### 提交 680482420 · Wrong Answer · 2025-11-25 10:11:14 CST · python3

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            for i in range(len(nums)):
                if len(path) == len(nums):
                    res.append(path[:])
                    return
                # 可能有重复，所以需要多一个剪枝条件
                if i > 0 and nums[i] == nums[i-1] and not nums[i-1]:
                    continue
                
                # 做出选择
                path.append(nums[i])
                used[i] = True

                # 开始进入下一层
                backtrack()

                # 撤销选择，状态重置
                path.pop()
                used[i] = False
        backtrack()
        return res
```

#### 提交 680482598 · Wrong Answer · 2025-11-25 10:11:53 CST · python3

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            for i in range(len(nums)):
                if len(path) == len(nums):
                    res.append(path[:])
                    return
                # 可能有重复，所以需要多一个剪枝条件
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                
                # 做出选择
                path.append(nums[i])
                used[i] = True

                # 开始进入下一层
                backtrack()

                # 撤销选择，状态重置
                path.pop()
                used[i] = False
        backtrack()
        return res
```

#### 提交 680482851 · Wrong Answer · 2025-11-25 10:12:44 CST · python3

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            if len(path) == len(nums):
                    res.append(path[:])
                    return
            for i in range(len(nums)):
                # 可能有重复，所以需要多一个剪枝条件
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                
                # 做出选择
                path.append(nums[i])
                used[i] = True

                # 开始进入下一层
                backtrack()

                # 撤销选择，状态重置
                path.pop()
                used[i] = False
        backtrack()
        return res
```

#### 提交 682997152 · Wrong Answer · 2025-12-06 11:51:31 CST · python3

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        n = len(nums)
        used = [False] * n
        def backtrack():
            if len(path) == n:
                res.append(list(path))
                return
            for i in range(n):
                if used[i]:
                    continue
                if i > 0 and nums[i] == nums[i - 1] and not nums[i - 1]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack()
                path.pop()
                used[i] = False
        backtrack()
        return res
```

#### 提交 682997258 · Wrong Answer · 2025-12-06 11:52:19 CST · python3

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        path = []
        n = len(nums)
        used = [False] * n
        def backtrack():
            if len(path) == n:
                res.append(list(path))
                return
            for i in range(n):
                if used[i]:
                    continue
                if i > 0 and nums[i] == nums[i - 1] and not nums[i - 1]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack()
                path.pop()
                used[i] = False
        backtrack()
        return res
```


## 子集 II (`subsets-ii`)

- 题目链接：https://leetcode.cn/problems/subsets-ii/
- 难度：Medium
- 标签：位运算, 数组, 回溯
- 总提交次数：6
- 最近提交时间：2025-12-26 21:35:28 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-08 14:59:44 CST

```markdown
每一层是一圈 for i in range(start_index, n)，这一圈里的 i 就是本层的兄弟节点。

去重的目标是：同一层不要出现两个以相同值开头的分支，所以我写

if i > start_index and nums[i] == nums[i - 1]: continue。

用 start_index 而不是 0，保证只在“当前层的兄弟之间”去重，不会误伤下一层正常的选择。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 683018964 | 2025-12-06 14:38:17 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 683019300 | 2025-12-06 14:39:58 CST | python3 | Accepted | 0 ms | 17.7 MB |
| 683408046 | 2025-12-08 14:48:10 CST | python3 | Wrong Answer | N/A | N/A |
| 683411461 | 2025-12-08 14:59:53 CST | python3 | Accepted | 0 ms | 17.9 MB |
| 687378373 | 2025-12-26 21:34:39 CST | python3 | Runtime Error | N/A | N/A |
| 687378511 | 2025-12-26 21:35:28 CST | python3 | Accepted | 0 ms | 17.2 MB |

### 未通过提交代码
#### 提交 683018964 · Memory Limit Exceeded · 2025-12-06 14:38:17 CST · python3

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        path = []
        n = len(nums)
        def backtrack(start_index):
            res.append(path[:])
            for i in range(n):
                if i > start_index and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 683408046 · Wrong Answer · 2025-12-08 14:48:10 CST · python3

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        path = []
        n = len(nums)
        def backtrack(start_index):
            res.append(list(path))
            for i in range(start_index, n):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res
```

#### 提交 687378373 · Runtime Error · 2025-12-26 21:34:39 CST · python3

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        path = []
        res = []
        n = len(nums)
        def backtrack(start_index, n):
            res.append(list(path))
            for i in range(start_index, n):
                if i > start_index and nums[i] == nums[i-1]:
                    continue
                path.append(nums[i])
                backtrack(i+1)
                path.pop()
        backtrack(0)
        return res
```


## 组合 (`combinations`)

- 题目链接：https://leetcode.cn/problems/combinations/
- 难度：Medium
- 标签：回溯
- 总提交次数：2
- 最近提交时间：2025-12-26 18:03:10 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-26 18:00:24 CST

```markdown
通过 start 参数控制树枝的遍历，避免产生重复的子集
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685312612 | 2025-12-17 10:36:45 CST | python3 | Accepted | 147 ms | 58.7 MB |
| 687343695 | 2025-12-26 18:03:10 CST | python3 | Accepted | 243 ms | 58.3 MB |

### 未通过提交代码
(所有提交均已通过)

## 子集 (`subsets`)

- 题目链接：https://leetcode.cn/problems/subsets/
- 难度：Medium
- 标签：位运算, 数组, 回溯
- 总提交次数：5
- 最近提交时间：2025-12-26 17:12:37 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-26 18:00:45 CST

```markdown
“全排列和子集虽然都是对数组的穷举，但对象不同：全排列要的是有序序列，子集要的是组合，顺序不重要。

对于全排列，我用 used 数组限制每个元素最多用一次，但允许不同顺序，从而得到所有排列。

对于子集，如果也用 used 而不加约束，会把同一组元素的不同排列都算一遍，比如 [1,2] 和 [2,1]，结果重复。所以我改用 start_index，规定下一层递归只能从当前位置之后选元素，这样只会生成按索引递增的组合，自然不会出现顺序不同但元素相同的重复子集。”

这道题没有显式的结束条件，因为每一个节点（路径）都是一个合法的子集

每个数都有选和不选两种情况
“选”： 体现在 for 循环内部的 path.append(nums[i]) 和随后的递归调用 backtrack(i + 1)。这代表：“我确定要 nums[i] 了，请帮我处理剩下的元素。”
“不选”： 体现在 for 循环的推进本身。当 i 从 0 变成 1 时，就意味着我们结束了所有包含 nums[0] 的可能性，接下来探索的所有路径，都天然地处于“不选 nums[0]”的那个分支上。

通过保证元素之间的相对顺序不变来防止出现重复的子集。

通过 start 参数控制树枝的遍历，避免产生重复的子集
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682597940 | 2025-12-04 12:35:50 CST | python3 | Accepted | 0 ms | 17.8 MB |
| 683016526 | 2025-12-06 14:25:08 CST | python3 | Wrong Answer | N/A | N/A |
| 683016774 | 2025-12-06 14:26:36 CST | python3 | Accepted | 0 ms | 17.8 MB |
| 683405933 | 2025-12-08 14:40:16 CST | python3 | Accepted | 0 ms | 17.8 MB |
| 687332047 | 2025-12-26 17:12:37 CST | python3 | Accepted | 3 ms | 17.4 MB |

### 未通过提交代码
#### 提交 683016526 · Wrong Answer · 2025-12-06 14:25:08 CST · python3

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        n = len(nums)
        def backtrack(start_index):
            res.append(list(path))
            for i in range(start_index, n):
                path.append(nums[start_index])
                backtrack(start_index + 1)
                path.pop()
        backtrack(0)
        return res
```


## N 皇后 II (`n-queens-ii`)

- 题目链接：https://leetcode.cn/problems/n-queens-ii/
- 难度：Hard
- 标签：回溯
- 总提交次数：2
- 最近提交时间：2025-12-26 16:29:16 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687316639 | 2025-12-26 16:28:47 CST | python3 | Runtime Error | N/A | N/A |
| 687316779 | 2025-12-26 16:29:16 CST | python3 | Accepted | 11 ms | 17.1 MB |

### 未通过提交代码
#### 提交 687316639 · Runtime Error · 2025-12-26 16:28:47 CST · python3

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        res = 0
        cols = set()
        diag_1 = set()
        diag_2 = set()
        def backtrack(row):
            if row == n:
                res += 1
            for col in range(n):
                if col in cols or (row - col) in diag_1 or (row + col) in diag_2:
                    continue
                cols.add(col)
                diag_1.add(row - col)
                diag_2.add(row + col)
                backtrack(row + 1)
                cols.remove(col)
                diag_1.remove(row - col)
                diag_2.remove(row + col)
        backtrack(0)
        return res
```


## N 皇后 (`n-queens`)

- 题目链接：https://leetcode.cn/problems/n-queens/
- 难度：Hard
- 标签：数组, 回溯
- 总提交次数：3
- 最近提交时间：2025-12-26 16:21:28 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-07 14:23:29 CST

```markdown
“这道题我使用回溯算法（Backtracking）来解决。
核心思路是逐行放置皇后。我维护了三个哈希集合（Set），分别记录当前已被占用的列、主对角线（利用 row - col 差值唯一）和副对角线（利用 row + col 和值唯一）。
递归过程中，如果当前位置不冲突，我就放置皇后并进入下一行；如果下一行无法放置，或者已经找到解，我就回退（撤销）刚才的操作，尝试当前行的下一列。这样能遍历所有可行解。”


时间复杂度： O(N!)。第一行有 N 种选法，第二行 N-1 种... 虽然有剪枝，但最坏情况接近阶乘级。
空间复杂度： O(N)。主要是递归栈的深度和三个集合的空间，都是 N 级别的。

不变式：在执行 backtrack(row) 函数体里的代码时，cols/diag1/diag2 里 只包含 0~row-1 行的皇后信息。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681661207 | 2025-11-30 14:53:50 CST | python3 | Accepted | 11 ms | 17.9 MB |
| 683205973 | 2025-12-07 14:43:07 CST | python3 | Accepted | 7 ms | 18 MB |
| 687313874 | 2025-12-26 16:21:28 CST | python3 | Accepted | 11 ms | 17.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 解数独 (`sudoku-solver`)

- 题目链接：https://leetcode.cn/problems/sudoku-solver/
- 难度：Hard
- 标签：数组, 哈希表, 回溯, 矩阵
- 总提交次数：7
- 最近提交时间：2025-12-26 15:27:15 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687283900 | 2025-12-26 14:45:37 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 687294696 | 2025-12-26 15:22:26 CST | python3 | Runtime Error | N/A | N/A |
| 687294760 | 2025-12-26 15:22:41 CST | python3 | Runtime Error | N/A | N/A |
| 687294853 | 2025-12-26 15:23:01 CST | python3 | Runtime Error | N/A | N/A |
| 687295288 | 2025-12-26 15:24:32 CST | python3 | Runtime Error | N/A | N/A |
| 687295843 | 2025-12-26 15:26:24 CST | python3 | Wrong Answer | N/A | N/A |
| 687296096 | 2025-12-26 15:27:15 CST | python3 | Accepted | 1435 ms | 17.2 MB |

### 未通过提交代码
#### 提交 687283900 · Time Limit Exceeded · 2025-12-26 14:45:37 CST · python3

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def is_valid(r, c, num):
            for i in range(9):
                if board[r][i] == num: return False
                if board[i][c] == num: return False
            
            start_r, start_c = (r // 3) * 3, (c // 3) * 3
            for i in range(3):
                for j in range(3):
                    if board[start_r + i][start_c + j] == num:
                        return False
            return True 


        def backtrack(r, c):
            if c == 9:
                return backtrack(r + 1, 0)
            if r == 9:
                return True
            if board[r][c] != '.':
                return backtrack(r, c + 1)
            
            for num in map(str, range(1, 10)):
                if is_valid(r, c, num):
                    board[r][c] = num
                    if backtrack(r, c + 1):
                        return True
                    board[r][c] = '.'
            return False
        
        backtrack(0, 0)
```

#### 提交 687294696 · Runtime Error · 2025-12-26 15:22:26 CST · python3

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = [[False] * 10 for _ in range(9)]
        col = [[False] * 10 for _ in range(9)]
        block = [[[False] * 10 for _ in range(3)] for _ in range(3)]
        for i in in range(9):
            for j in range(9):
                num = int(board[i][j])
                row[i][num] = True
                col[j][num] = True
                block[i // 3][j // 3][num] = True
        
        def backtrack(r, c):
            if c == 9:
                return backtrack(r + 1, 0)
            if r == 9:
                return True
            if board[r][c] != '.':
                return backtrack(r, c + 1)
            for num in map(str, range(1, 9)):
                b_r = r // 3, b_c = c // 3
                if row[r][num] or col[c][num] or block[b_r][b_c][num]:
                    continue
                row[r][num] = True
                col[c][num] = True
                block[b_r][b_c][num] = True
                board[r][c] = num
                if backtrack(r, c + 1):
                    return True
                board[r][c] = '.'
                row[r][num] = False
                col[c][num] = False
                block[b_r][b_c][num] = False
            return False
        backtrack(0, 0)
```

#### 提交 687294760 · Runtime Error · 2025-12-26 15:22:41 CST · python3

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = [[False] * 10 for _ in range(9)]
        col = [[False] * 10 for _ in range(9)]
        block = [[[False] * 10 for _ in range(3)] for _ in range(3)]
        for i in range(9):
            for j in range(9):
                num = int(board[i][j])
                row[i][num] = True
                col[j][num] = True
                block[i // 3][j // 3][num] = True
        
        def backtrack(r, c):
            if c == 9:
                return backtrack(r + 1, 0)
            if r == 9:
                return True
            if board[r][c] != '.':
                return backtrack(r, c + 1)
            for num in map(str, range(1, 9)):
                b_r = r // 3, b_c = c // 3
                if row[r][num] or col[c][num] or block[b_r][b_c][num]:
                    continue
                row[r][num] = True
                col[c][num] = True
                block[b_r][b_c][num] = True
                board[r][c] = num
                if backtrack(r, c + 1):
                    return True
                board[r][c] = '.'
                row[r][num] = False
                col[c][num] = False
                block[b_r][b_c][num] = False
            return False
        backtrack(0, 0)
```

#### 提交 687294853 · Runtime Error · 2025-12-26 15:23:01 CST · python3

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = [[False] * 10 for _ in range(9)]
        col = [[False] * 10 for _ in range(9)]
        block = [[[False] * 10 for _ in range(3)] for _ in range(3)]
        for i in range(9):
            for j in range(9):
                num = int(board[i][j])
                row[i][num] = True
                col[j][num] = True
                block[i // 3][j // 3][num] = True
        
        def backtrack(r, c):
            if c == 9:
                return backtrack(r + 1, 0)
            if r == 9:
                return True
            if board[r][c] != '.':
                return backtrack(r, c + 1)
            for num in map(str, range(1, 9)):
                b_r, b_c = r // 3, c // 3
                if row[r][num] or col[c][num] or block[b_r][b_c][num]:
                    continue
                row[r][num] = True
                col[c][num] = True
                block[b_r][b_c][num] = True
                board[r][c] = num
                if backtrack(r, c + 1):
                    return True
                board[r][c] = '.'
                row[r][num] = False
                col[c][num] = False
                block[b_r][b_c][num] = False
            return False
        backtrack(0, 0)
```

#### 提交 687295288 · Runtime Error · 2025-12-26 15:24:32 CST · python3

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = [[False] * 10 for _ in range(9)]
        col = [[False] * 10 for _ in range(9)]
        block = [[[False] * 10 for _ in range(3)] for _ in range(3)]
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = int(board[i][j])
                    row[i][num] = True
                    col[j][num] = True
                    block[i // 3][j // 3][num] = True
        
        def backtrack(r, c):
            if c == 9:
                return backtrack(r + 1, 0)
            if r == 9:
                return True
            if board[r][c] != '.':
                return backtrack(r, c + 1)
            for num in map(str, range(1, 9)):
                b_r, b_c = r // 3, c // 3
                if row[r][num] or col[c][num] or block[b_r][b_c][num]:
                    continue
                row[r][num] = True
                col[c][num] = True
                block[b_r][b_c][num] = True
                board[r][c] = num
                if backtrack(r, c + 1):
                    return True
                board[r][c] = '.'
                row[r][num] = False
                col[c][num] = False
                block[b_r][b_c][num] = False
            return False
        backtrack(0, 0)
```

#### 提交 687295843 · Wrong Answer · 2025-12-26 15:26:24 CST · python3

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = [[False] * 10 for _ in range(9)]
        col = [[False] * 10 for _ in range(9)]
        block = [[[False] * 10 for _ in range(3)] for _ in range(3)]
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = int(board[i][j])
                    row[i][num] = True
                    col[j][num] = True
                    block[i // 3][j // 3][num] = True
        
        def backtrack(r, c):
            if c == 9:
                return backtrack(r + 1, 0)
            if r == 9:
                return True
            if board[r][c] != '.':
                return backtrack(r, c + 1)
            for num in range(1, 9):
                b_r, b_c = r // 3, c // 3
                if row[r][num] or col[c][num] or block[b_r][b_c][num]:
                    continue
                row[r][num] = True
                col[c][num] = True
                block[b_r][b_c][num] = True
                board[r][c] = str(num)
                if backtrack(r, c + 1):
                    return True
                board[r][c] = '.'
                row[r][num] = False
                col[c][num] = False
                block[b_r][b_c][num] = False
            return False
        backtrack(0, 0)
```


## 全排列 (`permutations`)

- 题目链接：https://leetcode.cn/problems/permutations/
- 难度：Medium
- 标签：数组, 回溯
- 总提交次数：13
- 最近提交时间：2025-12-26 11:37:39 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-25 09:50:53 CST

```markdown
时间复杂度是 O(n * n!)，因为一共有 n! 个排列，每个排列长度为 n
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680476254 | 2025-11-25 09:45:48 CST | python3 | Runtime Error | N/A | N/A |
| 680476310 | 2025-11-25 09:45:59 CST | python3 | Runtime Error | N/A | N/A |
| 680476444 | 2025-11-25 09:46:36 CST | python3 | Wrong Answer | N/A | N/A |
| 680476618 | 2025-11-25 09:47:27 CST | python3 | Wrong Answer | N/A | N/A |
| 680476665 | 2025-11-25 09:47:38 CST | python3 | Accepted | 0 ms | 17.8 MB |
| 681559647 | 2025-11-29 21:59:50 CST | python3 | Accepted | 3 ms | 18 MB |
| 681605280 | 2025-11-30 10:31:13 CST | python3 | Wrong Answer | N/A | N/A |
| 681605399 | 2025-11-30 10:31:53 CST | python3 | Accepted | 3 ms | 17.7 MB |
| 681813450 | 2025-12-01 10:15:34 CST | python3 | Wrong Answer | N/A | N/A |
| 681813765 | 2025-12-01 10:16:51 CST | python3 | Accepted | 0 ms | 17.9 MB |
| 682994957 | 2025-12-06 11:37:23 CST | python3 | Wrong Answer | N/A | N/A |
| 682995113 | 2025-12-06 11:38:18 CST | python3 | Accepted | 3 ms | 17.7 MB |
| 687251533 | 2025-12-26 11:37:39 CST | python3 | Accepted | 4 ms | 17.2 MB |

### 未通过提交代码
#### 提交 680476254 · Runtime Error · 2025-11-25 09:45:48 CST · python3

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        # 记录路径
        path = []
        # 用于剪枝
        used = []
        def backtrack():
            # 终止条件
            if len(path) == len(nums):
                res.append(path[:])
                return
            # 开始遍历
            for i in range(len(nums):
                if used[i]:
                    continue
                # 做选择
                path.append(nums[i])
                nums[i] = True

                # 进入下一层决策树
                backtrack()

                # 撤销选择，状态重置
                path.pop()
                nums[i] = False
        backtrack()
        return res
```

#### 提交 680476310 · Runtime Error · 2025-11-25 09:45:59 CST · python3

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        # 记录路径
        path = []
        # 用于剪枝
        used = []
        def backtrack():
            # 终止条件
            if len(path) == len(nums):
                res.append(path[:])
                return
            # 开始遍历
            for i in range(len(nums)):
                if used[i]:
                    continue
                # 做选择
                path.append(nums[i])
                nums[i] = True

                # 进入下一层决策树
                backtrack()

                # 撤销选择，状态重置
                path.pop()
                nums[i] = False
        backtrack()
        return res
```

#### 提交 680476444 · Wrong Answer · 2025-11-25 09:46:36 CST · python3

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        # 记录路径
        path = []
        # 用于剪枝
        used = [False] * len(nums)
        def backtrack():
            # 终止条件
            if len(path) == len(nums):
                res.append(path[:])
                return
            # 开始遍历
            for i in range(len(nums)):
                if used[i]:
                    continue
                # 做选择
                path.append(nums[i])
                nums[i] = True

                # 进入下一层决策树
                backtrack()

                # 撤销选择，状态重置
                path.pop()
                nums[i] = False
        backtrack()
        return res
```

#### 提交 680476618 · Wrong Answer · 2025-11-25 09:47:27 CST · python3

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        # 记录路径
        path = []
        # 用于剪枝
        used = [False] * len(nums)
        def backtrack():
            # 终止条件
            if len(path) == len(nums):
                res.append(path[:])
                return
            # 开始遍历
            for i in range(len(nums)):
                if used[i]:
                    continue
                # 做选择
                path.append(nums[i])
                used[i] = True

                # 进入下一层决策树
                backtrack()

                # 撤销选择，状态重置
                path.pop()
                nums[i] = False
        backtrack()
        return res
```

#### 提交 681605280 · Wrong Answer · 2025-11-30 10:31:13 CST · python3

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
                return

            for i in range(len(nums)):
                if used[i]:
                    continue
                path.append(nums[i])
                backtrack()
                path.pop()
                
        backtrack()

        return res
```

#### 提交 681813450 · Wrong Answer · 2025-12-01 10:15:34 CST · python3

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        used = [False] * len(nums)
        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if used[i]:
                    continue
                path.append(nums[i])
                used[i] == True
                backtrack()
                path.pop()
                used[i] == False
        backtrack()
        return res
```

#### 提交 682994957 · Wrong Answer · 2025-12-06 11:37:23 CST · python3

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        n = len(nums)
        used = [False] * n
        def backtrack():
            if len(path) == n:
                res.append(list(path))
                return
            for i in range(n):
                if used[i]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack()
                path.pop()
                used[i] = False
        return res
```


## 二叉树的序列化与反序列化 (`serialize-and-deserialize-binary-tree`)

- 题目链接：https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/
- 难度：Hard
- 标签：树, 深度优先搜索, 广度优先搜索, 设计, 字符串, 二叉树
- 总提交次数：2
- 最近提交时间：2025-12-26 10:58:06 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687240877 | 2025-12-26 10:57:32 CST | python3 | Wrong Answer | N/A | N/A |
| 687241037 | 2025-12-26 10:58:06 CST | python3 | Accepted | 88 ms | 21.6 MB |

### 未通过提交代码
#### 提交 687240877 · Wrong Answer · 2025-12-26 10:57:32 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return "null"
        left_str = self.serialize(root.left)
        right_str = self.serialize(root.right)
        return f"{root.val},{left_str},{right_str}"
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None
        chars = iter(data.split(','))
        def _build():
            val = next(chars)
            if val == 'null':
                return None
            node = TreeNode(val)
            node.left = _build()
            node.right = _build()
        return _build()       

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
```


## 寻找重复的子树 (`find-duplicate-subtrees`)

- 题目链接：https://leetcode.cn/problems/find-duplicate-subtrees/
- 难度：Medium
- 标签：树, 深度优先搜索, 哈希表, 二叉树
- 总提交次数：4
- 最近提交时间：2025-12-26 10:04:20 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-26 09:49:48 CST

```markdown
核心问题：如何标识一棵子树
这道题用“分解问题”的思维模式最自然，因为它完美契合了“后序遍历”的特点：先处理完左右子树，再处理根节点。

为了判断子树是否重复，我需要给每棵子树一个唯一的签名。我用后序遍历的方式，将每棵子树序列化成一个包含空节点的字符串。同时，用一个哈希表来统计每种序列化字符串出现的次数。当某个字符串的计数值从 1 变成 2 时，就说明我找到了一个新的重复子树类型，把当前节点加入结果列表。这个过程每个节点只访问一次，时间复杂度是 O(N)，空间也是 O(N) 用来存哈希表和递归栈。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687225920 | 2025-12-26 09:56:33 CST | python3 | Runtime Error | N/A | N/A |
| 687226006 | 2025-12-26 09:57:00 CST | python3 | Accepted | 55 ms | 17.6 MB |
| 687226782 | 2025-12-26 10:00:37 CST | python3 | Wrong Answer | N/A | N/A |
| 687227531 | 2025-12-26 10:04:20 CST | python3 | Accepted | 99 ms | 17.9 MB |

### 未通过提交代码
#### 提交 687225920 · Runtime Error · 2025-12-26 09:56:33 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        res = []
        subtrees_count = {}
        def _serialize(node):
            if not node:
                return None
            left_key = _serialize(node.left)
            right_key = _serialize(node.right)
            curr_key = (node.val, left_key, right_key)
            count = subtrees_count.get(curr_key, 0)
            if count == 1:
                res.append(node)
            subtrees_count[curr_key] = count + 1
            return curr_key
        return _serialize(root)
```

#### 提交 687226782 · Wrong Answer · 2025-12-26 10:00:37 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        res = []
        subtree_count = collections.defaultdict(int)
        def _serialize(node):
            if not node:
                return None
            left_key = _serialize(node.left)
            right_key = _serialize(node.right)
            curr_key = (node.val, left_key, right_key)
            if subtree_count.get(curr_key) == 1:
                res.append(node)
            subtree_count[node] += 1
            return curr_key
        _serialize(root)
        return res
```


## 根据前序和后序遍历构造二叉树 (`construct-binary-tree-from-preorder-and-postorder-traversal`)

- 题目链接：https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/
- 难度：Medium
- 标签：树, 数组, 哈希表, 分治, 二叉树
- 总提交次数：8
- 最近提交时间：2025-12-25 17:00:14 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-25 16:13:57 CST

```markdown
对一个子树来说，前序的第一个是根，第二个一定是左子树的根；我们在后序里找到这个左子树根的位置，就能知道左子树有多少个节点。这样就可以在前序和后序中同时切出左子树和右子树对应的区间，递归构造左右子树。为了 O(1) 找左子树根在后序中的位置，我建了一个值到下标的哈希表。整体时间 O(n)，空间是递归栈 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687095451 | 2025-12-25 16:25:30 CST | python3 | Runtime Error | N/A | N/A |
| 687095515 | 2025-12-25 16:25:41 CST | python3 | Runtime Error | N/A | N/A |
| 687096251 | 2025-12-25 16:27:55 CST | python3 | Runtime Error | N/A | N/A |
| 687097106 | 2025-12-25 16:30:36 CST | python3 | Runtime Error | N/A | N/A |
| 687097996 | 2025-12-25 16:33:31 CST | python3 | Runtime Error | N/A | N/A |
| 687098153 | 2025-12-25 16:33:58 CST | python3 | Runtime Error | N/A | N/A |
| 687098281 | 2025-12-25 16:34:21 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 687107324 | 2025-12-25 17:00:14 CST | python3 | Accepted | 4 ms | 17.2 MB |

### 未通过提交代码
#### 提交 687095451 · Runtime Error · 2025-12-25 16:25:30 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        n = len(preorder)
        index_map = {val : i for i, val in enumerate(postorder)}
        def _build(pre_start, pre_end, post_start, post_end):
            if pre_start >= pre_end or post_start >= post_end:
                return
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            left_root_val = preorder[post_start + 1]
            left_root_index = postorder.get(left_root_val)
            left_size = left_root_index - post_start + 1
            root.left = _build(pre_start + 1, pre_start + 1 + left_size, post_start, left_root_index)
            root.right = _build(pre_start + left_size, pre_end, left_root_index + 1, post_end - 1)
        return _build(0, n, 0, n)
```

#### 提交 687095515 · Runtime Error · 2025-12-25 16:25:41 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        n = len(preorder)
        index_map = {val : i for i, val in enumerate(postorder)}
        def _build(pre_start, pre_end, post_start, post_end):
            if pre_start >= pre_end or post_start >= post_end:
                return
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            left_root_val = preorder[post_start + 1]
            left_root_index = index_map.get(left_root_val)
            left_size = left_root_index - post_start + 1
            root.left = _build(pre_start + 1, pre_start + 1 + left_size, post_start, left_root_index)
            root.right = _build(pre_start + left_size, pre_end, left_root_index + 1, post_end - 1)
        return _build(0, n, 0, n)
```

#### 提交 687096251 · Runtime Error · 2025-12-25 16:27:55 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        n = len(preorder)
        index_map = {val : i for i, val in enumerate(postorder)}
        def _build(pre_start, pre_end, post_start, post_end):
            if pre_start >= pre_end or post_start >= post_end:
                return
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            left_root_val = preorder[post_start + 1]
            left_root_index = index_map.get(left_root_val)
            left_size = left_root_index - post_start + 1
            root.left = _build(pre_start + 1, pre_start + 1 + left_size, post_start, left_root_index)
            root.right = _build(pre_start + 1 + left_size, pre_end, left_root_index + 1, post_end - 1)
        return _build(0, n, 0, n)
```

#### 提交 687097106 · Runtime Error · 2025-12-25 16:30:36 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        n = len(preorder)
        index_map = {val : i for i, val in enumerate(postorder)}
        def _build(pre_start, pre_end, post_start, post_end):
            if pre_start >= pre_end or post_start >= post_end:
                return
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            left_root_val = preorder[post_start + 1]
            left_root_index = index_map.get(left_root_val)
            left_size = left_root_index - post_start + 1
            root.left = _build(pre_start + 1, pre_start + 1 + left_size, post_start, post_start + left_size)
            root.right = _build(pre_start + 1 + left_size, pre_end, post_start + left_size, post_end - 1)
        return _build(0, n, 0, n)
```

#### 提交 687097996 · Runtime Error · 2025-12-25 16:33:31 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        n = len(preorder)
        index_map = {val : i for i, val in enumerate(postorder)}
        def _build(pre_start, pre_end, post_start, post_end):
            if pre_start >= pre_end or post_start >= post_end:
                return
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            left_root_val = preorder[post_start + 1]
            left_root_index = index_map.get(left_root_val)
            left_size = left_root_index - post_start + 1
            root.left = _build(pre_start + 1, pre_start + 1 + left_size, post_start, post_start + left_size)
            root.right = _build(pre_start + 1 + left_size, pre_end, post_start + left_size, post_end - 1)
            return root
        return _build(0, n, 0, n)
```

#### 提交 687098153 · Runtime Error · 2025-12-25 16:33:58 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        n = len(preorder)
        index_map = {val : i for i, val in enumerate(postorder)}
        def _build(pre_start, pre_end, post_start, post_end):
            if pre_start >= pre_end or post_start >= post_end:
                return
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            left_root_val = preorder[pre_start + 1]
            left_root_index = index_map.get(left_root_val)
            left_size = left_root_index - post_start + 1
            root.left = _build(pre_start + 1, pre_start + 1 + left_size, post_start, post_start + left_size)
            root.right = _build(pre_start + 1 + left_size, pre_end, post_start + left_size, post_end - 1)
            return root
        return _build(0, n, 0, n)
```


## 从中序与后序遍历序列构造二叉树 (`construct-binary-tree-from-inorder-and-postorder-traversal`)

- 题目链接：https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
- 难度：Medium
- 标签：树, 数组, 哈希表, 分治, 二叉树
- 总提交次数：3
- 最近提交时间：2025-12-25 15:59:53 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687086776 | 2025-12-25 15:57:38 CST | python3 | Runtime Error | N/A | N/A |
| 687087026 | 2025-12-25 15:58:28 CST | python3 | Runtime Error | N/A | N/A |
| 687087475 | 2025-12-25 15:59:53 CST | python3 | Accepted | 0 ms | 18.5 MB |

### 未通过提交代码
#### 提交 687086776 · Runtime Error · 2025-12-25 15:57:38 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        n = len(inorder)
        index_map = {val: i for i, val in enumerate(inorder)}
        def _build(in_start, in_end, post_start, post_end):
            if in_start >= in_end or post_start >= post_end:
                return
            root_val = postorder[-1]
            root = TreeNode(root_val)
            root_index = index_map.get(root_val)
            left_size = root_index - in_start
            root.left = _build(in_start, root_index, post_start, post_start + left_size)
            root.right = _build(root_index, in_end, post_start + left_size, post_end - 1)
            return root
        return _build(0, n, 0, n)
```

#### 提交 687087026 · Runtime Error · 2025-12-25 15:58:28 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        n = len(inorder)
        index_map = {val: i for i, val in enumerate(inorder)}
        def _build(in_start, in_end, post_start, post_end):
            if in_start >= in_end or post_start >= post_end:
                return
            root_val = postorder[post_end-1]
            root = TreeNode(root_val)
            root_index = index_map.get(root_val)
            left_size = root_index - in_start
            root.left = _build(in_start, root_index, post_start, post_start + left_size)
            root.right = _build(root_index, in_end, post_start + left_size, post_end - 1)
            return root
        return _build(0, n, 0, n)
```


## 从前序与中序遍历序列构造二叉树 (`construct-binary-tree-from-preorder-and-inorder-traversal`)

- 题目链接：https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
- 难度：Medium
- 标签：树, 数组, 哈希表, 分治, 二叉树
- 总提交次数：1
- 最近提交时间：2025-12-25 15:20:51 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687075251 | 2025-12-25 15:20:51 CST | python3 | Accepted | 3 ms | 18.5 MB |

### 未通过提交代码
(所有提交均已通过)

## 最大二叉树 (`maximum-binary-tree`)

- 题目链接：https://leetcode.cn/problems/maximum-binary-tree/
- 难度：Medium
- 标签：栈, 树, 数组, 分治, 二叉树, 单调栈
- 总提交次数：2
- 最近提交时间：2025-12-25 13:58:58 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-25 14:21:18 CST

```markdown
递归的思路，完全按照题目定义来。定义一个函数，输入一个数组，返回用它建好的最大二叉树。函数里先找到数组的最大值和索引，用最大值创建根节点。然后把数组在最大值处切成两半，用左边的部分递归调用函数，把返回的树挂到根的左子树上；用右边的部分递归调用，挂到右子树上。如果数组为空就返回 None。时间复杂度 O(n^2)，空间是递归栈 O(n)。

每次递归调用都要遍历当前子数组找最大值，假设子数组长度为 k，耗时 O(k)。
在最坏情况下（例如数组是排序好的 [1, 2, 3, 4]），每次最大值都在最右边，树退化成链表，递归 n 次，每次遍历长度 n, n-1, ..., 1，总共是 n + (n-1) + ... + 1 = O(n^2)。


构建笛卡尔树的解法：如果遇到一个比栈顶大的数，那么刚才那个栈顶肯定就是我的左孩子；如果我也比栈顶小，那我肯定就是栈顶的右孩子。

栈里放的，是那些左边已经处理、左子树已经确定，但右边还没看完、右子树结构还没最终定下来的节点；等遇到合适的新节点，我们再给它们“安排右边的归宿”，安排完就出栈。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687044800 | 2025-12-25 12:49:31 CST | python3 | Accepted | 42 ms | 17.7 MB |
| 687054683 | 2025-12-25 13:58:58 CST | python3 | Accepted | 24 ms | 17.5 MB |

### 未通过提交代码
(所有提交均已通过)

## K 个一组翻转链表 (`reverse-nodes-in-k-group`)

- 题目链接：https://leetcode.cn/problems/reverse-nodes-in-k-group/
- 难度：Hard
- 标签：递归, 链表
- 总提交次数：2
- 最近提交时间：2025-12-25 11:21:48 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-25 11:16:45 CST

```markdown
我用一个递归的分段思路。函数含义是：把从 head 开始的链表，每 k 个一组翻转，并返回新的头结点。实现时，先从 head 开始往后数 k 个节点，如果不足 k 个就直接返回 head，不翻转。够 k 个的话，就原地翻转这前 k 个节点，翻转后原来的 head 变成这一段的尾结点，它的 next 指向递归处理剩余部分得到的头结点，最后返回这段翻转后的新头。整体时间复杂度 O(n)，额外空间是递归栈 O(n/k)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684343497 | 2025-12-12 15:01:59 CST | python3 | Accepted | 0 ms | 18.4 MB |
| 687029431 | 2025-12-25 11:21:48 CST | python3 | Accepted | 0 ms | 17.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 二叉树展开为链表 (`flatten-binary-tree-to-linked-list`)

- 题目链接：https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/
- 难度：Medium
- 标签：栈, 树, 深度优先搜索, 链表, 二叉树
- 总提交次数：4
- 最近提交时间：2025-12-25 10:51:27 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-25 10:44:06 CST

```markdown
用指针 cur 从根开始往右走。对每个节点，如果没有左子树就直接走到右子节点。如果有左子树，就在左子树里找到最右的节点 pre，把当前节点原来的右子树挂到 pre.right 上，然后把左子树整体挪到 cur.right，cur.left 置空。这样等价于把“左子树插到右子树前面”，最后整棵树就被改成按前序顺序的右链表。时间复杂度 O(n)，额外空间 O(1)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 687012993 | 2025-12-25 10:23:58 CST | python3 | Runtime Error | N/A | N/A |
| 687013093 | 2025-12-25 10:24:26 CST | python3 | Accepted | 0 ms | 17.2 MB |
| 687019976 | 2025-12-25 10:49:43 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 687020486 | 2025-12-25 10:51:27 CST | python3 | Accepted | 0 ms | 17.1 MB |

### 未通过提交代码
#### 提交 687012993 · Runtime Error · 2025-12-25 10:23:58 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if  not root:
            return None
        self.flatten(root.left)
        self.flatten(root.right)
        left_list = root.left
        right_list = root.right
        root.left = None
        root.right = left_list
        p = root
        while p:
            p = p.right
        p.right = right_list
```

#### 提交 687019976 · Time Limit Exceeded · 2025-12-25 10:49:43 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        curr = root
        while curr is not None:
            if curr.left is not None:
                # 找到左子树最右侧的节点
                pre = curr.left
                while pre.right:
                    pre = pre.right
                # 将原右子树接到 pre 后面
                pre.right = curr.right
                # 将原左子树整个放到curr节点的右面，然后置空curr的左子树
                curr.right = curr.left
                curr.left = None
            curr = root.right
```


## 填充每个节点的下一个右侧节点指针 (`populating-next-right-pointers-in-each-node`)

- 题目链接：https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 链表, 二叉树
- 总提交次数：7
- 最近提交时间：2025-12-25 09:25:37 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-25 09:20:37 CST

```markdown
这题是完美二叉树，所以每个非叶子节点都有两个孩子，下一层是连续排满的。

我用一个指针 level_head 指向当前层最左的节点；在这一层里，用 curr 沿着 next 指针横向遍历。

对每个 curr，先连同一父亲的两个孩子：curr.left.next = curr.right；

如果 curr.next 存在，再连跨父节点的孩子：curr.right.next = curr.next.left。

当前层处理完后，下一层的 next 都已经建好，然后把 level_head 移到 level_head.left 继续。

这样只用几个指针，不用队列，空间 O(1)，时间 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685978989 | 2025-12-20 14:45:16 CST | python3 | Runtime Error | N/A | N/A |
| 685979434 | 2025-12-20 14:47:30 CST | python3 | Accepted | 45 ms | 18.5 MB |
| 685998660 | 2025-12-20 15:59:27 CST | python3 | Accepted | 53 ms | 18.4 MB |
| 686110091 | 2025-12-21 09:32:16 CST | python3 | Accepted | 51 ms | 18.7 MB |
| 686293054 | 2025-12-22 09:16:13 CST | python3 | Accepted | 50 ms | 18.3 MB |
| 686537660 | 2025-12-23 09:12:08 CST | python3 | Accepted | 56 ms | 18.6 MB |
| 687000364 | 2025-12-25 09:25:37 CST | python3 | Accepted | 56 ms | 18.4 MB |

### 未通过提交代码
#### 提交 685978989 · Runtime Error · 2025-12-20 14:45:16 CST · python3

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None
        leftmost = root
        while leftmost:
            head = leftmost
            while head:
                head.left.next = head.right
                if head.next:
                    head.right.next = head.next.left
                head = head.next
            leftmost = leftmost.left
        return root
```


## 翻转二叉树 (`invert-binary-tree`)

- 题目链接：https://leetcode.cn/problems/invert-binary-tree/
- 难度：Easy
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：2
- 最近提交时间：2025-12-25 09:16:51 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686998783 | 2025-12-25 09:14:27 CST | python3 | Accepted | 0 ms | 17 MB |
| 686999084 | 2025-12-25 09:16:51 CST | python3 | Accepted | 0 ms | 17 MB |

### 未通过提交代码
(所有提交均已通过)

## 引爆最多的炸弹 (`detonate-the-maximum-bombs`)

- 题目链接：https://leetcode.cn/problems/detonate-the-maximum-bombs/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 图, 几何, 数组, 数学
- 总提交次数：5
- 最近提交时间：2025-12-24 16:41:35 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-23 15:07:38 CST

```markdown
* 建模：几何问题 → 图论问题（节点 + 有向边）。
* 边界判断：距离用平方比较，避免浮点数误差。
* 模板化 BFS：
	* 用邻接表
	* visited 集防止重复
	* 队列按层遍历（虽然这里不关心层数，只关心可达数量）


* 识别关系： 看到“A 影响 B，B 影响 C”这种传递性的关系，就要立刻联想到图。
* 抽象建模： 把具体事物（炸弹）抽象成通用数据结构（节点），把具体关系（引爆）抽象成（边）。
* 套用算法： 一旦模型建立，问题就变成了“在图中，从一个点出发最多能访问多少个点”，直接套用标准的图遍历算法（BFS/DFS）即可。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686613153 | 2025-12-23 15:05:07 CST | python3 | Accepted | 315 ms | 17.2 MB |
| 686881697 | 2025-12-24 16:36:40 CST | python3 | Wrong Answer | N/A | N/A |
| 686882425 | 2025-12-24 16:39:05 CST | python3 | Wrong Answer | N/A | N/A |
| 686882869 | 2025-12-24 16:40:24 CST | python3 | Wrong Answer | N/A | N/A |
| 686883268 | 2025-12-24 16:41:35 CST | python3 | Accepted | 314 ms | 17.2 MB |

### 未通过提交代码
#### 提交 686881697 · Wrong Answer · 2025-12-24 16:36:40 CST · python3

```python
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        n = len(bombs)
        adj_list = collections.defaultdict(list)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                xi, yi, ri = bombs[i]
                xj, yj, _ = bombs[j]
                distance_sq = (xi - yi) ** 2 + (xj - yj) ** 2
                if distance_sq >= ri ** 2:
                    adj_list[i].append(j)
        max_detonation = 0
        for i in range(n):
            dq = collections.deque([i])
            visited = {i}
            for next_bomb in adj_list[i]:
                if next_bomb not in visited:
                    visited.add(next_bomb)
                    dq.append(next_bomb)
            max_detonation = max(max_detonation, len(visited))
        return max_detonation
```

#### 提交 686882425 · Wrong Answer · 2025-12-24 16:39:05 CST · python3

```python
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        n = len(bombs)
        adj_list = collections.defaultdict(list)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                xi, yi, ri = bombs[i]
                xj, yj, _ = bombs[j]
                distance_sq = (xi - yi) ** 2 + (xj - yj) ** 2
                if distance_sq >= ri ** 2:
                    adj_list[i].append(j)
        max_detonation = 0
        for i in range(n):
            dq = collections.deque([i])
            visited = {i}
            while dq:
                curr_bomb = dq.popleft()
                for next_bomb in adj_list[curr_bomb]:
                    if next_bomb not in visited:
                        visited.add(next_bomb)
                        dq.append(next_bomb)
            max_detonation = max(max_detonation, len(visited))
        return max_detonation
```

#### 提交 686882869 · Wrong Answer · 2025-12-24 16:40:24 CST · python3

```python
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        n = len(bombs)
        adj_list = collections.defaultdict(list)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                xi, yi, ri = bombs[i]
                xj, yj, _ = bombs[j]
                distance_sq = (xi - yi) ** 2 + (xj - yj) ** 2
                if distance_sq <= ri ** 2:
                    adj_list[i].append(j)
        max_detonation = 0
        for i in range(n):
            dq = collections.deque([i])
            visited = {i}
            while dq:
                curr_bomb = dq.popleft()
                for next_bomb in adj_list[curr_bomb]:
                    if next_bomb not in visited:
                        visited.add(next_bomb)
                        dq.append(next_bomb)
            max_detonation = max(max_detonation, len(visited))
        return max_detonation
```


## 二进制矩阵中的最短路径 (`shortest-path-in-binary-matrix`)

- 题目链接：https://leetcode.cn/problems/shortest-path-in-binary-matrix/
- 难度：Medium
- 标签：广度优先搜索, 数组, 矩阵
- 总提交次数：11
- 最近提交时间：2025-12-24 16:18:35 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-24 16:08:15 CST

```markdown
这题可以看成无权图上的最短路径问题，每个值为 0 的格子是一个节点，相邻 8 个方向的 0 之间有边。
无权图最短路用 BFS 最合适，因为 BFS 是按层扩展的，第一次走到终点的那一层就是最短路径长度。
具体做法是：如果起点或终点是 1 直接返回 -1；否则用队列从 (0,0) 开始 BFS，队列里存 (行, 列, 当前距离)，每次把 8 个方向中在边界内且为 0 的格子入队，并用修改 grid 的方式标记访问过。第一次弹出终点时返回对应距离，如果 BFS 结束没到终点就返回 -1。时间复杂度和空间复杂度都是 O(n²)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686393055 | 2025-12-22 16:09:31 CST | python3 | Runtime Error | N/A | N/A |
| 686393098 | 2025-12-22 16:09:40 CST | python3 | Wrong Answer | N/A | N/A |
| 686393593 | 2025-12-22 16:11:20 CST | python3 | Wrong Answer | N/A | N/A |
| 686394326 | 2025-12-22 16:13:40 CST | python3 | Accepted | 131 ms | 17.8 MB |
| 686575589 | 2025-12-23 11:53:11 CST | python3 | Wrong Answer | N/A | N/A |
| 686575736 | 2025-12-23 11:53:57 CST | python3 | Accepted | 135 ms | 17.4 MB |
| 686875342 | 2025-12-24 16:17:39 CST | python3 | Runtime Error | N/A | N/A |
| 686875374 | 2025-12-24 16:17:44 CST | python3 | Runtime Error | N/A | N/A |
| 686875417 | 2025-12-24 16:17:52 CST | python3 | Wrong Answer | N/A | N/A |
| 686875493 | 2025-12-24 16:18:06 CST | python3 | Accepted | 119 ms | 17.4 MB |
| 686875671 | 2025-12-24 16:18:35 CST | python3 | Accepted | 117 ms | 17.5 MB |

### 未通过提交代码
#### 提交 686393055 · Runtime Error · 2025-12-22 16:09:31 CST · python3

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] = 1:
            return -1
        if n == 1:
            return 1
        dq = collections.deque([(0, 0, 1)])
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ]
        while dq:
            curr_r, curr_c, dist = dq.popleft()
            if curr_r == n-1 and curr_c == n-1:
                return dist
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < n-1 and 0 <= nc < n-1 and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    dq.append((nr, nc, dist + 1))
        return -1
```

#### 提交 686393098 · Wrong Answer · 2025-12-22 16:09:40 CST · python3

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        if n == 1:
            return 1
        dq = collections.deque([(0, 0, 1)])
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ]
        while dq:
            curr_r, curr_c, dist = dq.popleft()
            if curr_r == n-1 and curr_c == n-1:
                return dist
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < n-1 and 0 <= nc < n-1 and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    dq.append((nr, nc, dist + 1))
        return -1
```

#### 提交 686393593 · Wrong Answer · 2025-12-22 16:11:20 CST · python3

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        if n == 1:
            return 1
        dq = collections.deque([(0, 0, 1)])
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ]
        grid[0][0] = 1
        while dq:
            curr_r, curr_c, dist = dq.popleft()
            if curr_r == n-1 and curr_c == n-1:
                return dist
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < n-1 and 0 <= nc < n-1 and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    dq.append((nr, nc, dist + 1))
        return -1
```

#### 提交 686575589 · Wrong Answer · 2025-12-23 11:53:11 CST · python3

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[n-1][n-1] == 1 or grid[0][0] == 1: 
            return -1
        if n == 1:
            return 0
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ]
        dq = collections.deque([(0, 0, 1)])
        grid[0][0] = 1
        while dq:
            curr_r, curr_c, dist = dq.popleft()
            if curr_r == curr_c == n - 1:
                return dist
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    dq.append((nr, nc, dist + 1))
        return -1
```

#### 提交 686875342 · Runtime Error · 2025-12-24 16:17:39 CST · python3

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        if n == 1:
            return 0
        dq = collections.deque([(0, 0, 1)])
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1], 
            [1, -1], [1, 0], [1, 1]
        ]
        while dq:
            r, c, dist = dq.popleft()
            if r == n-1 and c == n-1:
                return dist
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][n] == 1
                    dq.append((nr, nc, dist + 1))
        return -1
```

#### 提交 686875374 · Runtime Error · 2025-12-24 16:17:44 CST · python3

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        if n == 1:
            return 0
        dq = collections.deque([(0, 0, 1)])
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1], 
            [1, -1], [1, 0], [1, 1]
        ]
        while dq:
            r, c, dist = dq.popleft()
            if r == n-1 and c == n-1:
                return dist
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][n] = 1
                    dq.append((nr, nc, dist + 1))
        return -1
```

#### 提交 686875417 · Wrong Answer · 2025-12-24 16:17:52 CST · python3

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        if n == 1:
            return 0
        dq = collections.deque([(0, 0, 1)])
        directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 1], 
            [1, -1], [1, 0], [1, 1]
        ]
        while dq:
            r, c, dist = dq.popleft()
            if r == n-1 and c == n-1:
                return dist
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    dq.append((nr, nc, dist + 1))
        return -1
```


## 迷宫中离入口最近的出口 (`nearest-exit-from-entrance-in-maze`)

- 题目链接：https://leetcode.cn/problems/nearest-exit-from-entrance-in-maze/
- 难度：Medium
- 标签：广度优先搜索, 数组, 矩阵
- 总提交次数：4
- 最近提交时间：2025-12-24 15:58:51 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-24 15:52:37 CST

```markdown
我用 BFS 求无权网格上的最短路径。从入口开始做层序遍历，每次向四个方向扩展，只走到是 '.' 的空地，并且用原数组把访问过的位置改成 '+'，防止重复访问。判断出口时要注意不能把入口本身算进去：我只在访问到新的空地时，如果它在边界上，就立即返回当前步数加一，因为 BFS 按层扩展，第一次遇到的出口就是最近出口。如果 BFS 结束都没遇到出口，就返回 -1。时间复杂度和空间复杂度都是 O(mn)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686380255 | 2025-12-22 15:42:29 CST | python3 | Wrong Answer | N/A | N/A |
| 686383037 | 2025-12-22 15:51:28 CST | python3 | Accepted | 55 ms | 19.1 MB |
| 686572677 | 2025-12-23 11:40:00 CST | python3 | Accepted | 59 ms | 18.8 MB |
| 686863840 | 2025-12-24 15:58:51 CST | python3 | Accepted | 58 ms | 18.9 MB |

### 未通过提交代码
#### 提交 686380255 · Wrong Answer · 2025-12-22 15:42:29 CST · python3

```python
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        rows, cols = len(maze), len(maze[0])
        start_r, start_c = entrance
        maze[start_r][start_c] = '+'
        dq = collections.deque([(start_r, start_c, 0)])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while dq:
            curr_r, curr_c, steps = dq.popleft()
            if curr_r in [0, rows - 1] or curr_c in [0, cols - 1]:
                return steps
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if not (0 <= nr < rows) or not (0 <= nc < cols):
                    continue
                if maze[nr][nc] != '.':
                    continue
                dq.append((nr, nc, steps + 1))
                maze[nr][nc] = '+'
        return -1
```


## 太平洋大西洋水流问题 (`pacific-atlantic-water-flow`)

- 题目链接：https://leetcode.cn/problems/pacific-atlantic-water-flow/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 数组, 矩阵
- 总提交次数：9
- 最近提交时间：2025-12-24 15:37:48 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-24 15:38:59 CST

```markdown
这题我会用逆向思维来解。我不从每个点出发，而是反过来从两个大洋出发，看水能“倒灌”到哪些格子上。
我会做两次多源 BFS：
第一次从太平洋边界出发，找出所有能流到太平洋的格子。
第二次从大西洋边界出发，找出所有能流到大西洋的格子。
最后，我取这两个结果集的交集，就是答案。整个过程每个格子最多被访问常数次，所以时间和空间复杂度都是 O(mn)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686649283 | 2025-12-23 16:54:43 CST | python3 | Runtime Error | N/A | N/A |
| 686649435 | 2025-12-23 16:55:09 CST | python3 | Runtime Error | N/A | N/A |
| 686649608 | 2025-12-23 16:55:39 CST | python3 | Wrong Answer | N/A | N/A |
| 686649838 | 2025-12-23 16:56:23 CST | python3 | Wrong Answer | N/A | N/A |
| 686650221 | 2025-12-23 16:57:38 CST | python3 | Wrong Answer | N/A | N/A |
| 686650683 | 2025-12-23 16:59:03 CST | python3 | Wrong Answer | N/A | N/A |
| 686651058 | 2025-12-23 17:00:11 CST | python3 | Accepted | 39 ms | 18.3 MB |
| 686856730 | 2025-12-24 15:37:20 CST | python3 | Runtime Error | N/A | N/A |
| 686856860 | 2025-12-24 15:37:48 CST | python3 | Accepted | 41 ms | 18.4 MB |

### 未通过提交代码
#### 提交 686649283 · Runtime Error · 2025-12-23 16:54:43 CST · python3

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        rows, cols = len(heights), len(heights[0])
        def bfs(dq, reachable):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + nr, c + dc
                if not 0 <= nr < rows or not 0 <= nc < cols:
                    continue
                if (nr, nc) in reachable:
                    continue
                if heights[nr][nc] <= heights[r][c]:
                    continue
                reachable.add((nr, nc))
                dq.append(nr, nc)
        pacific_q = collections.deque()
        pacific_reachable = set()
        for r in range(rows):
            pacific_q.append((r, 0))
            pacific_reachable.add((r, 0))
        for c in range(1, cols):
            pacific_q.append((0, c))
            pacific_reachable.add((0, c))
        bfs(pacific_q, pacific_reachable)

        atlantic_q = collections.deque()
        atlantic_reachable = set()
        for r in range(rows):
            atlantic_q.append((r, cols-1))
            atlantic_reachable.add((r, cols-1))
        for c in range(cols-1):
            atlantic_q.append((rows-1, c))
            atlantic_reachable.add((rows-1, c))
        bfs(atlantic_q, atlantic_reachable)

        res = []
        for r in rows:
            for c in cols:
                if (r, c) in pacific_reachable and (r, c) in atlantic_reachable:
                    res.append((r, c))
        return res
```

#### 提交 686649435 · Runtime Error · 2025-12-23 16:55:09 CST · python3

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        rows, cols = len(heights), len(heights[0])
        def bfs(dq, reachable):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not 0 <= nr < rows or not 0 <= nc < cols:
                    continue
                if (nr, nc) in reachable:
                    continue
                if heights[nr][nc] <= heights[r][c]:
                    continue
                reachable.add((nr, nc))
                dq.append(nr, nc)
        pacific_q = collections.deque()
        pacific_reachable = set()
        for r in range(rows):
            pacific_q.append((r, 0))
            pacific_reachable.add((r, 0))
        for c in range(1, cols):
            pacific_q.append((0, c))
            pacific_reachable.add((0, c))
        bfs(pacific_q, pacific_reachable)

        atlantic_q = collections.deque()
        atlantic_reachable = set()
        for r in range(rows):
            atlantic_q.append((r, cols-1))
            atlantic_reachable.add((r, cols-1))
        for c in range(cols-1):
            atlantic_q.append((rows-1, c))
            atlantic_reachable.add((rows-1, c))
        bfs(atlantic_q, atlantic_reachable)

        res = []
        for r in rows:
            for c in cols:
                if (r, c) in pacific_reachable and (r, c) in atlantic_reachable:
                    res.append((r, c))
        return res
```

#### 提交 686649608 · Wrong Answer · 2025-12-23 16:55:39 CST · python3

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        rows, cols = len(heights), len(heights[0])
        def bfs(dq, reachable):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not 0 <= nr < rows or not 0 <= nc < cols:
                    continue
                if (nr, nc) in reachable:
                    continue
                if heights[nr][nc] <= heights[r][c]:
                    continue
                reachable.add((nr, nc))
                dq.append(nr, nc)
        pacific_q = collections.deque()
        pacific_reachable = set()
        for r in range(rows):
            pacific_q.append((r, 0))
            pacific_reachable.add((r, 0))
        for c in range(1, cols):
            pacific_q.append((0, c))
            pacific_reachable.add((0, c))
        bfs(pacific_q, pacific_reachable)

        atlantic_q = collections.deque()
        atlantic_reachable = set()
        for r in range(rows):
            atlantic_q.append((r, cols-1))
            atlantic_reachable.add((r, cols-1))
        for c in range(cols-1):
            atlantic_q.append((rows-1, c))
            atlantic_reachable.add((rows-1, c))
        bfs(atlantic_q, atlantic_reachable)

        res = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in pacific_reachable and (r, c) in atlantic_reachable:
                    res.append((r, c))
        return res
```

#### 提交 686649838 · Wrong Answer · 2025-12-23 16:56:23 CST · python3

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        rows, cols = len(heights), len(heights[0])
        def bfs(dq, reachable):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not 0 <= nr < rows or not 0 <= nc < cols:
                    continue
                if (nr, nc) in reachable:
                    continue
                if heights[nr][nc] <= heights[r][c]:
                    continue
                reachable.add((nr, nc))
                dq.append(nr, nc)

        pacific_q = collections.deque()
        pacific_reachable = set()
        for r in range(rows):
            pacific_q.append((r, 0))
            pacific_reachable.add((r, 0))
        for c in range(1, cols):
            pacific_q.append((0, c))
            pacific_reachable.add((0, c))
        bfs(pacific_q, pacific_reachable)

        atlantic_q = collections.deque()
        atlantic_reachable = set()
        for r in range(rows):
            atlantic_q.append((r, cols-1))
            atlantic_reachable.add((r, cols-1))
        for c in range(cols-1):
            atlantic_q.append((rows-1, c))
            atlantic_reachable.add((rows-1, c))
        bfs(atlantic_q, atlantic_reachable)

        res = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in pacific_reachable and (r, c) in atlantic_reachable:
                    res.append([r, c])
        return res
```

#### 提交 686650221 · Wrong Answer · 2025-12-23 16:57:38 CST · python3

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        rows, cols = len(heights), len(heights[0])
        def bfs(dq, reachable):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not 0 <= nr < rows or not 0 <= nc < cols:
                    continue
                if (nr, nc) in reachable:
                    continue
                if heights[nr][nc] <= heights[r][c]:
                    continue
                reachable.add((nr, nc))
                dq.append((nr, nc))

        pacific_q = collections.deque()
        pacific_reachable = set()
        for r in range(rows):
            pacific_q.append((r, 0))
            pacific_reachable.add((r, 0))
        for c in range(1, cols):
            pacific_q.append((0, c))
            pacific_reachable.add((0, c))
        bfs(pacific_q, pacific_reachable)

        atlantic_q = collections.deque()
        atlantic_reachable = set()
        for r in range(rows):
            atlantic_q.append((r, cols-1))
            atlantic_reachable.add((r, cols-1))
        for c in range(cols-1):
            atlantic_q.append((rows-1, c))
            atlantic_reachable.add((rows-1, c))
        bfs(atlantic_q, atlantic_reachable)

        res = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in pacific_reachable and (r, c) in atlantic_reachable:
                    res.append([r, c])
        return res
```

#### 提交 686650683 · Wrong Answer · 2025-12-23 16:59:03 CST · python3

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        rows, cols = len(heights), len(heights[0])
        def bfs(dq, reachable):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            while dq:
                r, c = dq.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if not 0 <= nr < rows or not 0 <= nc < cols:
                        continue
                    if (nr, nc) in reachable:
                        continue
                    if heights[nr][nc] <= heights[r][c]:
                        continue
                    reachable.add((nr, nc))
                    dq.append((nr, nc))

        pacific_q = collections.deque()
        pacific_reachable = set()
        for r in range(rows):
            pacific_q.append((r, 0))
            pacific_reachable.add((r, 0))
        for c in range(1, cols):
            pacific_q.append((0, c))
            pacific_reachable.add((0, c))
        bfs(pacific_q, pacific_reachable)

        atlantic_q = collections.deque()
        atlantic_reachable = set()
        for r in range(rows):
            atlantic_q.append((r, cols-1))
            atlantic_reachable.add((r, cols-1))
        for c in range(cols-1):
            atlantic_q.append((rows-1, c))
            atlantic_reachable.add((rows-1, c))
        bfs(atlantic_q, atlantic_reachable)

        res = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in pacific_reachable and (r, c) in atlantic_reachable:
                    res.append([r, c])
        return res
```

#### 提交 686856730 · Runtime Error · 2025-12-24 15:37:20 CST · python3

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        m, n = len(heights), len(heights[0])
        def bfs(dq, reachable):
            directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            while dq:
                r, c = dq.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < m and 0 <= nc < n):
                        continue
                    if (nr, nc) in reachable or heights[nr][nc] < heights[r][c]:
                        continue
                    reachable.add((nr, nc))
                    dq.append((nr, nc))
        pacific_q = collections.deque()
        pacific_reachable = set()
        for r in range(m):
            pacific_q.add((r, 0))
            pacific_reachable.add((r, 0))
        for c in range(1, n):
            pacific_q.add((0, c))
            pacific_reachable.add((0, c))
        bfs(pacific_q, pacific_reachable)

        atlantic_q = collections.deque()
        atlantic_reachable = set()
        for r in range(m):
            atlantic_q.add((r, n-1))
            atlantic_reachable.add((r, n-1))
        for c in range(n-1):
            atlantic_q.add((m-1, c))
            atlantic_reachable.add((m-1, c))
        bfs(atlantic_q, atlantic_reachable)

        res = []
        for r in range(m):
            for c in range(n):
                if (r, c) in pacific_reachable and (r, c) in atlantic_reachable:
                    res.append([r, c])
        return res
```


## 01 矩阵 (`01-matrix`)

- 题目链接：https://leetcode.cn/problems/01-matrix/
- 难度：Medium
- 标签：广度优先搜索, 数组, 动态规划, 矩阵
- 总提交次数：5
- 最近提交时间：2025-12-24 15:12:30 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-23 15:36:16 CST

```markdown
这道题是典型的多源广度优先搜索（BFS）问题。
我会把所有值为 0 的单元格作为 BFS 的起始点，将它们全部放入队列，并将它们的距离设为 0。
然后，我进行 BFS 遍历，按层扩展。当我从一个距离为 d 的单元格，访问到一个未被访问过的邻居时，这个邻居的最近距离就是 d+1。
这样通过一次 BFS，就能计算出所有 1 到最近 0 的最短距离。
整体的时间和空间复杂度都是 O(m*n)。

逆向思维/多源 BFS 框架：
* 识别信号：当题目要求“多个点”到“最近的某个目标点”的距离时，立刻想到反向操作——把所有“目标点”作为源头，进行一次多源 BFS。
* 适用场景：腐烂橘子（腐烂橘子 -> 新鲜橘子）、01矩阵（0 -> 1）、地图中求每个点到最近水源的距离等。
状态与空间复用：
* 在网格问题中，常常可以用一个结果矩阵（比如本题的 ans）来同时承担两个角色：存储最终结果 和 充当 visited 数组。
* 通过初始化一个特殊值（如 -1）来表示“未访问”，可以省去一个额外的 visited 集合或数组，代码更简洁。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686625623 | 2025-12-23 15:43:29 CST | python3 | Wrong Answer | N/A | N/A |
| 686625695 | 2025-12-23 15:43:43 CST | python3 | Wrong Answer | N/A | N/A |
| 686625937 | 2025-12-23 15:44:29 CST | python3 | Accepted | 109 ms | 19.6 MB |
| 686848413 | 2025-12-24 15:11:45 CST | python3 | Wrong Answer | N/A | N/A |
| 686848662 | 2025-12-24 15:12:30 CST | python3 | Accepted | 111 ms | 19.8 MB |

### 未通过提交代码
#### 提交 686625623 · Wrong Answer · 2025-12-23 15:43:29 CST · python3

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        if not mat or not mat[0]:
            return []
        m, n = len(mat), len(mat[0])
        res = [[-1] * n for _ in range(m)]
        dq = collections.deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    res[i][j] = 0
                    dq.append((i, j))
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while dq:
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and res[nr][nc] == -1:
                    res[nr][nc] = res[r][c] + 1
```

#### 提交 686625695 · Wrong Answer · 2025-12-23 15:43:43 CST · python3

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        if not mat or not mat[0]:
            return []
        m, n = len(mat), len(mat[0])
        res = [[-1] * n for _ in range(m)]
        dq = collections.deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    res[i][j] = 0
                    dq.append((i, j))
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while dq:
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and res[nr][nc] == -1:
                    res[nr][nc] = res[r][c] + 1
        return res
```

#### 提交 686848413 · Wrong Answer · 2025-12-24 15:11:45 CST · python3

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        if not mat or not mat[0]:
            return []
        m, n = len(mat), len(mat[0])
        res = [[-1] * n for _ in range(m)]
        dq = collections.deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    res[i][j] = 0
                    dq.append((i, j))
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while dq:
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < m and 0 <= nc < n):
                    continue
                if res[nr][nc] == '-1':
                    res[nr][nc] = res[r][c] + 1
                    dq.append((nr, nc))
        return res
```


## 腐烂的橘子 (`rotting-oranges`)

- 题目链接：https://leetcode.cn/problems/rotting-oranges/
- 难度：Medium
- 标签：广度优先搜索, 数组, 矩阵
- 总提交次数：7
- 最近提交时间：2025-12-24 14:07:57 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-23 14:07:57 CST

```markdown
这题是典型的多源 BFS 问题。
我先遍历网格，统计新鲜橘子数量，同时把所有腐烂橘子的位置入队，作为 BFS 的多个起点。
然后用队列按层 BFS，每一层代表 1 分钟：对当前队列中的每个腐烂橘子，向四个方向扩散，如果遇到新鲜橘子，就把它标记为腐烂并入队，同时新鲜数量减一。
每处理完一层，分钟数加一。
最后如果新鲜橘子为 0，就返回分钟数，否则说明还有橘子腐烂不到，返回 -1。

时间复杂度：O(m * n)，每个格子最多进队列一次。
空间复杂度：O(m * n)，队列最坏情况下装下所有格子。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686594233 | 2025-12-23 13:57:29 CST | python3 | Runtime Error | N/A | N/A |
| 686594381 | 2025-12-23 13:58:16 CST | python3 | Wrong Answer | N/A | N/A |
| 686594532 | 2025-12-23 13:59:02 CST | python3 | Wrong Answer | N/A | N/A |
| 686595330 | 2025-12-23 14:02:52 CST | python3 | Accepted | 3 ms | 17.2 MB |
| 686828993 | 2025-12-24 14:02:38 CST | python3 | Runtime Error | N/A | N/A |
| 686829025 | 2025-12-24 14:02:47 CST | python3 | Wrong Answer | N/A | N/A |
| 686830054 | 2025-12-24 14:07:57 CST | python3 | Accepted | 3 ms | 16.9 MB |

### 未通过提交代码
#### 提交 686594233 · Runtime Error · 2025-12-23 13:57:29 CST · python3

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        dq = collections.deque()
        fresh = 0
        for r in rows:
            for c in cols:
                if grid[r][c] == 2:
                    dq.append((r, c))
                else:
                    fresh += 1
        if fresh == 0:
            return 0
        minutes = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while dq and fresh > 0:
            sz = len(dq)
            for _ in range(sz):
                r, c = dq.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if not 0 <= nr < rows or not 0 <= nc < cols:
                        continue
                    if grid[nr][nc] == 1:
                        grid[nr][nc] == 2
                        fresh -= 1
                        dq.append((nr, nc))
            minutes += 1
        return minutes if fresh == 0 else -1
```

#### 提交 686594381 · Wrong Answer · 2025-12-23 13:58:16 CST · python3

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        dq = collections.deque()
        fresh = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    dq.append((r, c))
                else:
                    fresh += 1
        if fresh == 0:
            return 0
        minutes = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while dq and fresh > 0:
            sz = len(dq)
            for _ in range(sz):
                r, c = dq.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if not 0 <= nr < rows or not 0 <= nc < cols:
                        continue
                    if grid[nr][nc] == 1:
                        grid[nr][nc] == 2
                        fresh -= 1
                        dq.append((nr, nc))
            minutes += 1
        return minutes if fresh == 0 else -1
```

#### 提交 686594532 · Wrong Answer · 2025-12-23 13:59:02 CST · python3

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        dq = collections.deque()
        fresh = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    dq.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1
        if fresh == 0:
            return 0
        minutes = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while dq and fresh > 0:
            sz = len(dq)
            for _ in range(sz):
                r, c = dq.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if not 0 <= nr < rows or not 0 <= nc < cols:
                        continue
                    if grid[nr][nc] == 1:
                        grid[nr][nc] == 2
                        fresh -= 1
                        dq.append((nr, nc))
            minutes += 1
        return minutes if fresh == 0 else -1
```

#### 提交 686828993 · Runtime Error · 2025-12-24 14:02:38 CST · python3

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dq = collections.deque()
        fresh = 0
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 2:
                    dq.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1
        if fresh == 0:
            return 0
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        minutes = 0
        while dq and fresh > 0:
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not 0 <= nr < m or not 0 <= nc < n:
                    continue
                if grid[nr][nc] == 1:
                    grid[nr][nc] = 2:
                    fresh -= 1
                    dq.append((nr, nc))
            minutes += 1
        return minutes if fresh == 0 else -1
```

#### 提交 686829025 · Wrong Answer · 2025-12-24 14:02:47 CST · python3

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dq = collections.deque()
        fresh = 0
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 2:
                    dq.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1
        if fresh == 0:
            return 0
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        minutes = 0
        while dq and fresh > 0:
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not 0 <= nr < m or not 0 <= nc < n:
                    continue
                if grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    dq.append((nr, nc))
            minutes += 1
        return minutes if fresh == 0 else -1
```


## 除法求值 (`evaluate-division`)

- 题目链接：https://leetcode.cn/problems/evaluate-division/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 并查集, 图, 数组, 字符串, 最短路
- 总提交次数：6
- 最近提交时间：2025-12-24 11:53:45 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-21 09:04:04 CST

```markdown
把每个变量看成图的节点，每个等式 a/b=k 看成两条有向边：a→b 权重 k，b→a 权重 1/k。
对每个查询 x/y，我在图里从 x 用 BFS 或 DFS 找到 y，沿途把边权相乘就是 x/y 的结果。如果有变量没出现过，或者找不到路径，就返回 -1。
这样时间复杂度是建图 O(E)，每个查询一次图搜索，整体足够应对题目规模。

这个题本质上训练的是“关系网络建模 + 图搜索 / 并查集”这套通用框架。

类似题目：
“谁和谁是朋友 / 亲属 / 同组？”
“货币、概率、比例、权重在网络中传递”
“很多 pair + 查询”

dq 中存储的是从图的起始节点到当前节点的路径权重的乘积
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685934954 | 2025-12-20 09:39:30 CST | python3 | Wrong Answer | N/A | N/A |
| 685935148 | 2025-12-20 09:41:33 CST | python3 | Accepted | 0 ms | 17.2 MB |
| 686108989 | 2025-12-21 09:13:46 CST | python3 | Runtime Error | N/A | N/A |
| 686109127 | 2025-12-21 09:16:28 CST | python3 | Accepted | 0 ms | 17.1 MB |
| 686809821 | 2025-12-24 11:52:33 CST | python3 | Runtime Error | N/A | N/A |
| 686810043 | 2025-12-24 11:53:45 CST | python3 | Accepted | 0 ms | 17.1 MB |

### 未通过提交代码
#### 提交 685934954 · Wrong Answer · 2025-12-20 09:39:30 CST · python3

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        for (a, b), val in zip(equations, values):
            graph[a][b]= val
        
        def bfs(start, end):
            if start not in graph or end not in graph:
                return -1.0
            if start == end:
                return 1.0
            dq = collections.deque([(start, 1)])
            visited = set([start])
            while dq:
                cur_node, cur_val = dq.popleft()
                if cur_node == end:
                    return cur_val
                for neibor, weight in graph[cur_node].items():
                    if neibor in visited:
                        continue
                    visited.add(neibor)
                    dq.append((neibor, cur_val * weight))
            return -1.0
        
        res = []
        for x, y in queries:
            res.append(bfs(x, y))
        
        return res
```

#### 提交 686108989 · Runtime Error · 2025-12-21 09:13:46 CST · python3

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        for (a, b), val in zip(equations, values):
            graph[a][b] = val
            graph[b][a] = 1.0 / val
        
        def bfs(start, end):
            if start not in graph or end not in graph:
                return -1.0
            if start == end:
                return 1.0
            dq = collections.deque([(start, 1.0)])
            visited = set([start])
            while dq:
                curr_node, curr_val = dq.popleft()
                if curr_node == end:
                    return curr_val
                for next_node, weight in graph[curr_node].items():
                    if next_node in visited:
                        continue
                    visited.add(next_node)
                    dq.append((next_node, curr_val * weight))
        
        res = []
        for x, y in queries:
            res.append(bfs(x, y))
        
        return res
```

#### 提交 686809821 · Runtime Error · 2025-12-24 11:52:33 CST · python3

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = collections.defaultdict(dict)
        for (a, b), c in zip(equations, values):
            graph[a][b] = c
            graph[b][a] = 1.0 / c
        
        def bfs(start, end):
            if start not in graph or end not in graph:
                return -1.0
            if start == end:
                return 1.0
            dq = collections.deque([(start, 1.0)])
            visited = {start}
            while dq:
                curr, val = dq.popleft()
                if curr == end:
                    return val
                for next_node, weight in graph[curr].items():
                    if next_node not in visited:
                        visited.add(next_node)
                        dq.append((next_node, val * weight))
        res = []
        for x, y in queries:
            res.append(bfs(x, y))
        return res
```


## 单词接龙 (`word-ladder`)

- 题目链接：https://leetcode.cn/problems/word-ladder/
- 难度：Hard
- 标签：广度优先搜索, 哈希表, 字符串
- 总提交次数：7
- 最近提交时间：2025-12-24 11:36:06 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-16 11:20:37 CST

```markdown
面试口述要点
* 把每个词当成节点，差一字母连边。这是无权图最短路问题。
* BFS按层扩展，第一次到达end就是最短步数；实现用队列+哈希set去重，遍历时对每一位尝试26个字母生成邻居。
* 复杂度约O(N·L·26)，空间O(N)。若字典大，用双向BFS，从两端扩展并始终扩展小的一端，平均更快。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 678442754 | 2025-11-16 14:48:44 CST | python3 | Wrong Answer | N/A | N/A |
| 678442968 | 2025-11-16 14:49:47 CST | python3 | Accepted | 351 ms | 18.5 MB |
| 678445117 | 2025-11-16 14:59:32 CST | python3 | Accepted | 355 ms | 18.5 MB |
| 679182725 | 2025-11-19 13:46:48 CST | python3 | Runtime Error | N/A | N/A |
| 679182835 | 2025-11-19 13:47:26 CST | python3 | Wrong Answer | N/A | N/A |
| 679183215 | 2025-11-19 13:49:18 CST | python3 | Accepted | 371 ms | 18.4 MB |
| 686806112 | 2025-12-24 11:36:06 CST | python3 | Accepted | 347 ms | 18.7 MB |

### 未通过提交代码
#### 提交 678442754 · Wrong Answer · 2025-11-16 14:48:44 CST · python3

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if endWord not in word_set:
            return 0
        dq = collections.deque([(beginWord, 1)])
        visited = {beginWord}
        while dq:
            current_word, length = dq.popleft()
            if current_word == endWord:
                return length

            # 尝试构造新单词，遍历word的每一个位置
            for i in range(len(current_word)):
                # 从'a'到'z'遍历，尝试替换当前位置的字母
                for char_code in range(ord('a'), ord('z') + 1):
                    char = chr(char_code)
                    next_word = current_word[:i] + char + current_word[i:]
                    if next_word in word_set and next_word not in visited:
                        visited.add(next_word)
                        dq.append((next_word, length + 1))
        
        return 0
```

#### 提交 679182725 · Runtime Error · 2025-11-19 13:46:48 CST · python3

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if endWord not in word_set:
            return 0
        dq = collections.deque([beginWord, 1])
        visited = {beginWord}
        
        while dq:
            current_word, length = dq.pop()
            if current_word == endWord:
                return length
            # 尝试替换每一位字符
            for i in range(len(current_word)):
                for char_code in range(ord('a'), ord('z') + 1):
                    ch = chr(char_code)
                    next_word = current_word[:i] + ch + current_word[i+1:]
                    if next_word in word_set and next_word not in visited:
                        visited.add(next_word)
                        dq.append((next_word, length + 1))
        
        return 0
```

#### 提交 679182835 · Wrong Answer · 2025-11-19 13:47:26 CST · python3

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if endWord not in word_set:
            return 0
        dq = collections.deque([(beginWord, 1)])
        visited = {beginWord}
        
        while dq:
            current_word, length = dq.pop()
            if current_word == endWord:
                return length
            # 尝试替换每一位字符
            for i in range(len(current_word)):
                for char_code in range(ord('a'), ord('z') + 1):
                    ch = chr(char_code)
                    next_word = current_word[:i] + ch + current_word[i+1:]
                    if next_word in word_set and next_word not in visited:
                        visited.add(next_word)
                        dq.append((next_word, length + 1))
        
        return 0
```


## 账户合并 (`accounts-merge`)

- 题目链接：https://leetcode.cn/problems/accounts-merge/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 并查集, 数组, 哈希表, 字符串, 排序
- 总提交次数：3
- 最近提交时间：2025-12-24 11:15:34 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-24 10:26:23 CST

```markdown
多列表里有“公共元素就属于同一组” → 想“连通分量”
典型套路：
* 节点：最小粒度的唯一标识（本题是 email）
* 边：出现在同一组 / 同一账户里就连边
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686799647 | 2025-12-24 11:14:00 CST | python3 | Runtime Error | N/A | N/A |
| 686799996 | 2025-12-24 11:15:17 CST | python3 | Runtime Error | N/A | N/A |
| 686800092 | 2025-12-24 11:15:34 CST | python3 | Accepted | 31 ms | 20.3 MB |

### 未通过提交代码
#### 提交 686799647 · Runtime Error · 2025-12-24 11:14:00 CST · python3

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        adj_list = collections.defaultdict(list)
        emails_to_name = {}
        for account in accounts:
            name = account[0]
            emails = accounts[1:]
            if not emails:
                continue
            first_email = emails[0]
            emails_to_name[first_email] = name
            for email in emails[1:]:
                adj_list[first_email].append(email)
                adj_list[email].append(first_email)
                emails_to_name[email] = name
        res = []
        visited = set()
        for email in emails_to_name:
            if email not in visited:
                dq = collections.deque([email])
                visited.add(email)
                emails_in_group = []
                while dq:
                    e = dq.popleft()
                    emails_in_group.add(e)
                    for neighbor in adj_list[e]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            dq.append(neighbor)
                emails_in_group.sort()
                name = emails_to_name[email]
                res.append([name] + emails_in_group)
        return res
```

#### 提交 686799996 · Runtime Error · 2025-12-24 11:15:17 CST · python3

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        adj_list = collections.defaultdict(list)
        emails_to_name = {}
        for account in accounts:
            name = account[0]
            emails = account[1:]
            if not emails:
                continue
            first_email = emails[0]
            emails_to_name[first_email] = name
            for email in emails[1:]:
                adj_list[first_email].append(email)
                adj_list[email].append(first_email)
                emails_to_name[email] = name
        res = []
        visited = set()
        for email in emails_to_name:
            if email not in visited:
                dq = collections.deque([email])
                visited.add(email)
                emails_in_group = []
                while dq:
                    e = dq.popleft()
                    emails_in_group.add(e)
                    for neighbor in adj_list[e]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            dq.append(neighbor)
                emails_in_group.sort()
                name = emails_to_name[email]
                res.append([name] + emails_in_group)
        return res
```


## 水壶问题 (`water-and-jug-problem`)

- 题目链接：https://leetcode.cn/problems/water-and-jug-problem/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 数学
- 总提交次数：3
- 最近提交时间：2025-12-24 09:31:53 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-24 08:50:45 CST

```markdown
我把状态定义为两个水壶当前的水量 (a, b)，从初始状态 (0, 0) 开始做 BFS。
从一个状态可以通过 6 种操作得到新状态：分别是把任一壶灌满、倒空，以及两个壶之间相互倒水直到一方空或一方满。
用队列做层序遍历，用集合记录访问过的状态。
在搜索过程中，如果出现某个状态使得 a == z、b == z 或 a + b == z，就返回 True；如果 BFS 结束都没有，就返回 False。
这个 BFS 主要是为了练习状态建模，实际这题更推荐用 gcd 的数学解法。


这题本质上是一个数论问题，可以用裴蜀定理来解。
任何能被量出的水量，都必然是两个壶容量 x 和 y 的线性组合，也就是 ax + by 的形式。
根据裴蜀定理，这个组合出来的水量必须是 x 和 y 最大公约数 gcd(x, y) 的倍数。
同时，目标水量不能超过两个壶的总容量。
所以，只要 target <= x + y 并且 target % gcd(x, y) == 0，就一定可以量出。
这种解法复杂度只有求 GCD 的 O(log n)，非常高效。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686773346 | 2025-12-24 09:24:47 CST | python3 | Accepted | 0 ms | 17.3 MB |
| 686774400 | 2025-12-24 09:31:25 CST | python3 | Wrong Answer | N/A | N/A |
| 686774473 | 2025-12-24 09:31:53 CST | python3 | Accepted | 0 ms | 17.1 MB |

### 未通过提交代码
#### 提交 686774400 · Wrong Answer · 2025-12-24 09:31:25 CST · python3

```python
class Solution:
    def canMeasureWater(self, x: int, y: int, target: int) -> bool:
        return target % math.gcd(x, y) == 0
```


## 最小基因变化 (`minimum-genetic-mutation`)

- 题目链接：https://leetcode.cn/problems/minimum-genetic-mutation/
- 难度：Medium
- 标签：广度优先搜索, 哈希表, 字符串
- 总提交次数：6
- 最近提交时间：2025-12-23 11:30:40 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686373034 | 2025-12-22 15:19:19 CST | python3 | Runtime Error | N/A | N/A |
| 686373081 | 2025-12-22 15:19:30 CST | python3 | Runtime Error | N/A | N/A |
| 686373151 | 2025-12-22 15:19:45 CST | python3 | Accepted | 0 ms | 17.1 MB |
| 686570276 | 2025-12-23 11:30:26 CST | python3 | Runtime Error | N/A | N/A |
| 686570298 | 2025-12-23 11:30:31 CST | python3 | Runtime Error | N/A | N/A |
| 686570343 | 2025-12-23 11:30:40 CST | python3 | Accepted | 0 ms | 17 MB |

### 未通过提交代码
#### 提交 686373034 · Runtime Error · 2025-12-22 15:19:19 CST · python3

```python
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        bank_set = set(bank)
        dq = collection.deque([(startGene, 0)])
        visited = {startGene}
        ch_list = ['A', 'C', 'G', 'T']
        while dq:
            curr_gene, steps = dq.popleft()
            if curr_gene == endGene:
                return steps
            for i in range(8):
                for ch in ch_list:
                    if curr_gene[i] == ch:
                        continue
                    new_gene = curr_gene[:i] + ch + curr_gene[i+1:]
                    if new_gene in bank and new_gene not in visited:
                        visited.add(new_gene)
                        dq.append(new_gene, steps + 1)
        return -1
```

#### 提交 686373081 · Runtime Error · 2025-12-22 15:19:30 CST · python3

```python
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        bank_set = set(bank)
        dq = collections.deque([(startGene, 0)])
        visited = {startGene}
        ch_list = ['A', 'C', 'G', 'T']
        while dq:
            curr_gene, steps = dq.popleft()
            if curr_gene == endGene:
                return steps
            for i in range(8):
                for ch in ch_list:
                    if curr_gene[i] == ch:
                        continue
                    new_gene = curr_gene[:i] + ch + curr_gene[i+1:]
                    if new_gene in bank and new_gene not in visited:
                        visited.add(new_gene)
                        dq.append(new_gene, steps + 1)
        return -1
```

#### 提交 686570276 · Runtime Error · 2025-12-23 11:30:26 CST · python3

```python
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        bank_set = set(bank)
        dq = collections.deque([(startGene, 0)])
        visited = {startGene}
        ch_list = ['A', 'C', 'G', 'T']
        while dq:
            curr_gene, steps = dq.popleft()
            if curr_gene == endGene:
                return steps
            for i in range(8):
                for ch in ch_list:
                    if curr_gene[i] == ch:
                        continue
                    new_gene = = curr_gene[:i] + ch + curr_gene[i+1:]
                    if new_gene in bank_set and new_gene not in visited:
                        visited.add(new_gene)
                        dq.append((new_gene, steps + 1)
        return -1
```

#### 提交 686570298 · Runtime Error · 2025-12-23 11:30:31 CST · python3

```python
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        bank_set = set(bank)
        dq = collections.deque([(startGene, 0)])
        visited = {startGene}
        ch_list = ['A', 'C', 'G', 'T']
        while dq:
            curr_gene, steps = dq.popleft()
            if curr_gene == endGene:
                return steps
            for i in range(8):
                for ch in ch_list:
                    if curr_gene[i] == ch:
                        continue
                    new_gene = curr_gene[:i] + ch + curr_gene[i+1:]
                    if new_gene in bank_set and new_gene not in visited:
                        visited.add(new_gene)
                        dq.append((new_gene, steps + 1)
        return -1
```


## 跳跃游戏 III (`jump-game-iii`)

- 题目链接：https://leetcode.cn/problems/jump-game-iii/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 数组
- 总提交次数：3
- 最近提交时间：2025-12-23 11:11:30 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-22 14:55:43 CST

```markdown
最标准的“图的可达性”问题
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686367985 | 2025-12-22 15:02:31 CST | python3 | Runtime Error | N/A | N/A |
| 686368102 | 2025-12-22 15:02:53 CST | python3 | Accepted | 7 ms | 22.5 MB |
| 686565106 | 2025-12-23 11:11:30 CST | python3 | Accepted | 13 ms | 22.4 MB |

### 未通过提交代码
#### 提交 686367985 · Runtime Error · 2025-12-22 15:02:31 CST · python3

```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        if arr[start] == 0:
            return True
        n = len(arr)
        visited = [False] * n
        dq = collections.deque([start])
        visited[start] = True
        while dq:
            pos = dq.popleft()
            if arr[pos] == 0:
                return True
            for next_pos in (pos + arr[pos], pos - arr[pos]):
                if 0 <= next_pos < n and not visited[next_pos]:
                    visited.add(next_pos)
                    dq.append(next_pos)
        return False
```


## 钥匙和房间 (`keys-and-rooms`)

- 题目链接：https://leetcode.cn/problems/keys-and-rooms/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 图
- 总提交次数：5
- 最近提交时间：2025-12-23 11:05:49 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-22 14:49:07 CST

```markdown
这题可以看成图的可达性问题：每个房间是一个节点，房间里的钥匙就是到其他房间的有向边。从 0 号房间开始做 DFS 或 BFS，一直把拿到钥匙能开的房间加入到待访问集合，用一个 visited 数组记录已经访问过的房间。遍历结束后看 visited 里是否所有房间都被访问过，如果是就返回 True，否则返回 False。时间复杂度是 O(N + E)，N 是房间数，E 是钥匙总数。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686362722 | 2025-12-22 14:44:20 CST | python3 | Runtime Error | N/A | N/A |
| 686363209 | 2025-12-22 14:46:08 CST | python3 | Runtime Error | N/A | N/A |
| 686363724 | 2025-12-22 14:47:57 CST | python3 | Runtime Error | N/A | N/A |
| 686363780 | 2025-12-22 14:48:09 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 686563496 | 2025-12-23 11:05:49 CST | python3 | Accepted | 0 ms | 17.4 MB |

### 未通过提交代码
#### 提交 686362722 · Runtime Error · 2025-12-22 14:44:20 CST · python3

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        n = len(rooms)
        visited = [False] * n
        visited[0] = True
        dq = collections.deque([rooms[0]])
        while dq:
            room = dq.popleft()
            for neighbor in rooms[room]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    dq.append(neighbor)
        for v in visited:
            if not v:
                return False
        return True
```

#### 提交 686363209 · Runtime Error · 2025-12-22 14:46:08 CST · python3

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        n = len(rooms)
        adj_list = collections.defaultdict(list)
        for i, room in enumerate(rooms):
            adj_list[i].append(room)
        visited = [False] * n
        visited[0] = True
        dq = collections.deque([rooms[0]])
        while dq:
            room = dq.popleft()
            for neighbor in adj_list[room]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    dq.append(neighbor)
        for v in visited:
            if not v:
                return False
        return True
```

#### 提交 686363724 · Runtime Error · 2025-12-22 14:47:57 CST · python3

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        n = len(rooms)
        visited = [False] * n
        visited[0] = True
        dq = collections.deque(0)
        while dq:
            room = dq.popleft()
            for neighbor in rooms[room]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    dq.append(neighbor)
        for v in visited:
            if not v:
                return False
        return True
```


## 最小高度树 (`minimum-height-trees`)

- 题目链接：https://leetcode.cn/problems/minimum-height-trees/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 图, 拓扑排序
- 总提交次数：6
- 最近提交时间：2025-12-23 10:58:57 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-22 14:00:46 CST

```markdown
在“无权树”里，按通常的定义，中心点集合一定只可能是 1 个或 2 个点，而且如果是 2 个，它们必然相邻。

这题本质是在树里找所有能产生最小高度的根，其实就是找树的中心点。
暴力做法是对每个点 BFS 算高度，O(n^2) 会超时。

正解是“剥洋葱”思路：把所有度为 1 的叶子一起删掉，删掉后会有新的叶子，再继续一层层删。

每删一层，都是在去掉距离中心最远的那圈节点。

最后剩下的 1~2 个点，就是树的中心，也就是所有最小高度树的根。

整个过程每个节点和边只处理一次，时间复杂度 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686355416 | 2025-12-22 14:16:44 CST | python3 | Runtime Error | N/A | N/A |
| 686355483 | 2025-12-22 14:17:00 CST | python3 | Wrong Answer | N/A | N/A |
| 686356763 | 2025-12-22 14:22:07 CST | python3 | Accepted | 47 ms | 24.7 MB |
| 686358000 | 2025-12-22 14:27:08 CST | python3 | Accepted | 47 ms | 25.2 MB |
| 686561423 | 2025-12-23 10:58:39 CST | python3 | Runtime Error | N/A | N/A |
| 686561523 | 2025-12-23 10:58:57 CST | python3 | Accepted | 39 ms | 25.1 MB |

### 未通过提交代码
#### 提交 686355416 · Runtime Error · 2025-12-22 14:16:44 CST · python3

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        adj_list = collections.defaultdict(list)
        degree = [0] * n
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
            degree[u] += 1
            degree[v] += 1
        dq = collections.deque()
        for i in range(n):
            if degree[i] == 1:
                deque.append(i)
        remaining = n
        while remaining > 2:
            sz = len(dq)
            remaining -= sz
            leaf = dq.popleft()
            for neibor in adj_list[leaf]:
                degree[neibor] -= 1
                if degree[neibor] == 1:
                    dq.append(neibor)
        return list(dq)
```

#### 提交 686355483 · Wrong Answer · 2025-12-22 14:17:00 CST · python3

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        adj_list = collections.defaultdict(list)
        degree = [0] * n
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
            degree[u] += 1
            degree[v] += 1
        dq = collections.deque()
        for i in range(n):
            if degree[i] == 1:
                dq.append(i)
        remaining = n
        while remaining > 2:
            sz = len(dq)
            remaining -= sz
            leaf = dq.popleft()
            for neibor in adj_list[leaf]:
                degree[neibor] -= 1
                if degree[neibor] == 1:
                    dq.append(neibor)
        return list(dq)
```

#### 提交 686561423 · Runtime Error · 2025-12-23 10:58:39 CST · python3

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        adj_list = collections.defaultdict(list)
        indegree = [0] * n
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
            indegree[u] += 1
            indegree[v] += 1
        dq = collections.deque()
        for i in range(n):
            if indegree[i] == 1:
                dq.append(i)
        remaining = n
        while remaining > 2:
            sz = len(dq)
            remaining -= sz
            for _ in range(sz):
                leaf = dq.popleft()
                for neighbor in adj_list[neighbor]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 1:
                        dq.append(neighbor)
        return list(dq)
```


## 二叉树中所有距离为 K 的结点 (`all-nodes-distance-k-in-binary-tree`)

- 题目链接：https://leetcode.cn/problems/all-nodes-distance-k-in-binary-tree/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 哈希表, 二叉树
- 总提交次数：4
- 最近提交时间：2025-12-23 10:41:25 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-22 12:32:53 CST

```markdown
这题本质是从目标节点出发，在一张无向图里找距离为 K 的所有节点。

二叉树本身只能往下走，但距离允许往父节点走，所以我第一步用 DFS 遍历整棵树，给每个节点建立一个父指针映射。这样从一个节点可以走到它的左孩子、右孩子和父节点。

第二步从 target 节点开始做 BFS，队列里存当前节点和它到 target 的距离，用一个 visited 集合避免重复访问。当 BFS 扩展到距离为 K 的那一层时，这一层所有节点的值就是答案，直接返回即可。整体时间复杂度 O(N)，空间 O(N)。


时间 / 空间复杂度:
建父指针 DFS：O(N)
BFS：最多访问每个节点一次，也是 O(N)
总时间复杂度：O(N)
辅助空间：父指针字典 + 访问集合 + 队列，O(N)
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686337806 | 2025-12-22 12:46:19 CST | python3 | Runtime Error | N/A | N/A |
| 686337845 | 2025-12-22 12:46:36 CST | python3 | Accepted | 44 ms | 17.4 MB |
| 686556222 | 2025-12-23 10:41:15 CST | python3 | Runtime Error | N/A | N/A |
| 686556267 | 2025-12-23 10:41:25 CST | python3 | Accepted | 43 ms | 17.4 MB |

### 未通过提交代码
#### 提交 686337806 · Runtime Error · 2025-12-22 12:46:19 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        if not root:
            return []
        adj_list = collection.defaultdict(list)
        stack = [(root, None)]
        while stack:
            node, parent = stack.pop()
            if parent:
                adj_list[node].append(parent)
                adj_list[parent].append(node)
            if node.left:
                stack.append((node.left, node))
            if node.right:
                stack.append((node.right, node))
        dq = collection.deque([(target, 0)])
        visited = {target}
        res = []
        while dq:
            node, distance = dq.popleft()
            if distance == k:
                res.append(node.val)
                continue
            if distance > k:
                break
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dq.append(((neighbor, distance + 1)))
        return res
```

#### 提交 686556222 · Runtime Error · 2025-12-23 10:41:15 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        if not root:
            return []
        adj_list = collections.defaultdict(list)
        stack = [(root, None)]
        while stack:
            node, parent = stack.pop()
            if parent:
                adj_list[parent].append(node)
                adj_list[node].append(parent)
            if node.left:
                stack.append((node.left, node))
            if node.right:
                stack.append(node.right, node)
        dq = collections.deque([(target, 0)])
        visited = {target}
        res = []
        while dq:
            node, dist = dq.popleft()
            if dist == k:
                res.append(node.val)
            if dist > k:
                break
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dq.append((neighbor, dist + 1))
        return res
```


## 二叉树寻路 (`path-in-zigzag-labelled-binary-tree`)

- 题目链接：https://leetcode.cn/problems/path-in-zigzag-labelled-binary-tree/
- 难度：Medium
- 标签：树, 数学, 二叉树
- 总提交次数：7
- 最近提交时间：2025-12-23 10:31:24 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-20 16:55:19 CST

```markdown
在同一层中，任意一个节点 label 和它“对称位置”的节点 symmetric_label 的值加起来是一个恒定的值。这个值等于该层的最大编号 + 该层的最小编号。


这是一个完全二叉树，只是每一层的标号是之字形的。
在正常完全二叉树里，编号为 x 的节点父亲是 x // 2； 较难的是如何从之字形编号算父亲。
对每一层来说，它的编号区间是 [2^level, 2^(level+1)-1]，之字形和正常编号只是在这一段内做了一次左右翻转，因此： 当前 label 对应的“正常编号”是 start + end - label。
正常编号的父亲是 normal // 2，结合翻转关系，可以得到一个统一公式： parent = (start + end - label) // 2。
所以我从目标 label 开始，循环： 计算当前层的 start/end，用这个公式跳到父亲，直到 1， 把路径记录下来再反转即可，时间复杂度是 O(log label)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686012064 | 2025-12-20 16:57:30 CST | python3 | Accepted | 0 ms | 16.8 MB |
| 686110577 | 2025-12-21 09:39:06 CST | python3 | Wrong Answer | N/A | N/A |
| 686110627 | 2025-12-21 09:39:55 CST | python3 | Accepted | 0 ms | 17 MB |
| 686316362 | 2025-12-22 10:58:19 CST | python3 | Runtime Error | N/A | N/A |
| 686316642 | 2025-12-22 10:59:17 CST | python3 | Accepted | 0 ms | 17 MB |
| 686553526 | 2025-12-23 10:31:04 CST | python3 | Wrong Answer | N/A | N/A |
| 686553600 | 2025-12-23 10:31:24 CST | python3 | Accepted | 0 ms | 16.9 MB |

### 未通过提交代码
#### 提交 686110577 · Wrong Answer · 2025-12-21 09:39:06 CST · python3

```python
class Solution:
    def pathInZigZagTree(self, label: int) -> List[int]:
        path = []
        while label:
            level = int(math.log2(label))
            level_start = 1 << level
            level_end = (1 << (level + 1)) - 1
            label = (level_start + level_end - label) // 2
            path.append(label)
        return path[::-1]
```

#### 提交 686316362 · Runtime Error · 2025-12-22 10:58:19 CST · python3

```python
class Solution:
    def pathInZigZagTree(self, label: int) -> List[int]:
        path = []
        while label:
            path.append(label)
            level = math.log2(label)
            level_start = 1 << level
            level_end = (1 << (level + 1)) -1
            label = (level_start + level_end - label) // 2
        return path[::-1]
```

#### 提交 686553526 · Wrong Answer · 2025-12-23 10:31:04 CST · python3

```python
class Solution:
    def pathInZigZagTree(self, label: int) -> List[int]:
        path = []
        while label:
            path.append(label)
            level = int(math.log2(label))
            level_start = 1 << level
            level_end = 1 << (level + 1) - 1
            label = (level_start + level_end - label) // 2
        return path[::-1]
```


## 打开转盘锁 (`open-the-lock`)

- 题目链接：https://leetcode.cn/problems/open-the-lock/
- 难度：Medium
- 标签：广度优先搜索, 数组, 哈希表, 字符串
- 总提交次数：8
- 最近提交时间：2025-12-23 10:22:13 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-21 17:37:18 CST

```markdown
状态总数最多 10^4；每个状态最多 8 个邻居。
BFS 最多访问每个状态一次，所以时间复杂度 O(10000) ≈ O(1) 常数级。
空间复杂度同样 O(10000) ≈ O(1)，主要是 visited 和队列。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686204424 | 2025-12-21 17:33:02 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 686204735 | 2025-12-21 17:34:34 CST | python3 | Accepted | 325 ms | 18.3 MB |
| 686315012 | 2025-12-22 10:53:33 CST | python3 | Wrong Answer | N/A | N/A |
| 686315108 | 2025-12-22 10:53:55 CST | python3 | Accepted | 399 ms | 18.8 MB |
| 686549723 | 2025-12-23 10:16:08 CST | python3 | Runtime Error | N/A | N/A |
| 686549776 | 2025-12-23 10:16:24 CST | python3 | Wrong Answer | N/A | N/A |
| 686550920 | 2025-12-23 10:20:59 CST | python3 | Wrong Answer | N/A | N/A |
| 686551236 | 2025-12-23 10:22:13 CST | python3 | Accepted | 411 ms | 18.6 MB |

### 未通过提交代码
#### 提交 686204424 · Memory Limit Exceeded · 2025-12-21 17:33:02 CST · python3

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        deadends_set = set(deadends)
        start = '0000'
        if start in deadends_set:
            return -1
        if start == target:
            return 0
        dq = collections.deque([(start, 0)])
        visited = {start}
        while dq:
            curr_state, steps = dq.popleft()
            for i in range(4):
                digit = int(curr_state[i])
                for move in [1, -1]:
                    new_digit = (digit + move + 10) % 10
                    new_state = curr_state[:i] + str(new_digit) + curr_state[i + 1:]
                    if new_state in visited or new_state in deadends_set:
                        continue
                    if new_state == target:
                        return steps + 1
                    dq.append((new_state, steps + 1))
        return -1
```

#### 提交 686315012 · Wrong Answer · 2025-12-22 10:53:33 CST · python3

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        deadends_set = set(deadends)
        start = '0000'
        if start in deadends_set:
            return -1
        if start == target:
            return 0
        dq = collections.deque([(start, 0)])
        visited = {start}
        while dq:
            curr_state, steps = dq.popleft()
            if curr_state == target:
                return steps
            for i in range(4):
                digit = int(curr_state[i])
                for move in [1, -1]:
                    next_digit = (digit + 10 + move) % 10
                    new_state = curr_state[:i] + str(next_digit) + curr_state[i + 1:]
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    dq.append((new_state, steps + 1))
        return -1
```

#### 提交 686549723 · Runtime Error · 2025-12-23 10:16:08 CST · python3

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        start = '0000'
        deadends_set = set(deadends)
        if start == target:
            return 0
        dq = collections.deque([(start, 0)])
        visited = {start}
        while dq:
            curr_state, steps = dq.popleft()
            if curr_state == target:
                return steps
            for idx in range(4):
                digit = int(curr_state[idx])
                for move in [-1, 1]:
                    new_digit = (digit + move + 10) % 10
                    new_state = curr_state[:idx] + str(new_digit) + curr_state[i+1:]
                    if new_state in deadends_set:
                        continue
                    if new_state not in visited:
                        visited.add(new_state)
                        dq.append((new_state, steps + 1))
            return -1
```

#### 提交 686549776 · Wrong Answer · 2025-12-23 10:16:24 CST · python3

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        start = '0000'
        deadends_set = set(deadends)
        if start == target:
            return 0
        dq = collections.deque([(start, 0)])
        visited = {start}
        while dq:
            curr_state, steps = dq.popleft()
            if curr_state == target:
                return steps
            for idx in range(4):
                digit = int(curr_state[idx])
                for move in [-1, 1]:
                    new_digit = (digit + move + 10) % 10
                    new_state = curr_state[:idx] + str(new_digit) + curr_state[idx+1:]
                    if new_state in deadends_set:
                        continue
                    if new_state not in visited:
                        visited.add(new_state)
                        dq.append((new_state, steps + 1))
            return -1
```

#### 提交 686550920 · Wrong Answer · 2025-12-23 10:20:59 CST · python3

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        start = '0000'
        deadends_set = set(deadends)
        if start == target:
            return 0
        dq = collections.deque([(start, 0)])
        visited = {start}
        while dq:
            curr_state, steps = dq.popleft()
            if curr_state == target:
                return steps
            for idx in range(4):
                digit = int(curr_state[idx])
                for move in [-1, 1]:
                    new_digit = (digit + move + 10) % 10
                    new_state = curr_state[:idx] + str(new_digit) + curr_state[idx+1:]
                    if new_state in deadends_set:
                        continue
                    if new_state not in visited:
                        visited.add(new_state)
                        dq.append((new_state, steps + 1))
        return -1
```


## 滑动谜题 (`sliding-puzzle`)

- 题目链接：https://leetcode.cn/problems/sliding-puzzle/
- 难度：Hard
- 标签：广度优先搜索, 记忆化搜索, 数组, 动态规划, 回溯, 矩阵
- 总提交次数：8
- 最近提交时间：2025-12-23 10:06:50 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-22 10:43:07 CST

```markdown
这是一个在状态空间上求最短步数的问题，我用 BFS 做。

首先把 2x3 棋盘拉平成字符串，比如 "123405"，目标是 "123450"。

把所有可能的棋盘状态看成图上的点，每次移动 0 和它上下左右相邻的数交换，就是在图里走一条边。

因为每步代价相同，所以用 BFS 从起始状态开始搜索，第一次到达目标状态时的层数，就是最少步数。

实现上，我预先写好 2x3 棋盘上每个索引的邻居列表，用队列做 BFS，配合 visited 集合避免重复状态。如果 BFS 结束还没遇到目标，就返回 -1。

这个问题的状态最多只有 6! 种，所以 BFS 在时间和空间上都完全可行。

curr_list = list(curr_state) 的位置容易写错，应该写在 for 循环的里面
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686144820 | 2025-12-21 12:47:13 CST | python3 | Wrong Answer | N/A | N/A |
| 686144982 | 2025-12-21 12:48:22 CST | python3 | Wrong Answer | N/A | N/A |
| 686145136 | 2025-12-21 12:49:46 CST | python3 | Accepted | 7 ms | 17.4 MB |
| 686145771 | 2025-12-21 12:54:52 CST | python3 | Accepted | 7 ms | 17.1 MB |
| 686311770 | 2025-12-22 10:40:48 CST | python3 | Wrong Answer | N/A | N/A |
| 686312214 | 2025-12-22 10:42:30 CST | python3 | Accepted | 7 ms | 17.2 MB |
| 686547769 | 2025-12-23 10:06:34 CST | python3 | Runtime Error | N/A | N/A |
| 686547819 | 2025-12-23 10:06:50 CST | python3 | Accepted | 9 ms | 17.1 MB |

### 未通过提交代码
#### 提交 686144820 · Wrong Answer · 2025-12-21 12:47:13 CST · python3

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        start = ''.join(str(num) for row in board for num in row)
        steps = 0
        dq = collections.deque([(start, 0)])
        visited = {start}
        target = '012345'
        # 0 1 2
        # 3 4 5
        neighbors = [
            [1, 3], [0, 2, 4], [1, 5], [0, 4], [3, 1, 5], [4, 2]
        ]
        while dq:
            state, steps = dq.popleft()
            zero_idx = state.index('0')
            for next_idx in neighbors[zero_idx]:
                state_list = list(state)
                state_list[zero_idx], state_list[next_idx] = state_list[next_idx], state_list[zero_idx]
                new_state = ''.join(state_list)
                if new_state == target:
                    return steps + 1
                if new_state not in visited:
                    visited.add(new_state)
                    dq.append((new_state, steps + 1))
        return -1
```

#### 提交 686144982 · Wrong Answer · 2025-12-21 12:48:22 CST · python3

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        start = ''.join(str(num) for row in board for num in row)
        steps = 0
        dq = collections.deque([(start, 0)])
        visited = {start}
        target = '123450'
        # 0 1 2
        # 3 4 5
        neighbors = [
            [1, 3], [0, 2, 4], [1, 5], [0, 4], [3, 1, 5], [4, 2]
        ]
        while dq:
            state, steps = dq.popleft()
            zero_idx = state.index('0')
            for next_idx in neighbors[zero_idx]:
                state_list = list(state)
                state_list[zero_idx], state_list[next_idx] = state_list[next_idx], state_list[zero_idx]
                new_state = ''.join(state_list)
                if new_state == target:
                    return steps + 1
                if new_state not in visited:
                    visited.add(new_state)
                    dq.append((new_state, steps + 1))
        return -1
```

#### 提交 686311770 · Wrong Answer · 2025-12-22 10:40:48 CST · python3

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        start = ''.join(str(num) for row in board for num in row)
        target= '123450'
        # 0 1 2
        # 3 4 5
        neighbors = [
            [1, 3], [0, 4, 2], [1, 5], [0, 4], [3, 1, 5], [4, 2]
        ]
        dq = collections.deque([(start, 0)])
        visited = {start}
        while dq:
            curr_state, steps = dq.popleft()
            if curr_state == target:
                return steps
            zero_index = curr_state.index('0')
            curr_list = list(curr_state)
            for neighbor_index in neighbors[zero_index]:
                curr_list[neighbor_index], curr_list[zero_index] = curr_list[zero_index], curr_list[neighbor_index]
                new_state = ''.join(curr_list)
                if new_state in visited:
                    continue
                visited.add(new_state)
                dq.append((new_state, steps + 1))
        return -1
```

#### 提交 686547769 · Runtime Error · 2025-12-23 10:06:34 CST · python3

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        start = ''.join(str(num) for row in board for num in row)
        target = '123450'
        # 0 1 2
        # 3 4 5
        neighbors = [
            [1, 3], [0, 2, 4], [1, 5], [0, 4], [3, 1, 5], [4, 2]
        ]
        dq = collections.deque([(start, 0)])
        visited = {start}
        while dq:
            curr_state, steps = dq.popleft()
            if curr_state == target:
                return steps
            zero_index = curr_state.index('0')
            for neighbor_idx in neighbors[zero_index]:
                state_list = list(curr_state)
                state_list[zero_index], state_list[neighbor_idx] = state_list[neighbor_idx], state_list[zero_index]
                new_state = ''.join(state_list)
                if new_state not in visited:
                    visited.add(new_state)
                    dq.append(new_state, steps + 1)
        return -1
```


## 填充每个节点的下一个右侧节点指针 II (`populating-next-right-pointers-in-each-node-ii`)

- 题目链接：https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 链表, 二叉树
- 总提交次数：6
- 最近提交时间：2025-12-23 09:16:22 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-20 12:50:12 CST

```markdown
利用当前层已经建立好的 next 指针把这一层当作链表横向遍历，同时用一个虚拟头结点和尾指针在遍历过程中按顺序拼接出下一层的链表，因此无需队列等额外数据结构就实现了层序连接、空间 O(1)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685960072 | 2025-12-20 12:32:02 CST | python3 | Accepted | 51 ms | 18.1 MB |
| 685961972 | 2025-12-20 12:48:37 CST | python3 | Accepted | 50 ms | 18.1 MB |
| 686109535 | 2025-12-21 09:23:43 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 686109582 | 2025-12-21 09:24:44 CST | python3 | Accepted | 40 ms | 18 MB |
| 686292363 | 2025-12-22 09:09:56 CST | python3 | Accepted | 43 ms | 18.1 MB |
| 686538156 | 2025-12-23 09:16:22 CST | python3 | Accepted | 57 ms | 18 MB |

### 未通过提交代码
#### 提交 686109535 · Time Limit Exceeded · 2025-12-21 09:23:43 CST · python3

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
        curr = root
        while curr:
            dummy = Node()
            tail = dummy
            while curr:
                if curr.left:
                    tail.next = curr.left
                    tail = tail.next
                if curr.right:
                    tail.next = curr.left
                    tail = tail.next
                curr = curr.next
            curr = dummy.next
        return root
```


## 完全二叉树插入器 (`complete-binary-tree-inserter`)

- 题目链接：https://leetcode.cn/problems/complete-binary-tree-inserter/
- 难度：Medium
- 标签：树, 广度优先搜索, 设计, 二叉树
- 总提交次数：5
- 最近提交时间：2025-12-23 08:47:50 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-23 08:10:20 CST

```markdown
核心思路是利用一个队列来高效地找到下一次插入的位置，从而实现 O(1) 时间复杂度的插入。

在初始化时，我会对整棵树进行一次广度优先搜索（BFS），把所有孩子不满的节点，按顺序存入一个队列中。这个队列就成了我们的“候选父节点”池。
当插入新节点时，队头的节点就是新节点的父节点。我将新节点连接上去，并把这个新节点也加入队尾，因为它也成了新的候选父节点。
如果一个父节点因为这次插入而变满了（即左右孩子都有了），我就会将它从队头移除。
这种方法避免了每次插入都重新遍历树。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686289953 | 2025-12-22 08:39:46 CST | python3 | Wrong Answer | N/A | N/A |
| 686289999 | 2025-12-22 08:40:31 CST | python3 | Wrong Answer | N/A | N/A |
| 686290045 | 2025-12-22 08:41:23 CST | python3 | Accepted | 0 ms | 18.2 MB |
| 686535206 | 2025-12-23 08:46:05 CST | python3 | Wrong Answer | N/A | N/A |
| 686535337 | 2025-12-23 08:47:50 CST | python3 | Accepted | 3 ms | 18 MB |

### 未通过提交代码
#### 提交 686289953 · Wrong Answer · 2025-12-22 08:39:46 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class CBTInserter:

    def __init__(self, root: Optional[TreeNode]):
        self.root = root
        self.candidate_parents = collections.deque()
        dq = collections.deque([root])
        while dq:
            node = dq.popleft()
            if not node.left or node.right:
                self.candidate_parents.append(node)
            if node.left:
                dq.append(node.left)
            if node.right:
                dq.append(node.right)

    def insert(self, val: int) -> int:
        node = self.candidate_parents[0]
        new_node = TreeNode(val)
        if not node.left:
            node.left = new_node
        else:
            node.right = new_node
            self.candidate_parents.popleft()
        self.candidate_parents.append(new_node)
        

    def get_root(self) -> Optional[TreeNode]:
        return self.root
        


# Your CBTInserter object will be instantiated and called as such:
# obj = CBTInserter(root)
# param_1 = obj.insert(val)
# param_2 = obj.get_root()
```

#### 提交 686289999 · Wrong Answer · 2025-12-22 08:40:31 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class CBTInserter:

    def __init__(self, root: Optional[TreeNode]):
        self.root = root
        self.candidate_parents = collections.deque()
        dq = collections.deque([root])
        while dq:
            node = dq.popleft()
            if not node.left or node.right:
                self.candidate_parents.append(node)
            if node.left:
                dq.append(node.left)
            if node.right:
                dq.append(node.right)

    def insert(self, val: int) -> int:
        node = self.candidate_parents[0]
        new_node = TreeNode(val)
        if not node.left:
            node.left = new_node
        else:
            node.right = new_node
            self.candidate_parents.popleft()
        self.candidate_parents.append(new_node)
        return node.val
        

    def get_root(self) -> Optional[TreeNode]:
        return self.root
        


# Your CBTInserter object will be instantiated and called as such:
# obj = CBTInserter(root)
# param_1 = obj.insert(val)
# param_2 = obj.get_root()
```

#### 提交 686535206 · Wrong Answer · 2025-12-23 08:46:05 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class CBTInserter:

    def __init__(self, root: Optional[TreeNode]):
        self.root = root
        self.candidate_parents = collections.deque()
        dq = collections.deque([root])
        while dq:
            node = dq.popleft()
            if not node.left or not node.right:
                self.candidate_parents.append(node)
            if node.left:
                dq.append(node.left)
            if node.right:
                dq.append(node.right)


    def insert(self, val: int) -> int:
        new_node = TreeNode(val)
        parent = self.candidate_parents[0]
        if not parent.left:
            parent.left = new_node
        else:
            parent.right = new_node
            self.candidate_parents.popleft()
        self.candidate_parents.append(new_node)
        

    def get_root(self) -> Optional[TreeNode]:
        return self.root
        


# Your CBTInserter object will be instantiated and called as such:
# obj = CBTInserter(root)
# param_1 = obj.insert(val)
# param_2 = obj.get_root()
```


## 二叉树最大宽度 (`maximum-width-of-binary-tree`)

- 题目链接：https://leetcode.cn/problems/maximum-width-of-binary-tree/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：6
- 最近提交时间：2025-12-22 10:21:20 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-20 17:22:46 CST

```markdown
整体是标准 BFS，一次遍历所有节点，时间复杂度 O(N)，空间 O(N)，关键点就是“用完全二叉树下标模拟空位，再用当前层队头和队尾下标之差当宽度”。
队列里存 (节点, 下标)，下标按完全二叉树规则：根是 1，左子 2i，右子 2i+1。这样即使中间有空位，也会体现在下标差里。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686018344 | 2025-12-20 17:27:55 CST | python3 | Wrong Answer | N/A | N/A |
| 686018593 | 2025-12-20 17:28:59 CST | python3 | Wrong Answer | N/A | N/A |
| 686018682 | 2025-12-20 17:29:28 CST | python3 | Accepted | 3 ms | 17.8 MB |
| 686121579 | 2025-12-21 10:49:54 CST | python3 | Accepted | 7 ms | 18.1 MB |
| 686306863 | 2025-12-22 10:20:57 CST | python3 | Runtime Error | N/A | N/A |
| 686306952 | 2025-12-22 10:21:20 CST | python3 | Accepted | 3 ms | 17.8 MB |

### 未通过提交代码
#### 提交 686018344 · Wrong Answer · 2025-12-20 17:27:55 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        max_width = 0
        dq = collections.deque([(root, 1)])
        while dq:
            sz = len(dq)
            first_index = dq[0][1]
            last_index = dq[-1][1]
            node, idx = dq.popleft()
            if node.left:
                dq.append((node.left, idx * 2))
            if node.right:
                dq.append((node.right, idx * 2 + 1))
        max_width = max(max_width, last_index - first_index + 1)
    
        return max_width
```

#### 提交 686018593 · Wrong Answer · 2025-12-20 17:28:59 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        max_width = 0
        dq = collections.deque([(root, 1)])
        while dq:
            sz = len(dq)
            first_index = dq[0][1]
            last_index = dq[-1][1]
            for _ in range(sz):
                node, idx = dq.popleft()
                if node.left:
                    dq.append((node.left, idx * 2))
                if node.right:
                    dq.append((node.right, idx * 2 + 1))
        max_width = max(max_width, last_index - first_index + 1)
    
        return max_width
```

#### 提交 686306863 · Runtime Error · 2025-12-22 10:20:57 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        max_width = 0
        dq = collections.deque([(root, 1)])
        while dq:
            sz = len(dq)
            first_index = dq[0][1]
            last_index = dq[-1][1]
            for _ in range(sz):
                node, idx = dq.popleft()
                if node.left:
                    dq.append(node.left, 2 * idx)
                if node.right:
                    dq.append(node.right, 2 * idx + 1)
            max_width = max(max_width, last_index - first_index + 1)
        return max_width
```


## 奇偶树 (`even-odd-tree`)

- 题目链接：https://leetcode.cn/problems/even-odd-tree/
- 难度：Medium
- 标签：树, 广度优先搜索, 二叉树
- 总提交次数：4
- 最近提交时间：2025-12-21 11:11:56 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686059368 | 2025-12-20 21:40:47 CST | python3 | Wrong Answer | N/A | N/A |
| 686059998 | 2025-12-20 21:44:02 CST | python3 | Runtime Error | N/A | N/A |
| 686060039 | 2025-12-20 21:44:12 CST | python3 | Accepted | 51 ms | 43.6 MB |
| 686126615 | 2025-12-21 11:11:56 CST | python3 | Accepted | 55 ms | 43.6 MB |

### 未通过提交代码
#### 提交 686059368 · Wrong Answer · 2025-12-20 21:40:47 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        dq = collections.deque([root])
        level = 0
        while dq:
            if level % 2 == 0:
                prev = float('-inf')
            else:
                prev = float('inf')
            sz = len(dq)
            for _ in range(sz):
                node = dq.popleft()
                if level % 2 == 0 and node.val <= prev:
                    return False
                if level % 2 != 0 and node.val >= prev:
                    return False
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
                prev = node.val
            level += 1
        return True
```

#### 提交 686059998 · Runtime Error · 2025-12-20 21:44:02 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        dq = collections.deque([root])
        level = 0
        while dq:
            if level % 2 == 0:
                prev = float('-inf')
            else:
                prev = float('inf')
            sz = len(dq)
            for _ in range(sz):
                node = dq.popleft()
                if level % 2 == 0:
                    if node.val % 2 == 0 or node.val <= prev:
                        return False
                if level % 2 == 1:
                    if node.val % 2 == 1 or node.val >= prev
                        return False
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
                prev = node.val
            level += 1
        return True
```


## 最大层内元素和 (`maximum-level-sum-of-a-binary-tree`)

- 题目链接：https://leetcode.cn/problems/maximum-level-sum-of-a-binary-tree/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：2
- 最近提交时间：2025-12-21 11:01:30 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686054588 | 2025-12-20 21:16:32 CST | python3 | Accepted | 18 ms | 21.2 MB |
| 686124246 | 2025-12-21 11:01:30 CST | python3 | Accepted | 15 ms | 21.1 MB |

### 未通过提交代码
(所有提交均已通过)

## 层数最深叶子节点的和 (`deepest-leaves-sum`)

- 题目链接：https://leetcode.cn/problems/deepest-leaves-sum/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：1
- 最近提交时间：2025-12-20 21:24:02 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686056097 | 2025-12-20 21:24:02 CST | python3 | Accepted | 15 ms | 19.9 MB |

### 未通过提交代码
(所有提交均已通过)

## 二叉树的完全性检验 (`check-completeness-of-a-binary-tree`)

- 题目链接：https://leetcode.cn/problems/check-completeness-of-a-binary-tree/
- 难度：Medium
- 标签：树, 广度优先搜索, 二叉树
- 总提交次数：1
- 最近提交时间：2025-12-20 21:00:49 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-20 18:38:01 CST

```markdown
用层序遍历来判断完全二叉树。我用一个队列，从根节点开始往外一层层遍历。遍历的时候允许遇到空节点，一旦遇到空节点我打一个标记，从这之后按照完全二叉树的性质，就不应该再出现任何非空节点了：如果在标记之后还看到了非空节点，就说明树的中间存在“空洞”，即不是完全二叉树，直接返回 False。如果遍历完都没违反这个条件，就返回 True。时间复杂度 O(n)，空间 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686051592 | 2025-12-20 21:00:49 CST | python3 | Accepted | 4 ms | 17.1 MB |

### 未通过提交代码
(所有提交均已通过)

## 二叉树的层平均值 (`average-of-levels-in-binary-tree`)

- 题目链接：https://leetcode.cn/problems/average-of-levels-in-binary-tree/
- 难度：Easy
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：1
- 最近提交时间：2025-12-20 18:28:29 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686028063 | 2025-12-20 18:28:29 CST | python3 | Accepted | 3 ms | 18.8 MB |

### 未通过提交代码
(所有提交均已通过)

## 在每个树行中找最大值 (`find-largest-value-in-each-tree-row`)

- 题目链接：https://leetcode.cn/problems/find-largest-value-in-each-tree-row/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：1
- 最近提交时间：2025-12-20 18:24:27 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 686027519 | 2025-12-20 18:24:27 CST | python3 | Accepted | 7 ms | 18.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 二叉树的锯齿形层序遍历 (`binary-tree-zigzag-level-order-traversal`)

- 题目链接：https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/
- 难度：Medium
- 标签：树, 广度优先搜索, 二叉树
- 总提交次数：3
- 最近提交时间：2025-12-20 12:26:42 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685830927 | 2025-12-19 16:10:26 CST | python3 | Runtime Error | N/A | N/A |
| 685831111 | 2025-12-19 16:11:04 CST | python3 | Accepted | 0 ms | 17.1 MB |
| 685959425 | 2025-12-20 12:26:42 CST | python3 | Accepted | 0 ms | 17.4 MB |

### 未通过提交代码
#### 提交 685830927 · Runtime Error · 2025-12-19 16:10:26 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res = []
        dq = collections.deque([root])
        left_to_right = True
        while dq:
            current_size = len(dq)
            current_res = []
            for _ in range(current_size):
                current_res.append(node.val)
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
            if not left_to_right:
                current_res.reverse()
            res.append(current_res)
            left_to_right = not left_to_right
        return res
```


## 二叉树的层序遍历 II (`binary-tree-level-order-traversal-ii`)

- 题目链接：https://leetcode.cn/problems/binary-tree-level-order-traversal-ii/
- 难度：Medium
- 标签：树, 广度优先搜索, 二叉树
- 总提交次数：2
- 最近提交时间：2025-12-20 12:04:41 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685956631 | 2025-12-20 12:03:08 CST | python3 | Runtime Error | N/A | N/A |
| 685956812 | 2025-12-20 12:04:41 CST | python3 | Accepted | 0 ms | 17.3 MB |

### 未通过提交代码
#### 提交 685956631 · Runtime Error · 2025-12-20 12:03:08 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res = []
        dq = collections.deque([root])
        while dq:
            level_size = len(dq)
            level_nodes = []
            for _ in range(level_size):
                node = dq.popleft()
                level_nodes.append(node.val)
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
            res.appendleft(level_nodes)
        return res
```


## 爬楼梯 (`climbing-stairs`)

- 题目链接：https://leetcode.cn/problems/climbing-stairs/
- 难度：Easy
- 标签：记忆化搜索, 数学, 动态规划
- 总提交次数：1
- 最近提交时间：2025-12-19 15:59:44 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685827884 | 2025-12-19 15:59:44 CST | python3 | Accepted | 0 ms | 17 MB |

### 未通过提交代码
(所有提交均已通过)

## 匹配子序列的单词数 (`number-of-matching-subsequences`)

- 题目链接：https://leetcode.cn/problems/number-of-matching-subsequences/
- 难度：Medium
- 标签：字典树, 数组, 哈希表, 字符串, 二分查找, 动态规划, 排序
- 总提交次数：4
- 最近提交时间：2025-12-19 14:49:01 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-17 14:02:09 CST

```markdown
这道题本质上是解决多模式串匹配问题。
我没有选择让每个 Pattern 去轮询 Text，而是利用倒排索引的思想，根据 Pattern 当前期待的字符进行归并。
这样将 Text 视为事件流，触发 Bucket 里等待的 Pattern 进行状态转移，从而将时间复杂度从笛卡尔积（乘法）降维到了线性（加法）。

这题本质上考的是两点： 1）能不能识别暴力解法中对 s 的重复扫描，并通过反转循环顺序，把 ‘对每个 word 扫 s’ 变成 ‘只扫一次 s，同时推进所有 word 的匹配’； 2）能不能设计一个合适的数据结构，把每个 word 当前的匹配位置抽象成状态，并按它‘下一步需要的字符’分桶，变成类似事件驱动的模型——s 中字符出现一次，就把所有等它的任务推进一步。
这样做的好处是把时间复杂度从近似 s_len * words_len 降到 O(len(s) + 所有 word 总长度)，体现的是消灭重复工作、按需处理的算法思维。

先按所有单词的首字母进行分桶，首字母相同的单词都在同一个桶中
开始遍历 s 中的字符，当前正在遍历字符对应的桶中的单词就是所有含有这个字符的单词，然后使用迭代器将当前桶中的单词取出下一个字符，再根据新取出的字符重新分桶，如果遇到哪个单词的迭代器用完了，就说明这个单词是一个匹配的单词

时间复杂度： O(L + N)。其中 L 是 s 的长度，N 是所有 words 字符的总数。我们只扫描了 s 一次，且 words 中的每个字符最多被处理一次。
空间复杂度： O(N)。最坏情况下，我们要存储所有单词的迭代器。

这就好比你在食堂打饭（s 是窗口），学生（words）想要不同的菜。厨师喊一声“红烧肉好了”（s 中的字符），手里拿着盘子等红烧肉的学生（在 bucket 里的迭代器）就上前一步，装完后去排下一个菜的队。所有菜打完的学生就算完成了。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682358570 | 2025-12-03 12:41:07 CST | python3 | Accepted | 255 ms | 19.2 MB |
| 685352240 | 2025-12-17 13:57:39 CST | python3 | Accepted | 243 ms | 19.4 MB |
| 685534452 | 2025-12-18 10:06:50 CST | python3 | Accepted | 231 ms | 18.7 MB |
| 685808226 | 2025-12-19 14:49:01 CST | python3 | Accepted | 243 ms | 19.1 MB |

### 未通过提交代码
(所有提交均已通过)

## 基本计算器 (`basic-calculator`)

- 题目链接：https://leetcode.cn/problems/basic-calculator/
- 难度：Hard
- 标签：栈, 递归, 数学, 字符串
- 总提交次数：23
- 最近提交时间：2025-12-19 14:26:42 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-08 10:50:44 CST

```markdown
在扫描字符串时维护三个变量：res 是当前累计结果，num 是正在读的数字，sign 是这个数字前面的符号（+1 或 -1）。
每当遇到 +、-、) 或字符串结束，就说明前面的数字已经结束了，我用 res += sign * num 把它结算进去，然后把 num 置 0。
对于刚读到的 + 或 -，只是用来更新下一次数字的 sign。这样所有的减法都被转成了‘加上一个带符号的数字。


这题本质是“线性扫描 + 栈管理上下文”的通用模式。

我把复杂的表达式简化成三个状态：当前结果、当前数字、当前符号，一次扫描字符串；

遇到数字就累积，遇到运算符或右括号就把之前的数字按符号结算；

遇到左括号就把当前结果和符号压栈，相当于保存现场，开始算子表达式；

遇到右括号则把子表达式的结果乘以栈里保存的符号再加回去。

这个框架其实是通用的：所有“嵌套结构 + 表达式/状态”的题，都可以用‘状态机 + 栈保存上下文’这个模式来解。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 683363406 | 2025-12-08 10:57:53 CST | python3 | Wrong Answer | N/A | N/A |
| 683364154 | 2025-12-08 11:00:20 CST | python3 | Accepted | 47 ms | 18 MB |
| 683394034 | 2025-12-08 13:46:11 CST | python3 | Wrong Answer | N/A | N/A |
| 683394189 | 2025-12-08 13:47:05 CST | python3 | Wrong Answer | N/A | N/A |
| 683394299 | 2025-12-08 13:47:43 CST | python3 | Accepted | 47 ms | 18.3 MB |
| 683396385 | 2025-12-08 13:59:09 CST | python3 | Runtime Error | N/A | N/A |
| 683396439 | 2025-12-08 13:59:31 CST | python3 | Wrong Answer | N/A | N/A |
| 683396632 | 2025-12-08 14:00:35 CST | python3 | Accepted | 39 ms | 18.1 MB |
| 684118147 | 2025-12-11 14:59:21 CST | python3 | Wrong Answer | N/A | N/A |
| 684118637 | 2025-12-11 15:00:57 CST | python3 | Wrong Answer | N/A | N/A |
| 684118726 | 2025-12-11 15:01:14 CST | python3 | Wrong Answer | N/A | N/A |
| 684118862 | 2025-12-11 15:01:41 CST | python3 | Wrong Answer | N/A | N/A |
| 684119578 | 2025-12-11 15:04:05 CST | python3 | Accepted | 43 ms | 18 MB |
| 684119632 | 2025-12-11 15:04:12 CST | python3 | Wrong Answer | N/A | N/A |
| 684119704 | 2025-12-11 15:04:23 CST | python3 | Accepted | 39 ms | 18.1 MB |
| 684120419 | 2025-12-11 15:06:37 CST | python3 | Accepted | 45 ms | 18.1 MB |
| 684122399 | 2025-12-11 15:12:26 CST | python3 | Wrong Answer | N/A | N/A |
| 684122625 | 2025-12-11 15:13:04 CST | python3 | Wrong Answer | N/A | N/A |
| 684122720 | 2025-12-11 15:13:21 CST | python3 | Accepted | 47 ms | 18.2 MB |
| 685541242 | 2025-12-18 10:34:19 CST | python3 | Accepted | 47 ms | 18 MB |
| 685802024 | 2025-12-19 14:23:48 CST | python3 | Wrong Answer | N/A | N/A |
| 685802198 | 2025-12-19 14:24:34 CST | python3 | Wrong Answer | N/A | N/A |
| 685802662 | 2025-12-19 14:26:42 CST | python3 | Accepted | 43 ms | 18.2 MB |

### 未通过提交代码
#### 提交 683363406 · Wrong Answer · 2025-12-08 10:57:53 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)  # 可能是多位数字
            elif ch == '+' or ch == '-':
                # 对之前的数字进行结算
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                sign = 1
            elif ch == ')':
                res += sign * num
                pre_sign = stack.pop()
                pre_res = stack.pop()
                res = pre_res + pre_sign * res
                num = 0
            else:
                continue
        res += sign * num  # 如果最后的一位是数字，需要加上
        return res
```

#### 提交 683394034 · Wrong Answer · 2025-12-08 13:46:11 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                sign = 1
            elif ch == ')':
                res += sign * num
                pre_sign = stack.pop()
                pre_res = stack.pop()
                res += pre_res + pre_sign * res
                num = 0
            else:
                continue
        res += sign * num
        return res
```

#### 提交 683394189 · Wrong Answer · 2025-12-08 13:47:05 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                sign = 1
                res = 0
            elif ch == ')':
                res += sign * num
                pre_sign = stack.pop()
                pre_res = stack.pop()
                res += pre_res + pre_sign * res
                num = 0
            else:
                continue
        res += sign * num
        return res
```

#### 提交 683396385 · Runtime Error · 2025-12-08 13:59:09 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0  # 正在被读取的数字
        sign = 1  # 正在被读取的数字的符号
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)  # 处理多位数字的情况
            elif ch == '+' or ch == '-':
                res += sign * num  # 进行一次结算
                num = 0
                sign = 1 if ch == '+' else -1  # 为后面的计算重置
            elif ch == '(':  # 进入下一层计算
                stack.append(sign)
                res = 0
                num = 0
                sign = 1
            elif ch == ')':
                res += sign * num  # 对括号内的计算进行结算
                pre_sign = stack.pop()
                pre_res = stack.pop()
                res = pre_res + pre_sign * res
                num = 0
            else:
                continue
        res += sign * num  # 最后一位是数字的情况，不要漏掉
        return res
```

#### 提交 683396439 · Wrong Answer · 2025-12-08 13:59:31 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0  # 正在被读取的数字
        sign = 1  # 正在被读取的数字的符号
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)  # 处理多位数字的情况
            elif ch == '+' or ch == '-':
                res += sign * num  # 进行一次结算
                num = 0
                sign = 1 if ch == '+' else -1  # 为后面的计算重置
            elif ch == '(':  # 进入下一层计算
                stack.append(sign)
                stack.append(res)
                res = 0
                num = 0
                sign = 1
            elif ch == ')':
                res += sign * num  # 对括号内的计算进行结算
                pre_sign = stack.pop()
                pre_res = stack.pop()
                res = pre_res + pre_sign * res
                num = 0
            else:
                continue
        res += sign * num  # 最后一位是数字的情况，不要漏掉
        return res
```

#### 提交 684118147 · Wrong Answer · 2025-12-11 14:59:21 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                res = 0
            elif ch == ')':
                prev_sign = stack.pop()
                prev_res = stack.pop()
                res = prev_res + prev_sign * res
                num = 0
        return res
```

#### 提交 684118637 · Wrong Answer · 2025-12-11 15:00:57 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                res = 0
            elif ch == ')':
                prev_sign = stack.pop()
                prev_res = stack.pop()
                res = prev_res + prev_sign * res
                num = 0
        res += sign * num
        return res
```

#### 提交 684118726 · Wrong Answer · 2025-12-11 15:01:14 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                res = 0
                sign = 1
            elif ch == ')':
                prev_sign = stack.pop()
                prev_res = stack.pop()
                res = prev_res + prev_sign * res
                num = 0
        res += sign * num
        return res
```

#### 提交 684118862 · Wrong Answer · 2025-12-11 15:01:41 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                res = 0
                sign = 1
            elif ch == ')':
                prev_sign = stack.pop()
                prev_res = stack.pop()
                res = prev_res + prev_sign * res
                num = 0
            else:
                continue
        res += sign * num
        return res
```

#### 提交 684119632 · Wrong Answer · 2025-12-11 15:04:12 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                res = 0
                # sign = 1
            elif ch == ')':
                res += sign * num
                prev_sign = stack.pop()
                prev_res = stack.pop()
                res = prev_res + prev_sign * res
                num = 0
            else:
                continue
        res += sign * num
        return res
```

#### 提交 684122399 · Wrong Answer · 2025-12-11 15:12:26 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(num)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else 1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                res = 0
                num = 0
                sign = 1
            elif ch == ')':
                res += sign * num
                pre_sign = stack.pop()
                pre_res = stack.pop()
                res = pre_res + pre_sign * res
                num = 0
                sign = 1
            else:
                continue
        res += sign * num
        return res
```

#### 提交 684122625 · Wrong Answer · 2025-12-11 15:13:04 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(num)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                res = 0
                num = 0
                sign = 1
            elif ch == ')':
                res += sign * num
                pre_sign = stack.pop()
                pre_res = stack.pop()
                res = pre_res + pre_sign * res
                num = 0
                sign = 1
            else:
                continue
        res += sign * num
        return res
```

#### 提交 685802024 · Wrong Answer · 2025-12-19 14:23:48 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        num = 0
        res = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                sign = 1
            elif ch == ')':
                res += sign * num
                pre_sign = stack.pop()
                pre_res = stack.pop()
                num = 0
            else:
                continue
        res += num * sign
        return res
```

#### 提交 685802198 · Wrong Answer · 2025-12-19 14:24:34 CST · python3

```python
class Solution:
    def calculate(self, s: str) -> int:
        num = 0
        res = 0
        sign = 1
        stack = []
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '+' or ch == '-':
                res += sign * num
                num = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                num = 0
                sign = 1
            elif ch == ')':
                res += sign * num
                pre_sign = stack.pop()
                pre_res = stack.pop()
                res = pre_sign * res + pre_res
                num = 0
            else:
                continue
        res += num * sign
        return res
```


## 数组中的第K个最大元素 (`kth-largest-element-in-an-array`)

- 题目链接：https://leetcode.cn/problems/kth-largest-element-in-an-array/
- 难度：Medium
- 标签：数组, 分治, 快速选择, 排序, 堆（优先队列）
- 总提交次数：10
- 最近提交时间：2025-12-19 14:17:21 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-19 14:38:31 CST

```markdown
一个关键直觉：第 K 大的核心是“门槛”而不是“全序”。小顶堆把“门槛”暴露在堆顶，决定权变成了“是否超过门槛”的局部比较，而不是“把所有元素排好队”。

面试口述要点（30 秒）

* 目标是维护前 K 大集合的“门槛”。用大小为 K 的小顶堆，堆顶就是当前门槛。遍历数组：若元素大于堆顶就替换并下滤，否则跳过。这样时间 O(N log K)、空间 O(K)，对流式/海量数据也适用，性能稳定；K ≪ N 时比排序 O(N log N) 划算。堆里最后的堆顶就是第 K 大，同时还能顺便得到 Top K 列表。

heapq.heapify(x) 是一个原地（In-place）操作，它直接修改传入的列表，返回值为 None，所以不能写成 min_heap = heapq.heapify(nums[:k])
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 676636771 | 2025-11-08 11:21:35 CST | python3 | Accepted | 47 ms | 28.3 MB |
| 679193285 | 2025-11-19 14:28:06 CST | python3 | Runtime Error | N/A | N/A |
| 679193837 | 2025-11-19 14:29:54 CST | python3 | Runtime Error | N/A | N/A |
| 679194129 | 2025-11-19 14:30:50 CST | python3 | Runtime Error | N/A | N/A |
| 679194454 | 2025-11-19 14:31:57 CST | python3 | Runtime Error | N/A | N/A |
| 679194529 | 2025-11-19 14:32:11 CST | python3 | Accepted | 52 ms | 28.1 MB |
| 685542051 | 2025-12-18 10:37:23 CST | python3 | Accepted | 39 ms | 28.5 MB |
| 685800513 | 2025-12-19 14:16:50 CST | python3 | Runtime Error | N/A | N/A |
| 685800562 | 2025-12-19 14:17:04 CST | python3 | Accepted | 55 ms | 28.3 MB |
| 685800614 | 2025-12-19 14:17:21 CST | python3 | Accepted | 57 ms | 28.2 MB |

### 未通过提交代码
#### 提交 679193285 · Runtime Error · 2025-11-19 14:28:06 CST · python3

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = heapq.heapify(nums[:k])
        for i in range(k, len(nums)):
            if nums[i] > min_heap[0]:
                heapq.heapreaplace(min_heap, nums[i])
        return min_heap[0]
```

#### 提交 679193837 · Runtime Error · 2025-11-19 14:29:54 CST · python3

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = heapq.heapify(nums[:k])
        for num in nums[k:]:
            if nums > min_heap[0]:
                heapq.heapreaplace(min_heap, num)
        return min_heap[0]
```

#### 提交 679194129 · Runtime Error · 2025-11-19 14:30:50 CST · python3

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = heapq.heapify(nums[:k])
        for num in nums[k:]:
            if num > min_heap[0]:
                heapq.heapreaplace(min_heap, num)
        return min_heap[0]
```

#### 提交 679194454 · Runtime Error · 2025-11-19 14:31:57 CST · python3

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = nums[:k]
        heapq.heapify(min_heap)
        for num in nums[k:]:
            if num > min_heap[0]:
                heapq.heapreaplace(min_heap, num)
        return min_heap[0]
```

#### 提交 685800513 · Runtime Error · 2025-12-19 14:16:50 CST · python3

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = nums[:k]
        heapq.heapify(min_heap)
        for num in nums[k:]:
            if num > min_heap[0]:
                heapq.heappushpop(min_heap)
        return min_heap[0]
```


## 平衡二叉树 (`balanced-binary-tree`)

- 题目链接：https://leetcode.cn/problems/balanced-binary-tree/
- 难度：Easy
- 标签：树, 深度优先搜索, 二叉树
- 总提交次数：8
- 最近提交时间：2025-12-19 14:03:10 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-16 11:13:04 CST

```markdown
这题要判断一棵二叉树是不是高度平衡，要求每个节点左右子树高度差不超过 1。

如果直接对每个节点都去算左右子树高度，会重复计算，高度函数是递归的，整体会到 O(n²)。

更好的方法是用后序遍历，自底向上，一次递归同时返回两个信息：这棵子树是不是平衡的，以及它的高度。

对每个节点，先递归算出左右子树的 (平衡状态, 高度)，如果任一子树不平衡就直接返回不平衡；否则再看左右高度差是否大于 1。

每个节点只访问一次，时间复杂度 O(n)，空间是递归栈 O(h)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685078262 | 2025-12-16 11:16:33 CST | python3 | Runtime Error | N/A | N/A |
| 685078536 | 2025-12-16 11:17:27 CST | python3 | Runtime Error | N/A | N/A |
| 685078656 | 2025-12-16 11:17:52 CST | python3 | Wrong Answer | N/A | N/A |
| 685079204 | 2025-12-16 11:19:51 CST | python3 | Wrong Answer | N/A | N/A |
| 685079399 | 2025-12-16 11:20:30 CST | python3 | Accepted | 0 ms | 18.5 MB |
| 685548615 | 2025-12-18 10:59:55 CST | python3 | Runtime Error | N/A | N/A |
| 685548668 | 2025-12-18 11:00:04 CST | python3 | Accepted | 0 ms | 18 MB |
| 685797734 | 2025-12-19 14:03:10 CST | python3 | Accepted | 0 ms | 18.1 MB |

### 未通过提交代码
#### 提交 685078262 · Runtime Error · 2025-12-16 11:16:33 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        is_balanced, _ = self._check(root)
        return is_balanced
    
    def _check(self, node):
        if not node:
            return True, 0
        left_balanced, left_height = self._check(node.left)
        if not left_balanced:
            return False, 0
        right_balanced, right_height = self._check(node.right)
        if not right_balanced:
            return False, 0
        if abs(left_balanced - right_balanced) > 1:
            return False, 0
        return True, max(left_balanced + right_balanced) + 1
```

#### 提交 685078536 · Runtime Error · 2025-12-16 11:17:27 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        is_balanced, _ = self._check(root)
        return is_balanced
    
    def _check(self, node):
        if not node:
            return True, 0
        left_balanced, left_height = self._check(node.left)
        if not left_balanced:
            return False, 0
        right_balanced, right_height = self._check(node.right)
        if not right_balanced:
            return False, 0
        if abs(left_balanced - right_balanced) > 1:
            return False, 0
        height = max(left_balanced + right_balanced) + 1
        return True, height
```

#### 提交 685078656 · Wrong Answer · 2025-12-16 11:17:52 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        is_balanced, _ = self._check(root)
        return is_balanced
    
    def _check(self, node):
        if not node:
            return True, 0
        left_balanced, left_height = self._check(node.left)
        if not left_balanced:
            return False, 0
        right_balanced, right_height = self._check(node.right)
        if not right_balanced:
            return False, 0
        if abs(left_balanced - right_balanced) > 1:
            return False, 0
        return True, max(left_balanced, right_balanced) + 1
```

#### 提交 685079204 · Wrong Answer · 2025-12-16 11:19:51 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        is_balanced, _ = self._check(root)
        return is_balanced
    
    def _check(self, node):
        if not node:
            return True, 0
        left_balanced, left_height = self._check(node.left)
        if not left_balanced:
            return False, 0
        right_balanced, right_height = self._check(node.right)
        if not right_balanced:
            return False, 0
        if abs(left_height - right_height) > 1:
            return False, 0
        return True, max(left_balanced, right_balanced) + 1
```

#### 提交 685548615 · Runtime Error · 2025-12-18 10:59:55 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        is_balanced, _ = self._check(root)
        return is_balanced
    
    def _check(self, node):
        if not node:
            return True, 0
        left_balanced, left_height = self._check(node.left)
        if not left_balanced:
            return False, 0
        right_balanced, right_height = self._check(node.right)
        if not right_balanced:
            return False, 0
        if abs(left_height - right_height) > 1:
            reutrn False, 0
        return True, max(left_height, right_height) + 1
```


## 反转链表 II (`reverse-linked-list-ii`)

- 题目链接：https://leetcode.cn/problems/reverse-linked-list-ii/
- 难度：Medium
- 标签：链表
- 总提交次数：5
- 最近提交时间：2025-12-19 13:56:43 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-11 13:52:36 CST

```markdown
迭代 + 头插法
先找到待反转区域的前一个节点（pre 指针走 left - 1 步）
再用头插法执行 (right - left次)
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682052295 | 2025-12-02 08:53:18 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 684107730 | 2025-12-11 14:21:19 CST | python3 | Wrong Answer | N/A | N/A |
| 684107820 | 2025-12-11 14:21:38 CST | python3 | Accepted | 0 ms | 17.8 MB |
| 685552831 | 2025-12-18 11:14:38 CST | python3 | Accepted | 0 ms | 17.1 MB |
| 685796587 | 2025-12-19 13:56:43 CST | python3 | Accepted | 0 ms | 17.2 MB |

### 未通过提交代码
#### 提交 684107730 · Wrong Answer · 2025-12-11 14:21:19 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(-1, head)
        pre = dummy
        for _ in range(left - 1):
            pre = pre.next
        start = pre
        for _ in range(right - left):
            next_node = start.next
            start.next =  next_node.next
            next_node.next = pre.next
            pre.next = next_node
        return dummy.next
```


## 合并 K 个升序链表 (`merge-k-sorted-lists`)

- 题目链接：https://leetcode.cn/problems/merge-k-sorted-lists/
- 难度：Hard
- 标签：链表, 分治, 堆（优先队列）, 归并排序
- 总提交次数：8
- 最近提交时间：2025-12-19 13:52:52 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-02 10:49:18 CST

```markdown
1、链表节点不能直接入堆，一般以元组的形式入堆，比如(node.value, idx, node)，也将idx入堆的原因是为了避免节点value相等时无法比较大小的问题
2、优先级队列heapq入堆的用法：
```
min_heap: list[tuple[int, int, LinkNode]] = [] # 先声明一个空列表
heapq.heappush(min_heap, (node.value, idx, node))
```


用一个大小为 k 的最小堆，初始把每个链表的头结点放进去。
每次从堆里取出最小的节点接到结果链表后面，再把它的下一个节点压入堆。
堆里最多只有 k 个元素，所以每次 push/pop 是 log k。
每个节点只会被 push 和 pop 一次，所以总共是 N 次堆操作。
时间复杂度是 O(N log k)，空间复杂度是堆的大小 O(k)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 645378519 | 2025-07-20 09:40:10 CST | python3 | Accepted | 11 ms | 20 MB |
| 682077990 | 2025-12-02 10:46:37 CST | python3 | Runtime Error | N/A | N/A |
| 682078111 | 2025-12-02 10:46:59 CST | python3 | Accepted | 12 ms | 20 MB |
| 683581696 | 2025-12-09 10:06:55 CST | python3 | Runtime Error | N/A | N/A |
| 683581845 | 2025-12-09 10:07:32 CST | python3 | Accepted | 15 ms | 20 MB |
| 685555175 | 2025-12-18 11:22:32 CST | python3 | Runtime Error | N/A | N/A |
| 685555257 | 2025-12-18 11:22:48 CST | python3 | Accepted | 11 ms | 19.3 MB |
| 685795941 | 2025-12-19 13:52:52 CST | python3 | Accepted | 8 ms | 19.6 MB |

### 未通过提交代码
#### 提交 682077990 · Runtime Error · 2025-12-02 10:46:37 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        min_heap: list[tuple[int, int, ListNode]] = []
        dummy = ListNode()
        p_tail = dummy
        for idx, node in enumerate(lists):
            if node:
                heapq.heappush(min_heap, (node.val, idx, node))
        
        while min_heap:
            node = heapq.heappop(min_heap)
            p_tail.next = node
            p_tail = p_tail.next
            if node.next:
                heapq.heappush(min_heap, (node.next.val, idx, node.next))
        return dummy.next
```

#### 提交 683581696 · Runtime Error · 2025-12-09 10:06:55 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        min_heap: list[tuple(int, int, ListNode)] = []
        for idx, node in lists:
            if node:
                heapq.heappush(min_heap, (node.val, idx, node))
        
        dummy = ListNode()
        p_tail = dummy
        
        while min_heap:
            _, idx, node = heapq.heappop(min_heap)
            p_tail.next = node
            p_tail = p_tail.next
            if node.next:
                heapq.heappush(min_heap, (node.next.val, idx, node.next))
        
        return dummy.next
```

#### 提交 685555175 · Runtime Error · 2025-12-18 11:22:32 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        min_heap: list[int, int, ListNode] = []
        for idx, node in enumerate(lists):
            if node:
                heapq.heappush(min_heap, (node.val, idx, node))
        dummy = ListNode()
        tail = dummy
        while min_heap:
            _, idx, node = heapq.heappop(min_heap)
            tail.next = node
            tail = tail.next
            if node.next:
                heapq.heappush(node.next.val, idx, node.next)
        return dummy.next
```


## 无重复字符的最长子串 (`longest-substring-without-repeating-characters`)

- 题目链接：https://leetcode.cn/problems/longest-substring-without-repeating-characters/
- 难度：Medium
- 标签：哈希表, 字符串, 滑动窗口
- 总提交次数：20
- 最近提交时间：2025-12-19 13:42:40 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-05 08:28:00 CST

```markdown
30秒口述思路（面试可直接说）：
这题是连续子串的最值问题，且“是否重复”只会被新加入的字符影响，具备局部可维护性。用滑动窗口维护“无重复”的不变量：右指针扩张，若遇到重复，就把左指针跳到该字符上次出现位置的后一个。两个指针都只往前走，每个字符最多进出一次，时间 O(n)，空间 O(k)。这比暴力避免了重复检查，是最直接且最优的做法。

易错点： window[r_char] >= left 的判断是算法的关键。因为 window 存的是一个字符在全局的最后位置，而我们的滑动窗口只关心当前窗口内是否有重复。这个条件就是为了确保，只有当一个重复字符上一次出现的位置，确实落在了当前 left 指针定义的窗口内时，我们才需要收缩窗口。否则，那个重复字符就是已经被排除在外的‘历史记录’，我们不应该受它的影响。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 659079906 | 2025-09-03 14:01:09 CST | python3 | Runtime Error | N/A | N/A |
| 659083058 | 2025-09-03 14:11:42 CST | python3 | Accepted | 15 ms | 17.8 MB |
| 659084003 | 2025-09-03 14:14:43 CST | python3 | Accepted | 19 ms | 17.8 MB |
| 660909405 | 2025-09-09 10:45:54 CST | python3 | Runtime Error | N/A | N/A |
| 660909504 | 2025-09-09 10:46:06 CST | python3 | Accepted | 27 ms | 17.8 MB |
| 660929659 | 2025-09-09 11:25:13 CST | python3 | Accepted | 15 ms | 17.8 MB |
| 675932020 | 2025-11-05 08:20:44 CST | python3 | Wrong Answer | N/A | N/A |
| 675932140 | 2025-11-05 08:22:33 CST | python3 | Accepted | 12 ms | 17.8 MB |
| 681222359 | 2025-11-28 08:34:52 CST | python3 | Runtime Error | N/A | N/A |
| 681222374 | 2025-11-28 08:35:12 CST | python3 | Runtime Error | N/A | N/A |
| 681222404 | 2025-11-28 08:35:48 CST | python3 | Runtime Error | N/A | N/A |
| 681222457 | 2025-11-28 08:36:43 CST | python3 | Wrong Answer | N/A | N/A |
| 681222580 | 2025-11-28 08:38:33 CST | python3 | Accepted | 12 ms | 17.5 MB |
| 681666460 | 2025-11-30 15:14:47 CST | python3 | Wrong Answer | N/A | N/A |
| 681666736 | 2025-11-30 15:15:59 CST | python3 | Accepted | 19 ms | 17.8 MB |
| 681901261 | 2025-12-01 15:58:43 CST | python3 | Wrong Answer | N/A | N/A |
| 681901441 | 2025-12-01 15:59:21 CST | python3 | Accepted | 15 ms | 17.5 MB |
| 685569547 | 2025-12-18 12:39:55 CST | python3 | Wrong Answer | N/A | N/A |
| 685569619 | 2025-12-18 12:40:34 CST | python3 | Accepted | 19 ms | 17.1 MB |
| 685794311 | 2025-12-19 13:42:40 CST | python3 | Accepted | 19 ms | 17.1 MB |

### 未通过提交代码
#### 提交 659079906 · Runtime Error · 2025-09-03 14:01:09 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        left = 0
        window = set()
        for right, r_char in enumerate(s):
            if r_char not in window:
                max_len = max(right - left + 1, max_len)
            window.add(r_char)
            while len(window) != right - left + 1:
                l_char = s[left]
                window.remove(l_char)
                left += 1
        return max_len
```

#### 提交 660909405 · Runtime Error · 2025-09-09 10:45:54 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        left = 0
        window = collections.defaultdict(int)
        for right, r_char in s:
            if r_char not in window:
                max_len = max(right - left + 1, max_len)
            window[r_char] += 1
            while len(window) < right - left + 1:
                l_char = s[left]
                if l_char in window:
                    window[l_char] -= 1
                if window[l_char] == 0:
                    del window[l_char]
                left += 1
        return max_len
```

#### 提交 675932020 · Wrong Answer · 2025-11-05 08:20:44 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        left = 0
        window = {}
        for right, r_char in enumerate(s):
            if r_char in window:
                left = window[r_char] + 1
            window[r_char] = right
            max_len = max(max_len, right - left + 1)
        return max_len
```

#### 提交 681222359 · Runtime Error · 2025-11-28 08:34:52 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 连续子串的极值问题，可以用滑动窗口
         window = {}  # 用一个字典来记录字符上一次出现的索引位置
         left = 0
         max_len = 0
         for right, r_char in enumerate(s):
            # 出现重复了，而且上一次出现的位置在窗口范围内
            if r_char in window and window[r_char] >= left:
                left += 1 # 将 left 前进一步，保证窗口内无重复
            window[r_char] = right
            max_len = max(max_len, right - left + 1)
        return max_len
```

#### 提交 681222374 · Runtime Error · 2025-11-28 08:35:12 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 连续子串的极值问题，可以用滑动窗口
         window = {}  # 用一个字典来记录字符上一次出现的索引位置
         left = 0
         max_len = 0
         for right, r_char in enumerate(s):
            # 出现重复了，而且上一次出现的位置在窗口范围内
            if r_char in window and window[r_char] >= left:
                left += 1 # 将 left 前进一步，保证窗口内无重复
            window[r_char] = right
            max_len = max(max_len, right - left + 1)
        
        return max_len
```

#### 提交 681222404 · Runtime Error · 2025-11-28 08:35:48 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 连续子串的极值问题，可以用滑动窗口
         window = {}  # 用一个字典来记录字符上一次出现的索引位置
         left = 0
         max_len = 0
         for right, r_char in enumerate(s):
            # 出现重复了，而且上一次出现的位置在窗口范围内
            if r_char in window and window[r_char] >= left:
                left += 1 # 将 left 前进一步，保证窗口内无重复
            window[r_char] = right
            max_len = max(max_len, right - left + 1)
        
        return max_len
```

#### 提交 681222457 · Wrong Answer · 2025-11-28 08:36:43 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 连续子串的极值问题，可以用滑动窗口
        window = {}  # 用一个字典来记录字符上一次出现的索引位置
        left = 0
        max_len = 0
        for right, r_char in enumerate(s):
            # 出现重复了，而且上一次出现的位置在窗口范围内
            if r_char in window and window[r_char] >= left:
                left += 1 # 将 left 前进一步，保证窗口内无重复
            window[r_char] = right
            max_len = max(max_len, right - left + 1)
        return max_len
```

#### 提交 681666460 · Wrong Answer · 2025-11-30 15:14:47 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        window = {}
        left = 0
        for right, r_char in enumerate(s):
            if r_char in window and window[r_char] >= left:
                left = window[r_char] + 1
                max_len = max(max_len, right - left + 1)
            window[r_char] = right
        return max_len
```

#### 提交 681901261 · Wrong Answer · 2025-12-01 15:58:43 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        window = {}
        left = 0
        for right, r_char in enumerate(s):
            if r_char in window and window[r_char] >= left:
                left += 1
            window[r_char] = right
            max_len = max(max_len, right - left + 1)
        return max_len
```

#### 提交 685569547 · Wrong Answer · 2025-12-18 12:39:55 CST · python3

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        left = 0
        ch_map = collections.defaultdict(int)
        for right, r_char in enumerate(s):
            if r_char in ch_map and ch_map[r_char] > left:
                left = ch_map[r_char] + 1
            ch_map[r_char] = right
            max_len = max(max_len, right - left + 1)
        return max_len
```


## 盛最多水的容器 (`container-with-most-water`)

- 题目链接：https://leetcode.cn/problems/container-with-most-water/
- 难度：Medium
- 标签：贪心, 数组, 双指针
- 总提交次数：5
- 最近提交时间：2025-12-18 11:26:41 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-21 08:25:34 CST

```markdown
定义左右两个指针指向数组两端，计算当前面积并更新最大值。
移动指针的策略是：谁短动谁。
因为面积受限于短板，如果动长板，宽度变小且高度不可能突破短板，面积只会变小；只有动短板，才有可能找到更高的线来弥补宽度的损失，从而获得更大的面积。
循环直到两指针相遇，时间复杂度是 O(N)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 679615644 | 2025-11-21 08:24:16 CST | python3 | Runtime Error | N/A | N/A |
| 679615674 | 2025-11-21 08:24:48 CST | python3 | Accepted | 115 ms | 28 MB |
| 679950115 | 2025-11-22 21:31:46 CST | python3 | Accepted | 119 ms | 28.1 MB |
| 681945580 | 2025-12-01 18:31:37 CST | python3 | Accepted | 73 ms | 28 MB |
| 685556301 | 2025-12-18 11:26:41 CST | python3 | Accepted | 79 ms | 27.2 MB |

### 未通过提交代码
#### 提交 679615644 · Runtime Error · 2025-11-21 08:24:16 CST · python3

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height)
        max_area = 0

        while left < right:
            width = right - left
            current_h = min(height[left], height[right])
            max_area = max(max_area, width * current_h)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
```


## LRU 缓存 (`lru-cache`)

- 题目链接：https://leetcode.cn/problems/lru-cache/
- 难度：Medium
- 标签：设计, 哈希表, 链表, 双向链表
- 总提交次数：11
- 最近提交时间：2025-12-18 10:54:15 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-04 09:56:34 CST

```markdown
OrderedDict 内部就是哈希表加双向链表，提供了 LRU 恰好需要的两个 O(1) 操作：move_to_end 挪到最新,popitem(last=False) 从最旧处弹出,复杂度 get/put 都是 O(1)
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 675297632 | 2025-11-02 16:03:22 CST | python3 | Runtime Error | N/A | N/A |
| 675297724 | 2025-11-02 16:03:41 CST | python3 | Accepted | 113 ms | 76.7 MB |
| 675682157 | 2025-11-04 09:41:16 CST | python3 | Runtime Error | N/A | N/A |
| 675682205 | 2025-11-04 09:41:29 CST | python3 | Accepted | 103 ms | 76.6 MB |
| 677525763 | 2025-11-12 09:38:09 CST | python3 | Runtime Error | N/A | N/A |
| 677525816 | 2025-11-12 09:38:25 CST | python3 | Accepted | 116 ms | 76.6 MB |
| 681677856 | 2025-11-30 15:52:57 CST | python3 | Runtime Error | N/A | N/A |
| 681677953 | 2025-11-30 15:53:13 CST | python3 | Runtime Error | N/A | N/A |
| 681678132 | 2025-11-30 15:53:46 CST | python3 | Accepted | 119 ms | 76.4 MB |
| 681914975 | 2025-12-01 16:39:54 CST | python3 | Accepted | 120 ms | 76.7 MB |
| 685546894 | 2025-12-18 10:54:15 CST | python3 | Accepted | 85 ms | 76 MB |

### 未通过提交代码
#### 提交 675297632 · Runtime Error · 2025-11-02 16:03:22 CST · python3

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.od = OrderedDict()
        

    def get(self, key: int) -> int:
        if key not in self.od:
            return -1
        self.od.move_to_end(key) # 挪到右端表示最近使用
        return self.od[key]


    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od[key] = value
            self.move_to_end(key) # 挪到右端表示最近使用
        else:
            self.od[key] = value
            if len(self.od) > self.cap:
                self.od.popitem(last=False) # 弹出最左端（最久未使用）
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 提交 675682157 · Runtime Error · 2025-11-04 09:41:16 CST · python3

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.od = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.od:
            return -1
        self.od.move_to_end(key)
        return self.od[key]

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od[key] = value
            self.od.move_to_end(key) # 移动到队列末端表示最近使用
        else:
            self.od[key] = value
            if len(self.od) > self.cap:
                self.popitem(last=False) # 将队列头部的最久未使用的弹出


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 提交 677525763 · Runtime Error · 2025-11-12 09:38:09 CST · python3

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.od = collections.OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.od:
            self.od.move_to_end(key)
            return self.od[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od[key] = value
            self.od.move_to_end(key)
        else:
            self.od[key] = value
            if len(self.od) > capacity:
                self.od.popitem(last=False)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 提交 681677856 · Runtime Error · 2025-11-30 15:52:57 CST · python3

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.od = collection.OrderedDict()
        

    def get(self, key: int) -> int:
        if key in self.od:
            self.od.move_to_end(key)
            return self.od[key]
        else:
            return -1
        

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od[key] = value
            self.od.move_to_end(key)
        else:
            self.od[key] = value
            if len(self.od) > capacity:
                self.od.popitem(last=False)
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 提交 681677953 · Runtime Error · 2025-11-30 15:53:13 CST · python3

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.od = collections.OrderedDict()
        

    def get(self, key: int) -> int:
        if key in self.od:
            self.od.move_to_end(key)
            return self.od[key]
        else:
            return -1
        

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od[key] = value
            self.od.move_to_end(key)
        else:
            self.od[key] = value
            if len(self.od) > capacity:
                self.od.popitem(last=False)
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```


## 搜索二维矩阵 II (`search-a-2d-matrix-ii`)

- 题目链接：https://leetcode.cn/problems/search-a-2d-matrix-ii/
- 难度：Medium
- 标签：数组, 二分查找, 分治, 矩阵
- 总提交次数：3
- 最近提交时间：2025-12-18 10:27:45 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-18 10:25:39 CST

```markdown
Z型搜索查找
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685294418 | 2025-12-17 09:11:35 CST | python3 | Accepted | 153 ms | 23.6 MB |
| 685538517 | 2025-12-18 10:23:42 CST | python3 | Wrong Answer | N/A | N/A |
| 685539556 | 2025-12-18 10:27:45 CST | python3 | Accepted | 159 ms | 23 MB |

### 未通过提交代码
#### 提交 685538517 · Wrong Answer · 2025-12-18 10:23:42 CST · python3

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        left, right = 0, n - 1
        while left < right:
            current_val = matrix[left][right]
            if current_val == target:
                return True
            elif current_val < target:
                left += 1
            else:
                right -= 1
        return False
```


## 判断子序列 (`is-subsequence`)

- 题目链接：https://leetcode.cn/problems/is-subsequence/
- 难度：Easy
- 标签：双指针, 字符串, 动态规划
- 总提交次数：1
- 最近提交时间：2025-12-17 12:47:42 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685341508 | 2025-12-17 12:47:42 CST | python3 | Accepted | 0 ms | 17.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 子集 (`TVdhkn`)

- 题目链接：https://leetcode.cn/problems/TVdhkn/
- 难度：Medium
- 标签：位运算, 数组, 回溯
- 总提交次数：1
- 最近提交时间：2025-12-17 09:32:04 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685297554 | 2025-12-17 09:32:04 CST | python3 | Accepted | 52 ms | 17.8 MB |

### 未通过提交代码
(所有提交均已通过)

## 搜索二维矩阵 (`search-a-2d-matrix`)

- 题目链接：https://leetcode.cn/problems/search-a-2d-matrix/
- 难度：Medium
- 标签：数组, 二分查找, 矩阵
- 总提交次数：1
- 最近提交时间：2025-12-17 09:04:09 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685293541 | 2025-12-17 09:04:09 CST | python3 | Accepted | 0 ms | 17.9 MB |

### 未通过提交代码
(所有提交均已通过)

## 排序链表 (`sort-list`)

- 题目链接：https://leetcode.cn/problems/sort-list/
- 难度：Medium
- 标签：链表, 双指针, 分治, 排序, 归并排序
- 总提交次数：10
- 最近提交时间：2025-12-16 10:35:45 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-16 09:59:15 CST

```markdown
整体是归并排序的思想
先找到链表的中点，然后将链表断开成两部分，再对这两部分递归的分别再找中点，直到只剩一个节点或者空节点，然后开始不断归并
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 685059090 | 2025-12-16 10:05:56 CST | python3 | Runtime Error | N/A | N/A |
| 685059411 | 2025-12-16 10:07:19 CST | python3 | Runtime Error | N/A | N/A |
| 685059875 | 2025-12-16 10:09:26 CST | python3 | Runtime Error | N/A | N/A |
| 685060029 | 2025-12-16 10:10:09 CST | python3 | Accepted | 179 ms | 32 MB |
| 685060102 | 2025-12-16 10:10:30 CST | python3 | Runtime Error | N/A | N/A |
| 685060454 | 2025-12-16 10:11:56 CST | python3 | Accepted | 179 ms | 32.3 MB |
| 685062793 | 2025-12-16 10:21:54 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 685064003 | 2025-12-16 10:26:36 CST | python3 | Accepted | 179 ms | 32 MB |
| 685065480 | 2025-12-16 10:31:55 CST | python3 | Wrong Answer | N/A | N/A |
| 685066608 | 2025-12-16 10:35:45 CST | python3 | Accepted | 183 ms | 32 MB |

### 未通过提交代码
#### 提交 685059090 · Runtime Error · 2025-12-16 10:05:56 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        mid = self.find_mid_divide(head)
        left_head = self.sortList(head)
        right_head = self.sortList(mid)
        return merge_sort(left_head, right_head)
    
    def find_mid_divide(self, head):
        if not head or head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        slow.next = None
        return slow
    
    def merge_sort(self, l1, l2):
        dummy = ListNode(0)
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2
        return dummy.next
```

#### 提交 685059411 · Runtime Error · 2025-12-16 10:07:19 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        mid = self.find_mid_divide(head)
        left_head = self.sortList(head)
        right_head = self.sortList(mid)
        return merge_sort(left_head, right_head)
    
    def find_mid_divide(self, head):
        if not head or head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        slow.next = None
        return slow
    
    def merge_sort(self, l1, l2):
        dummy = ListNode(0)
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2
        return dummy.next
```

#### 提交 685059875 · Runtime Error · 2025-12-16 10:09:26 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        mid = self.find_mid_divide(head)
        left_head = self.sortList(head)
        right_head = self.sortList(mid)
        return merge_sort(left_head, right_head)
    
    def find_mid_divide(self, head):
        if not head or head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        slow.next = None
        return mid
    
    def merge_sort(self, l1, l2):
        dummy = ListNode(0)
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2
        return dummy.next
```

#### 提交 685060102 · Runtime Error · 2025-12-16 10:10:30 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        mid = self.find_mid_divide(head)
        left_head = self.sortList(head)
        right_head = self.sortList(mid)
        return self.merge_sort(left_head, right_head)
    
    def find_mid_divide(self, head):
        if not head or head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        slow.next = None
        return mid
    
    def merge_sort(self, l1, l2):
        dummy = ListNode(0)
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2
        return dummy.next
```

#### 提交 685062793 · Time Limit Exceeded · 2025-12-16 10:21:54 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        mid = self.find_mid_and_split(head)
        left_head = self.sortList(head)
        right_head = self.sortList(mid)
        return self.merge_sort(left_head, right_head)
    
    def find_mid_and_split(self, head):
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        slow.next = None
        return mid
    
    def merge_sort(self, l1, l2):
        dummy = ListNode(0)
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
            else:
                tail.next = l2
            tail = tail.next
        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next
```

#### 提交 685065480 · Wrong Answer · 2025-12-16 10:31:55 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or head.next:
            return head
        mid = self.find_mid_divide(head)
        left_head = self.sortList(head)
        right_head = self.sortList(mid)
        return self.merge_sort(left_head, right_head)

    def find_mid_divide(self, head):
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        slow.next = None
        return mid
    
    def merge_sort(self, l1, l2):
        dummy = ListNode(0)
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next
```


## 二叉树的最小深度 (`minimum-depth-of-binary-tree`)

- 题目链接：https://leetcode.cn/problems/minimum-depth-of-binary-tree/
- 难度：Easy
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：1
- 最近提交时间：2025-12-15 13:48:04 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684866812 | 2025-12-15 13:48:04 CST | python3 | Accepted | 2 ms | 48.9 MB |

### 未通过提交代码
(所有提交均已通过)

## 二叉树的直径 (`diameter-of-binary-tree`)

- 题目链接：https://leetcode.cn/problems/diameter-of-binary-tree/
- 难度：Easy
- 标签：树, 深度优先搜索, 二叉树
- 总提交次数：5
- 最近提交时间：2025-12-15 13:37:14 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-15 10:59:46 CST

```markdown
整棵树的最大直径，就是任意节点的 left_depth + right_depth 的最大值
其中 left_depth 和 right_depth 是任意节点的左右子树的最大深度
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684837522 | 2025-12-15 11:04:07 CST | python3 | Wrong Answer | N/A | N/A |
| 684837653 | 2025-12-15 11:04:37 CST | python3 | Wrong Answer | N/A | N/A |
| 684838180 | 2025-12-15 11:06:25 CST | python3 | Accepted | 8 ms | 20.4 MB |
| 684864961 | 2025-12-15 13:36:53 CST | python3 | Runtime Error | N/A | N/A |
| 684865017 | 2025-12-15 13:37:14 CST | python3 | Accepted | 1 ms | 20.4 MB |

### 未通过提交代码
#### 提交 684837522 · Wrong Answer · 2025-12-15 11:04:07 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_diameter = 0
        def dfs(node):
            if not node:
                return 0
            left_depth = dfs(node.left)
            right_depth = dfs(node.right)
            max_diameter = max(max_diameter, left_depth + right_depth)
            return max(left_depth, right_depth) + 1
        return max_diameter
```

#### 提交 684837653 · Wrong Answer · 2025-12-15 11:04:37 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.max_diameter = 0
        def dfs(node):
            if not node:
                return 0
            left_depth = dfs(node.left)
            right_depth = dfs(node.right)
            self.max_diameter = max(self.max_diameter, left_depth + right_depth)
            return max(left_depth, right_depth) + 1
        return self.max_diameter
```

#### 提交 684864961 · Runtime Error · 2025-12-15 13:36:53 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.max_diameter = 0
        def max_depth(node):
            if not node:
                return 0
            left_depth = max_depth(node.left)
            right_depth = max_depth(node.right)
            self.max_diameter = max(self.max_depth, left_depth + right_depth)
            return max(left_depth, right_depth) + 1
        max_depth(root)
        return self.max_diameter
```


## 二叉树的最大深度 (`maximum-depth-of-binary-tree`)

- 题目链接：https://leetcode.cn/problems/maximum-depth-of-binary-tree/
- 难度：Easy
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：2
- 最近提交时间：2025-12-15 12:40:39 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680118503 | 2025-11-23 17:56:35 CST | python3 | Accepted | 0 ms | 18.7 MB |
| 684857278 | 2025-12-15 12:40:39 CST | python3 | Accepted | 0 ms | 18.7 MB |

### 未通过提交代码
(所有提交均已通过)

## N 叉树的后序遍历 (`n-ary-tree-postorder-traversal`)

- 题目链接：https://leetcode.cn/problems/n-ary-tree-postorder-traversal/
- 难度：Easy
- 标签：栈, 树, 深度优先搜索
- 总提交次数：2
- 最近提交时间：2025-12-14 19:53:53 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684742227 | 2025-12-14 19:50:32 CST | python3 | Accepted | 41 ms | 19.2 MB |
| 684742878 | 2025-12-14 19:53:53 CST | python3 | Accepted | 57 ms | 19 MB |

### 未通过提交代码
(所有提交均已通过)

## N 叉树的前序遍历 (`n-ary-tree-preorder-traversal`)

- 题目链接：https://leetcode.cn/problems/n-ary-tree-preorder-traversal/
- 难度：Easy
- 标签：栈, 树, 深度优先搜索
- 总提交次数：2
- 最近提交时间：2025-12-14 18:16:39 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684721873 | 2025-12-14 17:46:17 CST | python3 | Accepted | 53 ms | 19.2 MB |
| 684727215 | 2025-12-14 18:16:39 CST | python3 | Accepted | 56 ms | 18.8 MB |

### 未通过提交代码
(所有提交均已通过)

## N 叉树的层序遍历 (`n-ary-tree-level-order-traversal`)

- 题目链接：https://leetcode.cn/problems/n-ary-tree-level-order-traversal/
- 难度：Medium
- 标签：树, 广度优先搜索
- 总提交次数：2
- 最近提交时间：2025-12-14 17:41:04 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684720764 | 2025-12-14 17:40:54 CST | python3 | Runtime Error | N/A | N/A |
| 684720800 | 2025-12-14 17:41:04 CST | python3 | Accepted | 59 ms | 19 MB |

### 未通过提交代码
#### 提交 684720764 · Runtime Error · 2025-12-14 17:40:54 CST · python3

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: Optional[int] = None, children: Optional[List['Node']] = None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []
        dq = collections.deque([root])
        res = []
        while dq:
            sz = len(dq)
            level_vals = []
            for _ in range(len(sz)):
                node = dq.popleft()
                level_vals.append(node.val)
                for child in node.children:
                    dq.append(child)
            res.append(level_vals)
        return res
```


## 二叉树的后序遍历 (`binary-tree-postorder-traversal`)

- 题目链接：https://leetcode.cn/problems/binary-tree-postorder-traversal/
- 难度：Easy
- 标签：栈, 树, 深度优先搜索, 二叉树
- 总提交次数：5
- 最近提交时间：2025-12-13 22:08:40 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684563111 | 2025-12-13 18:29:18 CST | python3 | Wrong Answer | N/A | N/A |
| 684563288 | 2025-12-13 18:30:33 CST | python3 | Accepted | 0 ms | 17.4 MB |
| 684599575 | 2025-12-13 22:08:03 CST | python3 | Runtime Error | N/A | N/A |
| 684599619 | 2025-12-13 22:08:16 CST | python3 | Runtime Error | N/A | N/A |
| 684599693 | 2025-12-13 22:08:40 CST | python3 | Accepted | 0 ms | 17.7 MB |

### 未通过提交代码
#### 提交 684563111 · Wrong Answer · 2025-12-13 18:29:18 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            res.append(node.val)
            dfs(node.right)
        dfs(root)
        return res
```

#### 提交 684599575 · Runtime Error · 2025-12-13 22:08:03 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        stack = []
        stack.append(root)
        while stack:
            node = stack.append()
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res[::-1]
```

#### 提交 684599619 · Runtime Error · 2025-12-13 22:08:16 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res[::-1]
```


## 二叉树的中序遍历 (`binary-tree-inorder-traversal`)

- 题目链接：https://leetcode.cn/problems/binary-tree-inorder-traversal/
- 难度：Easy
- 标签：栈, 树, 深度优先搜索, 二叉树
- 总提交次数：4
- 最近提交时间：2025-12-13 22:05:22 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-01 21:28:23 CST

```markdown
不断的将左子节点入栈
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681993183 | 2025-12-01 21:30:53 CST | python3 | Accepted | 0 ms | 17.4 MB |
| 684563249 | 2025-12-13 18:30:14 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 684596583 | 2025-12-13 21:51:39 CST | python3 | Accepted | 0 ms | 17.4 MB |
| 684599087 | 2025-12-13 22:05:22 CST | python3 | Accepted | 0 ms | 17.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 二叉树的前序遍历 (`binary-tree-preorder-traversal`)

- 题目链接：https://leetcode.cn/problems/binary-tree-preorder-traversal/
- 难度：Easy
- 标签：栈, 树, 深度优先搜索, 二叉树
- 总提交次数：3
- 最近提交时间：2025-12-13 21:58:18 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684561918 | 2025-12-13 18:19:29 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 684562509 | 2025-12-13 18:24:21 CST | python3 | Accepted | 0 ms | 17.4 MB |
| 684597816 | 2025-12-13 21:58:18 CST | python3 | Accepted | 0 ms | 17.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 二叉树的层序遍历 (`binary-tree-level-order-traversal`)

- 题目链接：https://leetcode.cn/problems/binary-tree-level-order-traversal/
- 难度：Medium
- 标签：树, 广度优先搜索, 二叉树
- 总提交次数：3
- 最近提交时间：2025-12-13 18:13:10 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-26 08:32:41 CST

```markdown
dq 初始化的错误写法：
dq = collections.deque()
dq.append([root])

正确的应该是dq = collections.deque([root])
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680723262 | 2025-11-26 08:30:34 CST | python3 | Runtime Error | N/A | N/A |
| 680723563 | 2025-11-26 08:34:47 CST | python3 | Accepted | 0 ms | 18.2 MB |
| 684561088 | 2025-12-13 18:13:10 CST | python3 | Accepted | 3 ms | 18.1 MB |

### 未通过提交代码
#### 提交 680723262 · Runtime Error · 2025-11-26 08:30:34 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        dq = collections.deque()
        dq.append([root])
        res = [] 

        while dq:
            level_size = len(dq)
            level_res = []
            for _ in range(level_size):
                node = dq.popleft()
                level_res.append(node.val)
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
            res.append(level_res)
        
        return res
```


## 回文链表 (`palindrome-linked-list`)

- 题目链接：https://leetcode.cn/problems/palindrome-linked-list/
- 难度：Easy
- 标签：栈, 递归, 链表, 双指针
- 总提交次数：1
- 最近提交时间：2025-12-12 16:54:40 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684375581 | 2025-12-12 16:54:40 CST | python3 | Accepted | 36 ms | 34.1 MB |

### 未通过提交代码
(所有提交均已通过)

## 交换链表中的节点 (`swapping-nodes-in-a-linked-list`)

- 题目链接：https://leetcode.cn/problems/swapping-nodes-in-a-linked-list/
- 难度：Medium
- 标签：链表, 双指针
- 总提交次数：1
- 最近提交时间：2025-12-12 15:59:26 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-12 15:48:41 CST

```markdown
倒数的第 k 个节点是正数的第 n-k+1 个节点
第一个指针先走 k -1 步骤达到第 k 个节点，此时离链表尾部节点还有 n - k 步
另一个指针也开始从head 开始走，走 n-k 步后正好达到第 n-k+1 个节点，也就是倒数第 k 个节点
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684359733 | 2025-12-12 15:59:26 CST | python3 | Accepted | 35 ms | 39.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 两两交换链表中的节点 (`swap-nodes-in-pairs`)

- 题目链接：https://leetcode.cn/problems/swap-nodes-in-pairs/
- 难度：Medium
- 标签：递归, 链表
- 总提交次数：4
- 最近提交时间：2025-12-12 15:24:27 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684347578 | 2025-12-12 15:16:21 CST | python3 | Wrong Answer | N/A | N/A |
| 684347685 | 2025-12-12 15:16:44 CST | python3 | Wrong Answer | N/A | N/A |
| 684347999 | 2025-12-12 15:17:49 CST | python3 | Accepted | 0 ms | 17.6 MB |
| 684349875 | 2025-12-12 15:24:27 CST | python3 | Accepted | 0 ms | 17.4 MB |

### 未通过提交代码
#### 提交 684347578 · Wrong Answer · 2025-12-12 15:16:21 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or head.next:
            return head
        first = head
        second = head.next
        rest = second.next
        second.next = first
        first.next = self.swapPairs(rest)
        return second
```

#### 提交 684347685 · Wrong Answer · 2025-12-12 15:16:44 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or head.next:
            return head
        first = head
        second = head.next
        rest = second.next
        first.next = self.swapPairs(rest)
        second.next = first
        return second
```


## 有界数组中指定下标处的最大值 (`maximum-value-at-a-given-index-in-a-bounded-array`)

- 题目链接：https://leetcode.cn/problems/maximum-value-at-a-given-index-in-a-bounded-array/
- 难度：Medium
- 标签：贪心, 数学, 二分查找
- 总提交次数：2
- 最近提交时间：2025-12-12 14:27:01 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-12 11:35:00 CST

```markdown
一句话思路：对于指定下标 index 的值 x，在固定 x 的前提下，为了最容易满足总和 ≤ maxSum，我们把数组摆成以 index 为顶点、两侧步长为 1 递减到 1 的山峰形状，此时数组的最小可能总和会随 x 单调递增，所以可以对 x 的取值做二分查找，找到最大仍然不超 maxSum 的那个 x。


这句就把三件事都说清楚了：
* 山峰形状是“固定 x 时的最省钱构造”
* 单调性说的是“最小总和随 x 单调增”
* 因为单调，所以可以对 x 二分查找最大可行值
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 684333813 | 2025-12-12 14:25:23 CST | python3 | Wrong Answer | N/A | N/A |
| 684334228 | 2025-12-12 14:27:01 CST | python3 | Accepted | 3 ms | 17.5 MB |

### 未通过提交代码
#### 提交 684333813 · Wrong Answer · 2025-12-12 14:25:23 CST · python3

```python
class Solution:
    def maxValue(self, n: int, index: int, maxSum: int) -> int:
        left, right = 1, maxSum
        res = 0
        while left <= right:
            mid = left + (right - left) // 2
            total_sum = self.get_sum(n, index, mid)
            if total_sum <= maxSum:
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res
    
    def get_sum(self, n, index, peak):
        total_sum = peak
        left_len = index
        if peak - 1 >= left_len:
            first = peak - 1
            last = peak - left_len
            total_sum += (first + last) * left_len // 2
        else:
            dec_len = peak - 1
            dec_sum = (peak - 1 + 1) * dec_len // 2
            total_sum += (left_len - dec_len) * 1
        right_len = n - index - 1
        if peak - 1 >= right_len:
            first = peak - 1
            last = peak - right_len
            total_sum += (first + last) * right_len // 2
        else:
            dec_len = peak - 1
            dec_sum = (peak - 1 + 1) * dec_len // 2
            total_sum += (right_len - dec_len) * 1
        return total_sum
```


## 反转链表 (`reverse-linked-list`)

- 题目链接：https://leetcode.cn/problems/reverse-linked-list/
- 难度：Easy
- 标签：递归, 链表
- 总提交次数：4
- 最近提交时间：2025-12-11 12:36:21 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-11 12:34:42 CST

```markdown
每一层做的事情其实都一样：

“我相信后面那一段已经被反转好了，我只负责把自己挂到那段链表的尾巴后面。”
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682053626 | 2025-12-02 09:04:41 CST | python3 | Accepted | 0 ms | 18.6 MB |
| 684074472 | 2025-12-11 11:21:14 CST | python3 | Accepted | 0 ms | 18.6 MB |
| 684088741 | 2025-12-11 12:36:02 CST | python3 | Wrong Answer | N/A | N/A |
| 684088781 | 2025-12-11 12:36:21 CST | python3 | Accepted | 3 ms | 18.7 MB |

### 未通过提交代码
#### 提交 684088741 · Wrong Answer · 2025-12-11 12:36:02 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or head.next:
            return head
        new_head = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return new_head
```


## 两数相加 II (`add-two-numbers-ii`)

- 题目链接：https://leetcode.cn/problems/add-two-numbers-ii/
- 难度：Medium
- 标签：栈, 链表, 数学
- 总提交次数：3
- 最近提交时间：2025-12-10 11:31:13 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-07-30 08:10:29 CST

```markdown
链表头插法：
head = None
new_node = ListNode(new_val)
new_node.next = head
head = new_node
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 648208272 | 2025-07-30 08:09:25 CST | python3 | Accepted | 3 ms | 17.8 MB |
| 683847079 | 2025-12-10 11:30:35 CST | python3 | Memory Limit Exceeded | N/A | N/A |
| 683847253 | 2025-12-10 11:31:13 CST | python3 | Accepted | 8 ms | 17.6 MB |

### 未通过提交代码
#### 提交 683847079 · Memory Limit Exceeded · 2025-12-10 11:30:35 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        stk_1 = []
        stk_2 = []
        curr_1, curr_2 = l1, l2
        while curr_1:
            stk_1.append(curr_1.val)
            curr_1 = curr_1.next
        while curr_2:
            stk_2.append(curr_2.val)
        
        head = None
        carry = 0
        while stk_1 or stk_2 or carry:
            val_1 = stk_1.pop() if stk_1 else 0
            val_2 = stk_2.pop() if stk_2 else 0
            total = val_1 + val_2 + carry
            new_val = total % 10
            carry = total // 10

            new_node = ListNode(new_val)
            new_node.next = head
            head = new_node
        
        return head
```


## 两数相加 (`add-two-numbers`)

- 题目链接：https://leetcode.cn/problems/add-two-numbers/
- 难度：Medium
- 标签：递归, 链表, 数学
- 总提交次数：3
- 最近提交时间：2025-12-10 11:11:38 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-03 15:11:37 CST

```markdown
链表是按低位在前存的，所以我直接模拟竖式加法，从头往后加。
用一个 carry 记录进位，循环条件是任意链表没走完或者 carry 不为 0。
每一步取当前两个节点的值（为空当 0），加上 carry，得到当前位 digit 和新的 carry。
digit 用一个新节点接到结果链表后面，指针后移，最后返回 dummy.next。
时间复杂度 O(max(m, n))，额外空间 O(1)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 648207438 | 2025-07-30 07:47:14 CST | python3 | Accepted | 3 ms | 17.7 MB |
| 682395013 | 2025-12-03 15:23:20 CST | python3 | Accepted | 0 ms | 17.6 MB |
| 683841236 | 2025-12-10 11:11:38 CST | python3 | Accepted | 3 ms | 17.8 MB |

### 未通过提交代码
(所有提交均已通过)

## 查找和最小的 K 对数字 (`find-k-pairs-with-smallest-sums`)

- 题目链接：https://leetcode.cn/problems/find-k-pairs-with-smallest-sums/
- 难度：Medium
- 标签：数组, 堆（优先队列）
- 总提交次数：2
- 最近提交时间：2025-12-10 10:49:15 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-07-29 08:13:11 CST

```markdown
思路类似合并k个有序列表，把问题看做len(num1)个虚拟的有序链表；
写代码时出现的卡点：while循环时k怎么用？当然是通过len(result)做判断啊！
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 647890938 | 2025-07-29 08:10:26 CST | python3 | Accepted | 87 ms | 33.7 MB |
| 683834261 | 2025-12-10 10:49:15 CST | python3 | Accepted | 79 ms | 34 MB |

### 未通过提交代码
(所有提交均已通过)

## 有序矩阵中第 K 小的元素 (`kth-smallest-element-in-a-sorted-matrix`)

- 题目链接：https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/
- 难度：Medium
- 标签：数组, 二分查找, 矩阵, 排序, 堆（优先队列）
- 总提交次数：4
- 最近提交时间：2025-12-10 10:36:17 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-10 10:28:48 CST

```markdown
这题用最小堆其实体现的是一个通用框架：

把矩阵的每一行看成一个有序数组，我有 n 个有序序列，要找整体第 k 小。

我只维护一个「候选集」——一开始每行的第 0 个元素入最小堆。

每次从堆里弹出当前全局最小的元素，这是第 1 小、第 2 小……，当弹到第 k 次就是答案。

对于弹出的元素，只从它的来源“往后”扩展下一个候选，比如同一行往右一个，再放回堆中。

这个就是典型的「多路归并 + 最小堆 + 边界扩展」框架，很多 K 路合并、最短路问题都可以这么想。
```

#### 笔记 2 · 更新于 2025-07-28 09:27:52 CST

```markdown
入堆的时候写错，入堆的应该是个元祖 ，heapq.heappush(min_heap, (value, row[idx], col[idx]))
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 647586206 | 2025-07-28 09:24:58 CST | python3 | Accepted | 35 ms | 20.8 MB |
| 683830301 | 2025-12-10 10:35:47 CST | python3 | Runtime Error | N/A | N/A |
| 683830390 | 2025-12-10 10:36:03 CST | python3 | Runtime Error | N/A | N/A |
| 683830456 | 2025-12-10 10:36:17 CST | python3 | Accepted | 31 ms | 20.9 MB |

### 未通过提交代码
#### 提交 683830301 · Runtime Error · 2025-12-10 10:35:47 CST · python3

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        min_heap = []
        for r in range(min(n, k)):
            heapq.heappush(min_heap, (matrix[r][0], r, 0))
        count = 0
        while min_heap:
            val, r, c = heaq.heappop(min_heap)
            count += 1
            if count == k:
                return val
            if c + 1 < n:
                heaq.heappush(min_heap, (matrix[r][c + 1], r, c + 1))
        return -1
```

#### 提交 683830390 · Runtime Error · 2025-12-10 10:36:03 CST · python3

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        min_heap = []
        for r in range(min(n, k)):
            heapq.heappush(min_heap, (matrix[r][0], r, 0))
        count = 0
        while min_heap:
            val, r, c = heapq.heappop(min_heap)
            count += 1
            if count == k:
                return val
            if c + 1 < n:
                heaq.heappush(min_heap, (matrix[r][c + 1], r, c + 1))
        return -1
```


## 从未排序的链表中移除重复元素 (`remove-duplicates-from-an-unsorted-linked-list`)

- 题目链接：https://leetcode.cn/problems/remove-duplicates-from-an-unsorted-linked-list/
- 难度：Medium
- 标签：哈希表, 链表
- 总提交次数：1
- 最近提交时间：2025-12-10 10:05:56 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 683822342 | 2025-12-10 10:05:56 CST | python3 | Accepted | 217 ms | 49.3 MB |

### 未通过提交代码
(所有提交均已通过)

## 删除排序链表中的重复元素 II (`remove-duplicates-from-sorted-list-ii`)

- 题目链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/
- 难度：Medium
- 标签：链表, 双指针
- 总提交次数：2
- 最近提交时间：2025-12-10 09:02:03 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-10 08:55:00 CST

```markdown
因为是排序的，所以值相同的节点都挨着
先把重复的都找出来，然后之前让重复部分的前一个节点的 next 指针指向重复部分后面的节点
注意head 指针也可能是要被删除的，所以需要 dummy
```

#### 笔记 2 · 更新于 2025-07-25 09:40:38 CST

```markdown
1、第二个while循环前面引入的dup_val不是冗余，是为了让“要跳过的目标值”在整个while循环里保持不变
2、写错的地方：判断下一节点不重复时，应该是
prev = curr
curr = curr.next
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 646850181 | 2025-07-25 09:36:03 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 683811198 | 2025-12-10 09:02:03 CST | python3 | Accepted | 4 ms | 17.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 删除排序链表中的重复元素 (`remove-duplicates-from-sorted-list`)

- 题目链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-list/
- 难度：Easy
- 标签：链表
- 总提交次数：2
- 最近提交时间：2025-12-10 08:21:42 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-10 08:18:05 CST

```markdown
while循环写错：
while fast:
            if fast.val != slow.val:
                slow = slow.next
                slow.next = fast
            fast = fast.next
对比力扣第26题，对于数组中的slow指针来说，前进一步是slow+=1，然后再原地修改，但对于链表节点来说是先修改slow.next然后再原地修改slow=slow.next


这道题链表是有序的，所以所有重复元素会挨在一起。我用一个指针从头往后扫，每次比较当前节点和下一个节点的值：如果相同，就把下一个节点跳过去，也就是 cur.next = cur.next.next；如果不同，就把指针往后移。这样一趟扫描就能原地去重，时间复杂度 O(n)，空间复杂度 O(1)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 649584787 | 2025-08-04 07:57:00 CST | python3 | Accepted | 0 ms | 17.4 MB |
| 683807965 | 2025-12-10 08:21:42 CST | python3 | Accepted | 0 ms | 17.7 MB |

### 未通过提交代码
(所有提交均已通过)

## 相交链表 (`intersection-of-two-linked-lists`)

- 题目链接：https://leetcode.cn/problems/intersection-of-two-linked-lists/
- 难度：Easy
- 标签：哈希表, 链表, 双指针
- 总提交次数：2
- 最近提交时间：2025-12-09 15:02:13 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-09 15:01:16 CST

```markdown
整体思路：通过某种方式，是的 p1 和 p2 同时达到相交点

while语句中写错的语句：
p1 = headB if p1.next is None else p1.next
p2 = headA if p2.next is None else p2.next
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 646539996 | 2025-07-24 08:06:07 CST | python3 | Accepted | 105 ms | 27.3 MB |
| 683649333 | 2025-12-09 15:02:13 CST | python3 | Accepted | 135 ms | 32.1 MB |

### 未通过提交代码
(所有提交均已通过)

## 删除链表的倒数第 N 个结点 (`remove-nth-node-from-end-of-list`)

- 题目链接：https://leetcode.cn/problems/remove-nth-node-from-end-of-list/
- 难度：Medium
- 标签：链表, 双指针
- 总提交次数：4
- 最近提交时间：2025-12-09 14:49:55 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-09 14:54:50 CST

```markdown
slow 和 fast 都从 dummy 出发，fast 先走 n+1 步后 fast 和 slow 相差 n+1 个
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 645609950 | 2025-07-21 07:57:52 CST | python3 | Accepted | 0 ms | 17.5 MB |
| 679636678 | 2025-11-21 10:38:39 CST | python3 | Accepted | 0 ms | 17.6 MB |
| 683645232 | 2025-12-09 14:49:24 CST | python3 | Runtime Error | N/A | N/A |
| 683645403 | 2025-12-09 14:49:55 CST | python3 | Accepted | 0 ms | 17.7 MB |

### 未通过提交代码
#### 提交 683645232 · Runtime Error · 2025-12-09 14:49:24 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        slow = fast = head
        for _ in range(n+1):
            fast = fast.next
        while fast:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummy.next
```


## 链表的中间结点 (`middle-of-the-linked-list`)

- 题目链接：https://leetcode.cn/problems/middle-of-the-linked-list/
- 难度：Easy
- 标签：链表, 双指针
- 总提交次数：2
- 最近提交时间：2025-12-09 14:33:43 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-09 14:33:02 CST

```markdown
当链表长度为偶数 2k 时，这个循环会执行 k 次，slow 走了 k 步，从第 1 个节点走到第 k+1 个节点。
两个中点是第 k 和第 k+1 个节点，所以这种写法自然返回的是第二个中点。
想要第一个中点，可以让 fast 从 head.next 开始
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 645916376 | 2025-07-22 07:26:34 CST | python3 | Accepted | 0 ms | 17.7 MB |
| 683640550 | 2025-12-09 14:33:43 CST | python3 | Accepted | 4 ms | 17.3 MB |

### 未通过提交代码
(所有提交均已通过)

## 环形链表 II (`linked-list-cycle-ii`)

- 题目链接：https://leetcode.cn/problems/linked-list-cycle-ii/
- 难度：Medium
- 标签：哈希表, 链表, 双指针
- 总提交次数：2
- 最近提交时间：2025-12-09 14:19:57 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-09 14:08:15 CST

```markdown
a -> 头节点到入环点的距离
b -> 入环点到第一次相遇点的距离
c -> 环的长度
第一次相遇时：
	slow 指针走了 a + b
	fast 指针走了 2(a+b)，而且 fast 指针比 slow 指针多走了若干圈 k * c
所以：2(a+b) = a+b+k * c
a + b = k * c
a = k * c - b
a = (k - 1) * c + c - b

其中 c - b 是从相遇点到入环点的距离
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 646235503 | 2025-07-23 08:13:01 CST | python3 | Accepted | 87 ms | 19.1 MB |
| 683637008 | 2025-12-09 14:19:57 CST | python3 | Accepted | 58 ms | 19.3 MB |

### 未通过提交代码
(所有提交均已通过)

## 快乐数 (`happy-number`)

- 题目链接：https://leetcode.cn/problems/happy-number/
- 难度：Easy
- 标签：哈希表, 数学, 双指针
- 总提交次数：2
- 最近提交时间：2025-12-09 12:48:54 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-09 12:38:38 CST

```markdown
把这个过程看成对数字做状态转移：每次把数字替换成各位数字平方和。由于状态是有限的，要么最后变成 1，要么进入循环，所以只要用一个集合记录出现过的数字。如果过程中 n 变成 1 就是快乐数；如果某个数字第二次出现，说明进入了循环，就不是快乐数。复杂度是 O(k) 时间、O(k) 空间。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 683620207 | 2025-12-09 12:42:27 CST | python3 | Runtime Error | N/A | N/A |
| 683621050 | 2025-12-09 12:48:54 CST | python3 | Accepted | 0 ms | 17.8 MB |

### 未通过提交代码
#### 提交 683620207 · Runtime Error · 2025-12-09 12:42:27 CST · python3

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        def next_num(x):
            total = 0
            digit = x % 10
            total += digit * digit
            x //= 10 
        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = next_num(n)
        return n == 1
```


## 环形链表 (`linked-list-cycle`)

- 题目链接：https://leetcode.cn/problems/linked-list-cycle/
- 难度：Easy
- 标签：哈希表, 链表, 双指针
- 总提交次数：2
- 最近提交时间：2025-12-09 11:13:56 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 645916468 | 2025-07-22 07:30:57 CST | python3 | Accepted | 46 ms | 19.6 MB |
| 683601885 | 2025-12-09 11:13:56 CST | python3 | Accepted | 52 ms | 19.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 分隔链表 (`partition-list`)

- 题目链接：https://leetcode.cn/problems/partition-list/
- 难度：Medium
- 标签：链表, 双指针
- 总提交次数：3
- 最近提交时间：2025-12-09 09:44:14 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-09 09:36:14 CST

```markdown
1、写到最后的时候发现dummy_large节点该怎么用了：
重点在于large_tail = dummy_large这一行，这行执行后large_tail和 dummy_large其实指向了同一块儿内存，当large_tail开始修改next时，dummy_large.next也就变了，也就一直指向大链表的头结点

2、另外针对将原链表的节点接到一个新链表时的场景，要养成一个习惯是先临时存储next_node，然后再断开，这样就可以避免成环的情况

一次遍历，用两个链表收集两类节点，最后拼起来
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 645150718 | 2025-07-19 08:20:43 CST | python3 | Accepted | 0 ms | 17.7 MB |
| 683576330 | 2025-12-09 09:41:56 CST | python3 | Runtime Error | N/A | N/A |
| 683576757 | 2025-12-09 09:44:14 CST | python3 | Accepted | 0 ms | 17.4 MB |

### 未通过提交代码
#### 提交 683576330 · Runtime Error · 2025-12-09 09:41:56 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        dummy_large = ListNode()
        dummy_small = ListNode()
        large_tail = dummy_large
        small_tail = dummy_small

        current = head
        while current:
            next_node = current.next  # 先暂存
            if current.val < x:
                small_tail.next = current
                small_tail = large_tail.next
            else:
                large_tail.next = current
                large_tail = large_tail.next
            current.next = None  # 与原链表断开，防止成环
            current = next_node
        small_tail.next = dummy_large.next
        return dummy_small.next
```


## 合并两个有序链表 (`merge-two-sorted-lists`)

- 题目链接：https://leetcode.cn/problems/merge-two-sorted-lists/
- 难度：Easy
- 标签：递归, 链表
- 总提交次数：2
- 最近提交时间：2025-12-09 09:01:23 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-07-17 08:09:29 CST

```markdown
1、致命错误：竟然把链表误当成了list...，一开始就搞错了
2、想象成拉拉链，得让尾指针不断前进
3、dummy节点的设计理念应该是只占位，不存数据
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 644610117 | 2025-07-17 08:07:37 CST | python3 | Accepted | 3 ms | 17.4 MB |
| 683569637 | 2025-12-09 09:01:23 CST | python3 | Accepted | 3 ms | 17.5 MB |

### 未通过提交代码
(所有提交均已通过)

## 括号生成 (`generate-parentheses`)

- 题目链接：https://leetcode.cn/problems/generate-parentheses/
- 难度：Medium
- 标签：字符串, 动态规划, 回溯
- 总提交次数：5
- 最近提交时间：2025-12-08 14:22:38 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-07 14:59:52 CST

```markdown
“这题本质是要枚举所有合法括号串，非常适合用回溯。我们从空串开始，每一步有两个决策：加左括号或右括号，但通过两个约束剪枝：左括号总数不能超过 n，右括号数量不能超过左括号。这样在递归构造字符串的过程中，所有非法分支都会被提前剪掉，既保证了只生成合法结果，又避免了无效枚举，典型的回溯应用场景。”

“看到‘所有组合’想回溯，遇到‘非法’即剪枝。”

backtrack 的参数只传“当前这一层做决策必需的最小状态”。能从外层闭包拿到的，就别当参数；每层都会变化、且用来确定下一步选择或剪枝的，才作为参数传下去。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681555422 | 2025-11-29 21:38:06 CST | python3 | Accepted | 3 ms | 17.6 MB |
| 681604545 | 2025-11-30 10:26:14 CST | python3 | Accepted | 0 ms | 17.6 MB |
| 681812361 | 2025-12-01 10:11:06 CST | python3 | Accepted | 3 ms | 17.8 MB |
| 683210359 | 2025-12-07 14:57:31 CST | python3 | Accepted | 3 ms | 17.9 MB |
| 683401461 | 2025-12-08 14:22:38 CST | python3 | Accepted | 0 ms | 17.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 拆分字符串使唯一子字符串的数目最大 (`split-a-string-into-the-max-number-of-unique-substrings`)

- 题目链接：https://leetcode.cn/problems/split-a-string-into-the-max-number-of-unique-substrings/
- 难度：Medium
- 标签：哈希表, 字符串, 回溯
- 总提交次数：2
- 最近提交时间：2025-12-05 09:12:54 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-05 09:09:48 CST

```markdown
* 这题我用回溯加 set 来做。
* 从左到右枚举切分，每次从当前下标 i 开始，尝试所有可能的子串 s[i:j]。
* 如果这个子串不在 set 里，就把它加入 set，然后递归处理后面的部分。
* 当走到字符串末尾时，用当前 set 的大小更新最大答案。
* 因为字符串长度最多 16，最坏情况回溯的复杂度接近 2^n，配合 set 剪枝在这里是可以接受的。
* 最终返回搜索过程中的最大子串数量。

* 在写回溯的时候会注意 Python 的作用域规则。
* 如果在内层函数里给外层变量赋值，比如 res = ...，Python 会把它当成本地变量，导致 UnboundLocalError，并且不会更新外层的值。
* 正确做法是用 nonlocal res 显式声明，或者更干净一点，用返回值把“当前分支的最优结果”往上传，这样就不需要依赖外部可变状态。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682783327 | 2025-12-05 09:09:30 CST | python3 | Runtime Error | N/A | N/A |
| 682783742 | 2025-12-05 09:12:54 CST | python3 | Accepted | 171 ms | 17.5 MB |

### 未通过提交代码
#### 提交 682783327 · Runtime Error · 2025-12-05 09:09:30 CST · python3

```python
class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        res = 0
        used = set()
        n = len(s)
        def backtrack(start_index):
            if start_index == n:
                res = max(res, len(used))
                return
            for end_index in range(start_index + 1, n + 1):
                sub_str = s[start_index : end_index]
                if sub_str in used:
                    continue
                used.add(sub_str)
                backtrack(end_index)
                used.remove(sub_str)
        backtrack(0)
        return res
```


## 螺旋矩阵 (`spiral-matrix`)

- 题目链接：https://leetcode.cn/problems/spiral-matrix/
- 难度：Medium
- 标签：数组, 矩阵, 模拟
- 总提交次数：1
- 最近提交时间：2025-12-04 11:27:53 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682585678 | 2025-12-04 11:27:53 CST | python3 | Accepted | 0 ms | 17.3 MB |

### 未通过提交代码
(所有提交均已通过)

## 缺失的第一个正数 (`first-missing-positive`)

- 题目链接：https://leetcode.cn/problems/first-missing-positive/
- 难度：Hard
- 标签：数组, 哈希表
- 总提交次数：2
- 最近提交时间：2025-12-04 10:50:23 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-04 10:16:22 CST

```markdown
核心结论：
对于长度为 N 的数组，缺失的第一个正数一定在 [1, N+1] 这个范围内。
* 如果数组是 [1, 2, 3]，缺失的是 4 (即 N+1)。
* 如果数组是 [100, -1, 3]，缺失的是 1。


为什么只用管 1..N+1？
* 负数、0 不可能是答案（我们要“正整数”）
* 很大的数比如 1000，也不可能是最小缺失正数
* 比如有 1000 缺 1，答案还是 1，不会轮到 1000

更精确一点：
* 数组长度是 N，最多能装 N 个不同的数
* 如果 1..N 都出现了，那最小缺失正数只能是 N+1
* 如果 1..N 里面有缺的，那答案就在 1..N 里

所以：最小缺失正数一定在 [1, N+1] 这个范围中

这就有两个直接用处：
* 所有 <=0 和 >N 的数，对答案完全没贡献，可以忽略
* 我们只需要一种办法，快速判断 1..N 里每个数在不在数组里

所谓的「归位」思想：每个有效元素（1..N）尽量被放到自己“应该呆”的坑里。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682573839 | 2025-12-04 10:49:20 CST | python3 | Runtime Error | N/A | N/A |
| 682574160 | 2025-12-04 10:50:23 CST | python3 | Accepted | 47 ms | 28.3 MB |

### 未通过提交代码
#### 提交 682573839 · Runtime Error · 2025-12-04 10:49:20 CST · python3

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        # 缺失的数一定在 1 到 n + 1 之间
        n = len(nums)
        # 先归位
        for i in range(n):
            # 要注意避免重复的数相互交换导致的死循环
            while 1 <= nums[i] <= n + 1 and nums[nums[i] - 1] != nums[i]:
                target_index = nums[i] - 1  # nums[i] 应该待的位置
                nums[target_index], nums[i] = nums[i], nums[target_index]
        
        # 再扫描，找到一个不匹配的位置
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        # 都匹配了，则缺失的是 n + 1
        return n + 1
```


## 寻找两个正序数组的中位数 (`median-of-two-sorted-arrays`)

- 题目链接：https://leetcode.cn/problems/median-of-two-sorted-arrays/
- 难度：Hard
- 标签：数组, 二分查找, 分治
- 总提交次数：4
- 最近提交时间：2025-12-03 17:01:44 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-03 16:51:14 CST

```markdown
把中位数转化为在两个有序数组里找第 k 小。
每次比较两边的第 k/2 个值，较小一侧的这 k/2 个一定不可能是第 k 小，因为它们的最大排名也只有 k-1，所以可以整体丢掉，并把 k 减少。重复直到 k==1。总复杂度 O(log(m+n))。长度为偶数时取中间两数的平均。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681810301 | 2025-12-01 10:01:00 CST | python3 | Runtime Error | N/A | N/A |
| 681810527 | 2025-12-01 10:02:05 CST | python3 | Runtime Error | N/A | N/A |
| 681810675 | 2025-12-01 10:02:46 CST | python3 | Accepted | 11 ms | 17.9 MB |
| 682429149 | 2025-12-03 17:01:44 CST | python3 | Accepted | 1 ms | 18.1 MB |

### 未通过提交代码
#### 提交 681810301 · Runtime Error · 2025-12-01 10:01:00 CST · python3

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def find_kth(k, array_1, array_2):
            m, n = len(array_1), len(array_2)
            if m > 2:
                return find_kth(k, array_2, array_1)
            if m == 0:
                return array_2[k - 1]
            if k == 1:
                return min(array_1[0], array_2[0])
            i = min(k // 2, m)
            j = min(k // 2, n)
            if array_1[i] < array_2[j]:
                return find_kth(k - i, array_1[i:], array_2)
            else:
                return find_kth(k - j, array_1, array_2[j:])
        total_len = len(nums1) + len(nums2)
        if total_len % 2 == 1:
            return find_kth((total_len + 1) // 2, nums1, nums2)
        else:
            left = find_kth(total_len // 2, nums1, nums2)
            right = find_kth(total_len // 2 + 1, nums1, nums2)
            return (left + right) / 2
```

#### 提交 681810527 · Runtime Error · 2025-12-01 10:02:05 CST · python3

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def find_kth(k, array_1, array_2):
            m, n = len(array_1), len(array_2)
            if m > 2:
                return find_kth(k, array_2, array_1)
            if m == 0:
                return array_2[k - 1]
            if k == 1:
                return min(array_1[0], array_2[0])
            i = min(k // 2, m)
            j = min(k // 2, n)
            if array_1[i-1] < array_2[j-1]:
                return find_kth(k - i, array_1[i:], array_2)
            else:
                return find_kth(k - j, array_1, array_2[j:])
        total_len = len(nums1) + len(nums2)
        if total_len % 2 == 1:
            return find_kth((total_len + 1) // 2, nums1, nums2)
        else:
            left = find_kth(total_len // 2, nums1, nums2)
            right = find_kth(total_len // 2 + 1, nums1, nums2)
            return (left + right) / 2
```


## 最长连续序列 (`longest-consecutive-sequence`)

- 题目链接：https://leetcode.cn/problems/longest-consecutive-sequence/
- 难度：Medium
- 标签：并查集, 数组, 哈希表
- 总提交次数：7
- 最近提交时间：2025-12-03 15:54:21 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-03 15:52:05 CST

```markdown
“只从起点扩展”是本题的本质。

因为 logic if (num - 1) not in num_set 的存在，while 循环只会对序列的第一个数字执行

每个元素最多被“看”两次，所以是 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681796819 | 2025-12-01 08:21:43 CST | python3 | Accepted | 43 ms | 33.5 MB |
| 681973525 | 2025-12-01 20:27:09 CST | python3 | Wrong Answer | N/A | N/A |
| 681973882 | 2025-12-01 20:28:28 CST | python3 | Wrong Answer | N/A | N/A |
| 681974666 | 2025-12-01 20:30:51 CST | python3 | Wrong Answer | N/A | N/A |
| 681975892 | 2025-12-01 20:33:23 CST | python3 | Accepted | 55 ms | 32.6 MB |
| 682404833 | 2025-12-03 15:52:03 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 682405638 | 2025-12-03 15:54:21 CST | python3 | Accepted | 55 ms | 32.7 MB |

### 未通过提交代码
#### 提交 681973525 · Wrong Answer · 2025-12-01 20:27:09 CST · python3

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_len = 0
        nums_set = set(nums)
        for num in nums_set:
            if num - 1 in nums_set:
                continue
            current_num = num
            current_len = 1
            while current_num + 1 in nums_set:
                current_num = current_num + 1
                max_len = max(max_len, current_len + 1)
        return max_len
```

#### 提交 681973882 · Wrong Answer · 2025-12-01 20:28:28 CST · python3

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_len = 0
        nums_set = set(nums)
        for num in nums_set:
            if num - 1 in nums_set:
                continue
            current_num = num
            current_len = 1
            while current_num + 1 in nums_set:
                current_num += 1
                current_len += 1
                max_len = max(max_len, current_len)
        return max_len
```

#### 提交 681974666 · Wrong Answer · 2025-12-01 20:30:51 CST · python3

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_len = 0
        nums_set = set(nums)
        for num in nums_set:
            if num - 1 in nums_set:
                continue
            current_num = num
            current_len = 1
            while current_num + 1 in nums_set:
                current_num += 1
                current_len += 1
                # max_len = max(max_len, current_len)
                if current_len > max_len:
                    max_len = current_len
        return max_len
```

#### 提交 682404833 · Time Limit Exceeded · 2025-12-03 15:52:03 CST · python3

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_len = 0
        nums_set = set(nums)
        for num in nums:
            if num - 1 in nums_set:
                continue
            curr_num = num
            curr_len = 1
            while curr_num + 1 in nums_set:
                curr_num += 1
                curr_len += 1
            max_len = max(max_len, curr_len)
        return max_len
```


## 合并两个有序数组 (`merge-sorted-array`)

- 题目链接：https://leetcode.cn/problems/merge-sorted-array/
- 难度：Easy
- 标签：数组, 双指针, 排序
- 总提交次数：2
- 最近提交时间：2025-12-03 15:44:18 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-03 15:40:51 CST

```markdown
这道题我使用三指针、从后往前的策略。
因为 nums1 的后半部分是空的，从后往前填数字可以避免覆盖掉 nums1 中尚未处理的有效元素，同时避免了数据搬移。
我设置三个指针：分别指向 nums1 的有效尾部、nums2 的尾部，以及 nums1 的总尾部。
比较两个数组尾部元素，谁大谁就填入总尾部，然后指针前移。
最后，如果 nums2 还有剩余元素，直接拷贝到 nums1 头部即可。”

时间复杂度：$O(m+n)$，每个元素只处理一次
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 665877273 | 2025-09-25 13:05:26 CST | python3 | Accepted | 0 ms | 17.8 MB |
| 682402229 | 2025-12-03 15:44:18 CST | python3 | Accepted | 0 ms | 17.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 和为 K 的子数组 (`subarray-sum-equals-k`)

- 题目链接：https://leetcode.cn/problems/subarray-sum-equals-k/
- 难度：Medium
- 标签：数组, 哈希表, 前缀和
- 总提交次数：5
- 最近提交时间：2025-12-03 14:53:42 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-03 14:28:55 CST

```markdown
这道题我采用前缀和结合哈希表的方法来优化时间复杂度。
核心思路是遍历数组计算当前的累加和，并检查哈希表中是否存在 当前和 - K 的键。如果存在，说明中间有一段子数组的和为 K，将对应的次数累加到结果中。
然后，将当前和出现的次数更新到哈希表中。需要注意的是，哈希表初始化时要放入 {0: 1}，这是为了处理从数组下标 0 开始就满足条件的边界情况。整体时间复杂度是 $O(N)$。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 662224766 | 2025-09-13 10:06:15 CST | python3 | Accepted | 27 ms | 20.1 MB |
| 663591836 | 2025-09-17 15:58:15 CST | python3 | Wrong Answer | N/A | N/A |
| 663592105 | 2025-09-17 15:58:48 CST | python3 | Accepted | 30 ms | 20 MB |
| 682385328 | 2025-12-03 14:53:10 CST | python3 | Wrong Answer | N/A | N/A |
| 682385489 | 2025-12-03 14:53:42 CST | python3 | Accepted | 47 ms | 20.1 MB |

### 未通过提交代码
#### 提交 663591836 · Wrong Answer · 2025-09-17 15:58:15 CST · python3

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        res = 0
        pre_sum_map = {0: 1}
        pre_sum = 0

        for num in nums:
            pre_sum += num
            target = pre_sum - k
            if target in pre_sum_map:
                res += pre_sum_map.get(target)
            pre_sum_map[pre_sum] = pre_sum_map.get(target, 0) + 1

        return res
```

#### 提交 682385328 · Wrong Answer · 2025-12-03 14:53:10 CST · python3

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = 0
        pre_sum = 0
        pre_sum_map = collections.defaultdict(int)
        for num in nums:
            pre_sum += num
            target = pre_sum - k
            if target in pre_sum_map:
                cnt += pre_sum_map.get(target)
            pre_sum_map[pre_sum] += 1
        return cnt
```


## 跳跃游戏 (`jump-game`)

- 题目链接：https://leetcode.cn/problems/jump-game/
- 难度：Medium
- 标签：贪心, 数组, 动态规划
- 总提交次数：1
- 最近提交时间：2025-12-03 13:48:03 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-12-03 15:27:34 CST

```markdown
核心不在于“具体每一步跳哪里”，而在于“我最远能覆盖到哪里”。
只要当前位置在我的“覆盖范围”内，我就能利用当前位置继续延伸我的“覆盖范围”

本质是贪心算法
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 682369117 | 2025-12-03 13:48:03 CST | python3 | Accepted | 35 ms | 18.3 MB |

### 未通过提交代码
(所有提交均已通过)

## 字母异位词分组 (`group-anagrams`)

- 题目链接：https://leetcode.cn/problems/group-anagrams/
- 难度：Medium
- 标签：数组, 哈希表, 字符串, 排序
- 总提交次数：2
- 最近提交时间：2025-12-01 18:19:26 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-19 08:10:32 CST

```markdown
面试口述要点（30秒内）
* 用哈希表分组，关键是给每个词一个“签名”。签名可以是排序后的字符串，或26维字母频次元组。遍历一遍，把同签名的丢到同一桶里即可。时间复杂度排序版 O(n·k log k)，计数版 O(n·k)，空间 O(n·k)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 679109295 | 2025-11-19 08:08:55 CST | python3 | Accepted | 19 ms | 22.2 MB |
| 681941805 | 2025-12-01 18:19:26 CST | python3 | Accepted | 19 ms | 22.2 MB |

### 未通过提交代码
(所有提交均已通过)

## 三数之和 (`3sum`)

- 题目链接：https://leetcode.cn/problems/3sum/
- 难度：Medium
- 标签：数组, 双指针, 排序
- 总提交次数：5
- 最近提交时间：2025-12-01 17:10:45 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-30 17:54:03 CST

```markdown
“这道题我采用排序加双指针的方法来解决，时间复杂度是 $O(N^2)$。
首先对数组进行排序，然后遍历数组固定第一个数 nums[i]。对于剩下的部分，我使用左右双指针来寻找两数之和等于 -nums[i]。
这里的难点在于去重：
第一，在外层循环中，如果当前数和上一个数相同，需要跳过；
第二，在找到可行解后，左右指针移动时也要跳过重复的值。
这样就能保证结果集中不包含重复的三元组。”
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681710969 | 2025-11-30 18:00:30 CST | python3 | Wrong Answer | N/A | N/A |
| 681711311 | 2025-11-30 18:02:13 CST | python3 | Accepted | 451 ms | 20.3 MB |
| 681924467 | 2025-12-01 17:09:11 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 681924849 | 2025-12-01 17:10:23 CST | python3 | Accepted | 830 ms | 20.4 MB |
| 681924972 | 2025-12-01 17:10:45 CST | python3 | Accepted | 923 ms | 20.4 MB |

### 未通过提交代码
#### 提交 681710969 · Wrong Answer · 2025-11-30 18:00:30 CST · python3

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        if n < 3:
            return []
        nums.sort()
        res = []
        for i in range(n - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, right = i + 1, n - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total == 0:
                    res.append([nums[i], nums[left], nums[left]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        return res
```

#### 提交 681924467 · Time Limit Exceeded · 2025-12-01 17:09:11 CST · python3

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        n = len(nums)
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, right = i + 1, n - 1
            while left < right:
                if nums[i] + nums[left] + nums[right] == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                elif nums[i] + nums[left] + nums[right] < 0:
                    left += 1
                else:
                    right -= 1
        return res
```


## 接雨水 (`trapping-rain-water`)

- 题目链接：https://leetcode.cn/problems/trapping-rain-water/
- 难度：Hard
- 标签：栈, 数组, 双指针, 动态规划, 单调栈
- 总提交次数：2
- 最近提交时间：2025-12-01 16:53:11 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-30 17:23:49 CST

```markdown
本质一句话：每一格能接多少水，取决于它左边最高柱子、右边最高柱子的较小值，减去它自己的高度： water[i] = max(min(left_max[i], right_max[i]) - height[i], 0)

“这类接雨水问题，核心是每个位置的水量由左最大和右最大共同决定，是一个典型的‘两端约束中间’的问题，非常适合双指针。用双指针从两端往中间走，维护 left_max 和 right_max。每次比较两端高度，矮的一侧一定可以结算，因为水位上限已经被这边的最大值和对侧当前高度锁死了，未来对侧再怎么变高都不会改变这一格的结果。这样就可以一边移动指针一边累加答案，只遍历一次，时间 O(N)，只用常数级额外空间，比预先建两个 max 数组更节省内存，也更符合工程上的高效实现。”


这种“只要知道哪边更短，就能确定这一边的答案”的单调性，特别适合用两个指针夹逼，因为每一步都能确认至少一侧的位置再也不受对侧未来变化的影响。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681703839 | 2025-11-30 17:26:06 CST | python3 | Accepted | 4 ms | 19 MB |
| 681919444 | 2025-12-01 16:53:11 CST | python3 | Accepted | 3 ms | 18.9 MB |

### 未通过提交代码
(所有提交均已通过)

## 两数之和 (`two-sum`)

- 题目链接：https://leetcode.cn/problems/two-sum/
- 难度：Easy
- 标签：数组, 哈希表
- 总提交次数：2
- 最近提交时间：2025-11-30 15:07:18 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681222981 | 2025-11-28 08:44:34 CST | python3 | Accepted | 0 ms | 18.7 MB |
| 681664260 | 2025-11-30 15:07:18 CST | python3 | Accepted | 0 ms | 18.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 排序数组 (`sort-an-array`)

- 题目链接：https://leetcode.cn/problems/sort-an-array/
- 难度：Medium
- 标签：数组, 分治, 桶排序, 计数排序, 基数排序, 排序, 堆（优先队列）, 归并排序
- 总提交次数：2
- 最近提交时间：2025-11-28 18:37:27 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 681357945 | 2025-11-28 18:37:04 CST | python3 | Runtime Error | N/A | N/A |
| 681357980 | 2025-11-28 18:37:27 CST | python3 | Time Limit Exceeded | N/A | N/A |

### 未通过提交代码
#### 提交 681357945 · Runtime Error · 2025-11-28 18:37:04 CST · python3

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self._quick_sort(nums, 0, len(nums)-1)
        return nums

    def _quick_sort(self, nums, left, right):
        if left >= right:
            return
        pivot_idx = self._partition(nums, left, right)
        self._quick_sort(nums, left, pivot_idx - 1)
        self._quick_sort(nums, pivot_idx + 1, right)
    
    def _partition(self, nums, left, right):
        random_pivot_idx = random.randomint(left, right)
        nums[random_pivot_idx], nums[right] = nums[right], nums[random_pivot_idx]
        pivot = nums[right]
        i = left
        for j in range(left, right):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i
```

#### 提交 681357980 · Time Limit Exceeded · 2025-11-28 18:37:27 CST · python3

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self._quick_sort(nums, 0, len(nums)-1)
        return nums

    def _quick_sort(self, nums, left, right):
        if left >= right:
            return
        pivot_idx = self._partition(nums, left, right)
        self._quick_sort(nums, left, pivot_idx - 1)
        self._quick_sort(nums, pivot_idx + 1, right)
    
    def _partition(self, nums, left, right):
        random_pivot_idx = random.randint(left, right)
        nums[random_pivot_idx], nums[right] = nums[right], nums[random_pivot_idx]
        pivot = nums[right]
        i = left
        for j in range(left, right):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i
```


## 找到字符串中所有字母异位词 (`find-all-anagrams-in-a-string`)

- 题目链接：https://leetcode.cn/problems/find-all-anagrams-in-a-string/
- 难度：Medium
- 标签：哈希表, 字符串, 滑动窗口
- 总提交次数：21
- 最近提交时间：2025-11-28 14:38:26 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-08-08 08:07:24 CST

```markdown
卡点1：写窗口收缩while更新条件时又不知道怎么写了，是写判断valid==len(need)，还是写right-left+1 > len(p)？其实就是混淆了固定长度窗口和可变长度窗口
错误1：符合条件进行添加结果时，写错valid和len(need)的判断，应该是valid和len(need)
错误2：忘记收缩窗口 left += 1
错误3：收集结果的判断应该放在while循环外，保证检查是当前窗口
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 650942994 | 2025-08-08 08:04:25 CST | python3 | Accepted | 39 ms | 18.1 MB |
| 658774437 | 2025-09-02 15:46:09 CST | python3 | Wrong Answer | N/A | N/A |
| 658777014 | 2025-09-02 15:51:22 CST | python3 | Wrong Answer | N/A | N/A |
| 658778152 | 2025-09-02 15:53:40 CST | python3 | Wrong Answer | N/A | N/A |
| 658782294 | 2025-09-02 16:01:58 CST | python3 | Wrong Answer | N/A | N/A |
| 658784183 | 2025-09-02 16:05:53 CST | python3 | Accepted | 55 ms | 18.4 MB |
| 660895002 | 2025-09-09 10:13:54 CST | python3 | Wrong Answer | N/A | N/A |
| 660896828 | 2025-09-09 10:18:20 CST | python3 | Runtime Error | N/A | N/A |
| 660896937 | 2025-09-09 10:18:34 CST | python3 | Accepted | 45 ms | 18.4 MB |
| 663109265 | 2025-09-16 08:40:13 CST | python3 | Wrong Answer | N/A | N/A |
| 663109284 | 2025-09-16 08:40:21 CST | python3 | Accepted | 39 ms | 18.2 MB |
| 663109471 | 2025-09-16 08:41:55 CST | python3 | Runtime Error | N/A | N/A |
| 663109495 | 2025-09-16 08:42:08 CST | python3 | Accepted | 39 ms | 18.2 MB |
| 663184987 | 2025-09-16 12:35:34 CST | python3 | Wrong Answer | N/A | N/A |
| 663185043 | 2025-09-16 12:36:00 CST | python3 | Accepted | 39 ms | 18.2 MB |
| 681291741 | 2025-11-28 14:29:26 CST | python3 | Runtime Error | N/A | N/A |
| 681291779 | 2025-11-28 14:29:34 CST | python3 | Runtime Error | N/A | N/A |
| 681292140 | 2025-11-28 14:30:57 CST | python3 | Wrong Answer | N/A | N/A |
| 681292454 | 2025-11-28 14:32:11 CST | python3 | Accepted | 35 ms | 18.1 MB |
| 681294050 | 2025-11-28 14:38:14 CST | python3 | Runtime Error | N/A | N/A |
| 681294113 | 2025-11-28 14:38:26 CST | python3 | Accepted | 39 ms | 18.2 MB |

### 未通过提交代码
#### 提交 658774437 · Wrong Answer · 2025-09-02 15:46:09 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        valid = 0
        left = 0
        res = []
        for right, r_char in enumerate(s):
            window[r_char] += 1
            if r_char in need and window[r_char] == need[r_char]:
                valid += 1
            if valid == len(need):
                res.append(left)
                l_char = s[left]
                if window[l_char] == need[l_char]:
                    valid -= 1
                left += 1
        return res
```

#### 提交 658777014 · Wrong Answer · 2025-09-02 15:51:22 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        valid = 0
        left = 0
        res = []
        for right, r_char in enumerate(s):
            window[r_char] += 1
            if r_char in need and window[r_char] == need[r_char]:
                valid += 1
            if right - left + 1 > len(p):
                if valid == len(need):
                    res.append(left)
                l_char = s[left]
                if window[l_char] == need[l_char]:
                    valid -= 1
                left += 1
        return res
```

#### 提交 658778152 · Wrong Answer · 2025-09-02 15:53:40 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        valid = 0
        left = 0
        res = []
        for right, r_char in enumerate(s):
            window[r_char] += 1
            if r_char in need and window[r_char] == need[r_char]:
                valid += 1
            if right - left + 1 > len(p):
                if valid == len(need):
                    res.append(left)
                    l_char = s[left]
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    left += 1
        return res
```

#### 提交 658782294 · Wrong Answer · 2025-09-02 16:01:58 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        valid = 0
        left = 0
        res = []
        for right, r_char in enumerate(s):
            window[r_char] += 1
            if r_char in need and window[r_char] == need[r_char]:
                valid += 1
            if right - left + 1 == len(p):
                if valid == len(need):
                    res.append(left)
                l_char = s[left]
                if window[l_char] == need[l_char]:
                    valid -= 1
                left += 1
        return res
```

#### 提交 660895002 · Wrong Answer · 2025-09-09 10:13:54 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        start = []
        left = valid = 0

        for right, r_char in enumerate(p):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            while right - left + 1 >= len(p):
                if valid == len(need):
                    start.append(left)
                l_char = p[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        
        return start
```

#### 提交 660896828 · Runtime Error · 2025-09-09 10:18:20 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        start = []
        left = valid = 0

        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            while right - left + 1 >= len(p):
                if valid == len(need):
                    start.append(left)
                l_char = p[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        
        return start
```

#### 提交 663109265 · Wrong Answer · 2025-09-16 08:40:13 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        start = []
        left = valid = 0

        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            while right - left + 1 > len(p):
                if valid == len(need):
                    start.append(left)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        
        return start
```

#### 提交 663109471 · Runtime Error · 2025-09-16 08:41:55 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        start = []
        left = valid = 0

        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            #while right - left + 1 >= len(p):
            if >= len(p)-1:
                if valid == len(need):
                    start.append(left)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        
        return start
```

#### 提交 663184987 · Wrong Answer · 2025-09-16 12:35:34 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        res = []
        left = valid = 0
        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            if right >= len(p):
                if valid == len(need):
                    res.append(left)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        return res
```

#### 提交 681291741 · Runtime Error · 2025-11-28 14:29:26 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        left = valid = 0
        res = []
        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] = need[r_char]:
                    valid += 1
        
            if right >= len(s) - 1:
                if  valid == len(need):
                    res.append(i)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        return res
```

#### 提交 681291779 · Runtime Error · 2025-11-28 14:29:34 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        left = valid = 0
        res = []
        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
        
            if right >= len(s) - 1:
                if  valid == len(need):
                    res.append(i)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        return res
```

#### 提交 681292140 · Wrong Answer · 2025-11-28 14:30:57 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        left = valid = 0
        res = []
        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
        
            if right >= len(s) - 1:
                if valid == len(need):
                    res.append(left)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1

        return res
```

#### 提交 681294050 · Runtime Error · 2025-11-28 14:38:14 CST · python3

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = collections.Counter(p)
        window = collections.defaultdict(int)
        valid = left = 0
        res = 0
        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1

            if right >= len(p) - 1:
                if valid == len(need):
                    res.append(left)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        return res
```


## 反转字符串中的单词 (`reverse-words-in-a-string`)

- 题目链接：https://leetcode.cn/problems/reverse-words-in-a-string/
- 难度：Medium
- 标签：双指针, 字符串
- 总提交次数：15
- 最近提交时间：2025-11-28 14:18:38 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 665123102 | 2025-09-22 21:35:53 CST | python3 | Runtime Error | N/A | N/A |
| 665123600 | 2025-09-22 21:37:17 CST | python3 | Wrong Answer | N/A | N/A |
| 665123775 | 2025-09-22 21:37:43 CST | python3 | Accepted | 15 ms | 17.7 MB |
| 665187629 | 2025-09-23 07:21:03 CST | python3 | Runtime Error | N/A | N/A |
| 665187638 | 2025-09-23 07:21:27 CST | python3 | Runtime Error | N/A | N/A |
| 665187686 | 2025-09-23 07:22:50 CST | python3 | Wrong Answer | N/A | N/A |
| 665187736 | 2025-09-23 07:25:18 CST | python3 | Runtime Error | N/A | N/A |
| 665187742 | 2025-09-23 07:25:38 CST | python3 | Wrong Answer | N/A | N/A |
| 665188587 | 2025-09-23 07:33:52 CST | python3 | Accepted | 7 ms | 17.6 MB |
| 665188708 | 2025-09-23 07:38:22 CST | python3 | Accepted | 7 ms | 17.6 MB |
| 665505939 | 2025-09-24 08:22:39 CST | python3 | Runtime Error | N/A | N/A |
| 665505981 | 2025-09-24 08:23:19 CST | python3 | Runtime Error | N/A | N/A |
| 665506007 | 2025-09-24 08:23:38 CST | python3 | Wrong Answer | N/A | N/A |
| 665506455 | 2025-09-24 08:28:59 CST | python3 | Accepted | 7 ms | 17.8 MB |
| 681289036 | 2025-11-28 14:18:38 CST | python3 | Accepted | 0 ms | 17.5 MB |

### 未通过提交代码
#### 提交 665123102 · Runtime Error · 2025-09-22 21:35:53 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        clean_list = []
        i = 0
        n = len(s)
        # 清除前置空格和中间的连续空格
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            if clean_list:
                clean_list.append(' ')
            
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
        self.reverse_string(clean_list, 0, len(clean_list) - 1)
        start = 0
        for end in range(len(clean_list)+1):
            if clean_list[end] == ' ' or end == len(clean_list):
                self.reverse_string(clean_list, start, end-1)
                start = end+1
        return ' '.join(clean_list)

    def reverse_string(self, char_list, start, end):
        while start < end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1
```

#### 提交 665123600 · Wrong Answer · 2025-09-22 21:37:17 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        clean_list = []
        i = 0
        n = len(s)
        # 清除前置空格和中间的连续空格
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            if clean_list:
                clean_list.append(' ')
            
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
        self.reverse_string(clean_list, 0, len(clean_list) - 1)
        start = 0
        for end in range(len(clean_list)+1):
            if end == len(clean_list) or clean_list[end] == ' ':
                self.reverse_string(clean_list, start, end-1)
                start = end+1
        return ' '.join(clean_list)

    def reverse_string(self, char_list, start, end):
        while start < end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1
```

#### 提交 665187629 · Runtime Error · 2025-09-23 07:21:03 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        clean_list = []
        i = 0
        n = len(s)
        # 去除前导空格和中间的连续空格
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            if clean_list:
                clean_list.append(' ')
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
        self.reverse_string(clean_list, 0, n)
        start = 0
        for end in range(len(clean_list) + 1):
            if end != len(clean_list) and clean_list[end] == '':
                self.reverse_string(clean_list, start, end-1)
            start = end+1
        return ''.join(clean_list)

    def reverse_string(self, char_list: list, start, end):
        while start < end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1
```

#### 提交 665187638 · Runtime Error · 2025-09-23 07:21:27 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        clean_list = []
        i = 0
        n = len(s)
        # 去除前导空格和中间的连续空格
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            if clean_list:
                clean_list.append(' ')
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
        self.reverse_string(clean_list, 0, len(clean_list))
        start = 0
        for end in range(len(clean_list) + 1):
            if end != len(clean_list) and clean_list[end] == '':
                self.reverse_string(clean_list, start, end-1)
            start = end+1
        return ''.join(clean_list)

    def reverse_string(self, char_list: list, start, end):
        while start < end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1
```

#### 提交 665187686 · Wrong Answer · 2025-09-23 07:22:50 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        clean_list = []
        i = 0
        n = len(s)
        # 去除前导空格和中间的连续空格
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            if clean_list:
                clean_list.append(' ')
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
        self.reverse_string(clean_list, 0, len(clean_list)-1)
        start = 0
        for end in range(len(clean_list) + 1):
            if end != len(clean_list) and clean_list[end] == '':
                self.reverse_string(clean_list, start, end-1)
            start = end+1
        return ''.join(clean_list)

    def reverse_string(self, char_list: list, start, end):
        while start < end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1
```

#### 提交 665187736 · Runtime Error · 2025-09-23 07:25:18 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        clean_list = []
        i = 0
        n = len(s)
        # 去除前导空格和中间的连续空格
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            if clean_list:
                clean_list.append(' ')
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
        self.reverse_string(clean_list, 0, len(clean_list)-1)
        start = 0
        for end in range(len(clean_list) + 1):
            if end == len(clean_list) and clean_list[end] == ' ':
                self.reverse_string(clean_list, start, end-1)
            start = end+1
        return ''.join(clean_list)

    def reverse_string(self, char_list: list, start, end):
        while start < end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1
```

#### 提交 665187742 · Wrong Answer · 2025-09-23 07:25:38 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        clean_list = []
        i = 0
        n = len(s)
        # 去除前导空格和中间的连续空格
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            if clean_list:
                clean_list.append(' ')
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
        self.reverse_string(clean_list, 0, len(clean_list)-1)
        start = 0
        for end in range(len(clean_list) + 1):
            if end == len(clean_list) or clean_list[end] == ' ':
                self.reverse_string(clean_list, start, end-1)
            start = end+1
        return ''.join(clean_list)

    def reverse_string(self, char_list: list, start, end):
        while start < end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1
```

#### 提交 665505939 · Runtime Error · 2025-09-24 08:22:39 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # 去除前导空格和中间的连续空格
        clean_list = []
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            
            if clean_list:
                clean_list.append(' ')
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
            clean_len = len(clean_list)
            # 整体反转
            self.resverse_string(clean_list, 0, clean_len)
            start = 0
            for end in range(clean_len+1):
                if end == clean_len or clean_list[end] == ' ':
                    # 局部反转
                    self.reverse_string(clean_list, start, end-1)
                    start = end+1
            return ''.join(clean_list)
            

    def resverse_string(self, chars_list, start, end):
        while start < end:
            chars_list[start], chars_list[end] = chars_list[end], chars_list[start]
            start += 1
            end -= 1
```

#### 提交 665505981 · Runtime Error · 2025-09-24 08:23:19 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # 去除前导空格和中间的连续空格
        clean_list = []
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            
            if clean_list:
                clean_list.append(' ')
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
            clean_len = len(clean_list)
            # 整体反转
            self.resverse_string(clean_list, 0, clean_len-1)
            start = 0
            for end in range(clean_len+1):
                if end == clean_len or clean_list[end] == ' ':
                    # 局部反转
                    self.reverse_string(clean_list, start, end-1)
                    start = end+1
            return ''.join(clean_list)
            

    def resverse_string(self, chars_list, start, end):
        while start < end:
            chars_list[start], chars_list[end] = chars_list[end], chars_list[start]
            start += 1
            end -= 1
```

#### 提交 665506007 · Wrong Answer · 2025-09-24 08:23:38 CST · python3

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # 去除前导空格和中间的连续空格
        clean_list = []
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i] == ' ':
                i += 1
            if i == n:
                break
            
            if clean_list:
                clean_list.append(' ')
            while i < n and s[i] != ' ':
                clean_list.append(s[i])
                i += 1
            clean_len = len(clean_list)
            # 整体反转
            self.resverse_string(clean_list, 0, clean_len-1)
            start = 0
            for end in range(clean_len+1):
                if end == clean_len or clean_list[end] == ' ':
                    # 局部反转
                    self.resverse_string(clean_list, start, end-1)
                    start = end+1
            return ''.join(clean_list)
            

    def resverse_string(self, chars_list, start, end):
        while start < end:
            chars_list[start], chars_list[end] = chars_list[end], chars_list[start]
            start += 1
            end -= 1
```


## 存在重复元素 III (`7WqeDu`)

- 题目链接：https://leetcode.cn/problems/7WqeDu/
- 难度：Medium
- 标签：数组, 桶排序, 有序集合, 排序, 滑动窗口
- 总提交次数：2
- 最近提交时间：2025-11-27 10:33:39 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680999552 | 2025-11-27 10:30:42 CST | python3 | Wrong Answer | N/A | N/A |
| 681000389 | 2025-11-27 10:33:39 CST | python3 | Accepted | 127 ms | 19.6 MB |

### 未通过提交代码
#### 提交 680999552 · Wrong Answer · 2025-11-27 10:30:42 CST · python3

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        window = []
        for i, num in enumerate(nums):
            if i > k:
                old_val = nums[i-k-1]
                idx = bisect.bisect_left(window, old_val)
                window.pop(old_val)
            target_val = num - t
            idx = bisect.bisect_left(window, target_val)
            if idx < len(window) and window[idx] <= num + t:
                return True
            bisect.insort(window, num)
        return False
```


## 存在重复元素 III (`contains-duplicate-iii`)

- 题目链接：https://leetcode.cn/problems/contains-duplicate-iii/
- 难度：Hard
- 标签：数组, 桶排序, 有序集合, 排序, 滑动窗口
- 总提交次数：9
- 最近提交时间：2025-11-27 10:20:15 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-27 10:04:00 CST

```markdown
有序窗口+二分查找
1、维护一个窗口长度小于等于indexDiff的窗口
2、对于每个新元素 u：
	 判断窗口中是否存在一个元素 v 满足abs(u - v) <= valueDiff
	 等价于在窗口中判断是否存在 u - valueDiff <= v <= u + valueDiff
3、使用二分查找在窗口中查找 第一个大于等于 u - valueDiff 的数的位置，如果这个位置小于窗口长度且数值小于等于 u + valueDiff 则说明找到了
如果当前 u 不满足，则使用bisect.insort 将 u 插入到窗口中

一句话总结：“使用滑动窗口锁定了‘合法的索引范围’，而有序集合配合二分查找，实现了在‘合法的数值范围’内进行 $O(\log k)$ 的极速定位，而不是 $O(k)$ 的盲目扫描。”
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 654141063 | 2025-08-18 15:54:17 CST | python3 | Wrong Answer | N/A | N/A |
| 654144159 | 2025-08-18 16:00:40 CST | python3 | Accepted | 375 ms | 30.7 MB |
| 654878768 | 2025-08-20 16:58:25 CST | python3 | Runtime Error | N/A | N/A |
| 654879014 | 2025-08-20 16:58:53 CST | python3 | Accepted | 135 ms | 36 MB |
| 661329696 | 2025-09-10 14:24:04 CST | python3 | Runtime Error | N/A | N/A |
| 661329780 | 2025-09-10 14:24:18 CST | python3 | Runtime Error | N/A | N/A |
| 661329863 | 2025-09-10 14:24:30 CST | python3 | Accepted | 189 ms | 36.1 MB |
| 680991752 | 2025-11-27 10:19:47 CST | python3 | Wrong Answer | N/A | N/A |
| 680991882 | 2025-11-27 10:20:15 CST | python3 | Accepted | 264 ms | 30.6 MB |

### 未通过提交代码
#### 提交 654141063 · Wrong Answer · 2025-08-18 15:54:17 CST · python3

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        window: list[int] = []
        for i, r_num in enumerate(nums):
            pos = bisect_left(window, r_num-indexDiff)
            if pos < len(window) and window[pos] <= r_num+valueDiff:
                return True
            insort(window, r_num)
            if i >= indexDiff:
                l_num = nums[i-indexDiff]
                del_pos = bisect_left(window, l_num)
                if del_pos < len(window) and window[del_pos] == l_num:
                    window.pop(del_pos)
        return False
```

#### 提交 654878768 · Runtime Error · 2025-08-20 16:58:25 CST · python3

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        buckets = {}
        w = valueDiff + 1
        for i, x in enumerate(nums):
            bucket_id = x // w
            if bucket_id in buckets:
                return True
            if bucket_id - 1 in buckets and abs(buckets[bucket_id - 1]-x) <= valueDiff:
                return True
            if bucket_id + 1 in buckets and abs(buckets[bucket_id + 1]-x) <= valueDiff:
                return True
            buckets[bucket_id] = x
            if i >= k:
                old_bucket_id = nums[i-k] // w
                del buckets[old_bucket_id]
        return False
```

#### 提交 661329696 · Runtime Error · 2025-09-10 14:24:04 CST · python3

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int):
        # 通过滑动窗口来实现判断离得近
        # 通过分桶来实现判断长得像，桶的宽度是valueDiff+1，这样长得像的两个数一定在同一个桶或者相邻桶
        left = 0
        buckets = {}
        w = valueDiff + 1
        for i, num in enumerate(nums):
            bucket_id = num // w
            if bucket_id in buckets:
                return True
            if bucket_id - 1 in buckets and abs(buckets[bucket_id-1] - num) <= valueDiff:
                return True
            if bucket_id + 1 in buckets and abs(buckets[bucket_id+1] - num) <= valueDiff:
                return True
            buckets[bucket_id] = num
            if i >= indexDiff:
                old_num = num[i-indexDiff]
                old_id = old_num // w
                if old_id in buckets and buckets[old_id] == old_num:
                    del buckets[old_id]
        retrun False
```

#### 提交 661329780 · Runtime Error · 2025-09-10 14:24:18 CST · python3

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int):
        # 通过滑动窗口来实现判断离得近
        # 通过分桶来实现判断长得像，桶的宽度是valueDiff+1，这样长得像的两个数一定在同一个桶或者相邻桶
        left = 0
        buckets = {}
        w = valueDiff + 1
        for i, num in enumerate(nums):
            bucket_id = num // w
            if bucket_id in buckets:
                return True
            if bucket_id - 1 in buckets and abs(buckets[bucket_id-1] - num) <= valueDiff:
                return True
            if bucket_id + 1 in buckets and abs(buckets[bucket_id+1] - num) <= valueDiff:
                return True
            buckets[bucket_id] = num
            if i >= indexDiff:
                old_num = num[i-indexDiff]
                old_id = old_num // w
                if old_id in buckets and buckets[old_id] == old_num:
                    del buckets[old_id]
        return False
```

#### 提交 680991752 · Wrong Answer · 2025-11-27 10:19:47 CST · python3

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        window = []
        for i, num in enumerate(nums):
            # 维护窗口大小，如果 i 大于indexDiff，则将最老的元素从窗口中剔除，最老的元素是 nums[i-k-1]
            if i > indexDiff:
                old_value = nums[i-indexDiff-1]
                idx = bisect.bisect_left(window, old_value)
                window.pop(idx)
            
            target_val = num - valueDiff
            idx = bisect.bisect_left(window, target_val)
            if idx < len(window) and window[idx] <= valueDiff:
                return True
            bisect.insort(window, num)
        return False
```


## 存在重复元素 II (`contains-duplicate-ii`)

- 题目链接：https://leetcode.cn/problems/contains-duplicate-ii/
- 难度：Easy
- 标签：数组, 哈希表, 滑动窗口
- 总提交次数：5
- 最近提交时间：2025-11-27 08:40:23 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 653454646 | 2025-08-16 07:27:24 CST | python3 | Accepted | 51 ms | 30.2 MB |
| 658425933 | 2025-09-01 15:37:37 CST | python3 | Accepted | 39 ms | 30 MB |
| 658557214 | 2025-09-01 21:34:38 CST | python3 | Accepted | 35 ms | 29.9 MB |
| 661310713 | 2025-09-10 13:07:50 CST | python3 | Accepted | 27 ms | 35.9 MB |
| 680974043 | 2025-11-27 08:40:23 CST | python3 | Accepted | 35 ms | 36 MB |

### 未通过提交代码
(所有提交均已通过)

## 存在重复元素 (`contains-duplicate`)

- 题目链接：https://leetcode.cn/problems/contains-duplicate/
- 难度：Easy
- 标签：数组, 哈希表, 排序
- 总提交次数：2
- 最近提交时间：2025-11-27 08:35:22 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680973655 | 2025-11-27 08:34:14 CST | python3 | Wrong Answer | N/A | N/A |
| 680973718 | 2025-11-27 08:35:22 CST | python3 | Accepted | 10 ms | 31 MB |

### 未通过提交代码
#### 提交 680973655 · Wrong Answer · 2025-11-27 08:34:14 CST · python3

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(set(nums)) == len(nums)
```


## 我的日程安排表 I (`my-calendar-i`)

- 题目链接：https://leetcode.cn/problems/my-calendar-i/
- 难度：Medium
- 标签：设计, 线段树, 数组, 二分查找, 有序集合
- 总提交次数：1
- 最近提交时间：2025-11-27 08:10:44 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-27 08:13:56 CST

```markdown
* 这类日程/区间冲突问题，本质上是一个‘在一堆区间中，快速找到和新区间相邻的区间’的问题。
* 如果我们不用排序，每次都全表扫描，插入第 N 个区间要 O(N)，整体接近 O(N²)。
* 我选择维护一个按 start 有序的列表，然后用二分查找在 O(log N) 时间内找到新区间的插入位置 i。
* 因为整体有序，可能发生冲突的只会是前一个区间和后一个区间，所以只需要 O(1) 做局部检查。
* 这就是有序集合的典型使用场景：把全局问题变成局部问题，时间复杂度从 O(N) 降到 O(log N + 1)，也更利于后续扩展成支持查询前驱/后继、范围查询等功能。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680972490 | 2025-11-27 08:10:44 CST | python3 | Accepted | 24 ms | 18.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 课程表 II (`course-schedule-ii`)

- 题目链接：https://leetcode.cn/problems/course-schedule-ii/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 图, 拓扑排序
- 总提交次数：4
- 最近提交时间：2025-11-26 13:59:34 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-08 10:18:42 CST

```markdown
我们需要把“有先后依赖的任务”线性化，这正是拓扑排序的定义与用途
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 676627244 | 2025-11-08 10:25:54 CST | python3 | Runtime Error | N/A | N/A |
| 676627264 | 2025-11-08 10:26:03 CST | python3 | Wrong Answer | N/A | N/A |
| 676627484 | 2025-11-08 10:27:33 CST | python3 | Accepted | 2 ms | 18.9 MB |
| 680795169 | 2025-11-26 13:59:34 CST | python3 | Accepted | 0 ms | 18.6 MB |

### 未通过提交代码
#### 提交 676627244 · Runtime Error · 2025-11-08 10:25:54 CST · python3

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 初始化入度数组和邻接表
        in_degree = [0] * numCourses
        adj_list = [[] for _ in range(numCourses)]

        # 遍历依赖关系，构建入度数组和邻接表
        for course, prereq in prerequisites:
            in_degree[course] += 1
            adj_list[prereq].append(course)
        
        # 初始化双向队列，将入度为0的节点加入队列
        deque = collections.deque()
        for course in range(numCourses):
            if in_degree[course] == 0:
                deque.append(course)
        
        # 开始 BFS
        res = []
        while deque:
            current_course = deque.popleft()
            res.append()
            for next_course in adj_list[current_course]:
                if in_degree[next_course] == 0:
                    deque.append(next_course)
        
        if len(res) == numCourses:
            return res
        else:
            return []
```

#### 提交 676627264 · Wrong Answer · 2025-11-08 10:26:03 CST · python3

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 初始化入度数组和邻接表
        in_degree = [0] * numCourses
        adj_list = [[] for _ in range(numCourses)]

        # 遍历依赖关系，构建入度数组和邻接表
        for course, prereq in prerequisites:
            in_degree[course] += 1
            adj_list[prereq].append(course)
        
        # 初始化双向队列，将入度为0的节点加入队列
        deque = collections.deque()
        for course in range(numCourses):
            if in_degree[course] == 0:
                deque.append(course)
        
        # 开始 BFS
        res = []
        while deque:
            current_course = deque.popleft()
            res.append(current_course)
            for next_course in adj_list[current_course]:
                if in_degree[next_course] == 0:
                    deque.append(next_course)
        
        if len(res) == numCourses:
            return res
        else:
            return []
```


## 课程表 (`course-schedule`)

- 题目链接：https://leetcode.cn/problems/course-schedule/
- 难度：Medium
- 标签：深度优先搜索, 广度优先搜索, 图, 拓扑排序
- 总提交次数：6
- 最近提交时间：2025-11-26 10:23:37 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-26 10:01:36 CST

```markdown
一句话结论
* 这题本质是“把带前后依赖的任务排成一条线”的问题；拓扑排序的定义就是在有向无环图里，找到一个满足所有依赖的线性序。DAG 与“存在拓扑序”是等价的，因此拓扑排序是最贴合、最省事的解法。

正确性来自“剥洋葱”与偏序理论：
* 关键事实：DAG 一定存在至少一个入度为 0 的点（否则沿依赖一直往前追必然成环）
* 安全选择：取任何入度为 0 的节点放到序列最前，不会违反依赖（它不需要别人）
* 迭代至尽：不断移除入度 0 的节点并更新入度，若能移完所有节点，得到合法顺序
* 若有环：环内节点彼此依赖，入度永远降不到 0，过程卡住，正好判定“不可完成”

30 秒可直接口述
* 这是把前置依赖线性化的问题，课程图是有向图。拓扑排序专门解决“偏序能否线性扩展”。我用 Kahn 算法：统计入度，把所有入度为 0 的课入队，不断出队、削减后继的入度，新变成 0 的继续入队。能处理完所有节点就无环且可完成，否则有环。这个方法线性复杂度，还能直接产出一个合法顺序，完全匹配题目场景。

复杂度
* 时间：O(V+E)，V 为课程数，E 为先修关系数。
* 空间：O(V+E)，邻接表与入度表。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 676430554 | 2025-11-07 10:02:22 CST | python3 | Runtime Error | N/A | N/A |
| 676430638 | 2025-11-07 10:02:48 CST | python3 | Accepted | 7 ms | 18.5 MB |
| 676432739 | 2025-11-07 10:13:23 CST | python3 | Accepted | 0 ms | 18.7 MB |
| 676475184 | 2025-11-07 13:48:21 CST | python3 | Runtime Error | N/A | N/A |
| 676475278 | 2025-11-07 13:48:53 CST | python3 | Accepted | 0 ms | 18.7 MB |
| 680747631 | 2025-11-26 10:23:37 CST | python3 | Accepted | 7 ms | 18.8 MB |

### 未通过提交代码
#### 提交 676430554 · Runtime Error · 2025-11-07 10:02:22 CST · python3

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 初始化入度数组和邻接表
        in_degree = [0] * numCourses
        adj_list = [[] for _ in numCourses]

        # 遍历依赖关系，构建入度数组和邻接表
        for course, prereq in prerequisites:
            in_degree[course] += 1
            adj_list[prereq].append(course)
        
        # 初始化队列，将所有入度为0的课程加入队列
        queue = collections.deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        # BFS过程
        course_taken = 0
        while queue:
            current_course = queue.popleft()
            course_taken += 1
            for next_course in adj_list[current_course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
    
        return course_taken == numCourses
```

#### 提交 676475184 · Runtime Error · 2025-11-07 13:48:21 CST · python3

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 初始化入度数组和邻接表
        in_degree = [0] * numCourses
        adj_list = [[] for _ in range(numCourses)]

        # 遍历连接关系，构建入度数组和邻接表
        for course, prereq in prerequisites:
            in_degree[course] += 1
            adj_list[prereq].append(course)
        
        # 将入度为0的课程加入队列
        queue = collections.deque()
        for i in range(in_degree):
            if in_degree[i] == 0:
                queue.append(i)
        
        # 开始 BFS
        course_taken = 0
        while queue:
            current_course = queue.popleft()
            course_taken += 1
            for next_course in adj_list[current_course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return course_taken == numCourses
```


## 二叉树的右视图 (`WNC0Lk`)

- 题目链接：https://leetcode.cn/problems/WNC0Lk/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：2
- 最近提交时间：2025-11-26 08:52:36 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680725065 | 2025-11-26 08:52:23 CST | python3 | Runtime Error | N/A | N/A |
| 680725087 | 2025-11-26 08:52:36 CST | python3 | Accepted | 43 ms | 17.4 MB |

### 未通过提交代码
#### 提交 680725065 · Runtime Error · 2025-11-26 08:52:23 CST · python3

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        dq = collections.deque([root])
        res = []
        while dq:
            level_size = len(dq)
            for _ in range(level_size):
                node = dq.popleft()
                if i == level_size-1:
                    res.append(node.val)
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
        return res
```


## 二叉树的右视图 (`binary-tree-right-side-view`)

- 题目链接：https://leetcode.cn/problems/binary-tree-right-side-view/
- 难度：Medium
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：1
- 最近提交时间：2025-11-26 08:46:18 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680724483 | 2025-11-26 08:46:18 CST | python3 | Accepted | 0 ms | 17.5 MB |

### 未通过提交代码
(所有提交均已通过)

## 计算二叉树的深度 (`er-cha-shu-de-shen-du-lcof`)

- 题目链接：https://leetcode.cn/problems/er-cha-shu-de-shen-du-lcof/
- 难度：Easy
- 标签：树, 深度优先搜索, 广度优先搜索, 二叉树
- 总提交次数：1
- 最近提交时间：2025-11-26 08:38:44 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680723842 | 2025-11-26 08:38:44 CST | python3 | Accepted | 3 ms | 18.5 MB |

### 未通过提交代码
(所有提交均已通过)

## 滑动窗口最大值 (`sliding-window-maximum`)

- 题目链接：https://leetcode.cn/problems/sliding-window-maximum/
- 难度：Hard
- 标签：队列, 数组, 滑动窗口, 单调队列, 堆（优先队列）
- 总提交次数：14
- 最近提交时间：2025-11-25 13:50:56 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-07 14:19:45 CST

```markdown
本质回答（道）：
* 问题的本质是流式的“右进左出”：每一步窗口右端加入一个新元素，左端可能移出一个旧元素。我们需要在这两个方向都以 O(1) 完成操作，并能快速给出当前最大值。
* 单调队列的本质是“维护极值的最小候选集”：把所有被新元素支配（更小且更早）的候选从队尾淘汰，只保留可能在将来某个窗口成为最大值的少数索引；过期的从队首淘汰。这样每个元素最多进队一次、出队一次，整体 O(n)。

30 秒面试口述版：
* 滑动窗口是“右进左出”，我们要在两端 O(1) 操作并随时拿最大。用**双端队列存索引并保持值递减**：队尾用于淘汰更弱者，队首用于淘汰过期者，队首即当前最大。这样每个元素最多进队一次、出队一次，总体 O(n)，空间 O(k)。相比堆/平衡树省去了 O(log k) 的维护和惰性删除，工程实现也最简洁高效。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 675444049 | 2025-11-03 10:52:19 CST | python3 | Wrong Answer | N/A | N/A |
| 675444577 | 2025-11-03 10:53:54 CST | python3 | Wrong Answer | N/A | N/A |
| 675445017 | 2025-11-03 10:55:15 CST | python3 | Wrong Answer | N/A | N/A |
| 675445776 | 2025-11-03 10:57:31 CST | python3 | Accepted | 163 ms | 30.6 MB |
| 675945726 | 2025-11-05 09:53:58 CST | python3 | Accepted | 163 ms | 30.9 MB |
| 676480799 | 2025-11-07 14:16:16 CST | python3 | Wrong Answer | N/A | N/A |
| 676481318 | 2025-11-07 14:18:25 CST | python3 | Wrong Answer | N/A | N/A |
| 676481534 | 2025-11-07 14:19:17 CST | python3 | Accepted | 196 ms | 31.6 MB |
| 677332756 | 2025-11-11 14:35:55 CST | python3 | Wrong Answer | N/A | N/A |
| 677332964 | 2025-11-11 14:36:32 CST | python3 | Accepted | 171 ms | 31.8 MB |
| 677584205 | 2025-11-12 13:54:31 CST | python3 | Runtime Error | N/A | N/A |
| 677584285 | 2025-11-12 13:54:52 CST | python3 | Wrong Answer | N/A | N/A |
| 677584425 | 2025-11-12 13:55:38 CST | python3 | Accepted | 163 ms | 30.9 MB |
| 680532123 | 2025-11-25 13:50:56 CST | python3 | Accepted | 164 ms | 30.8 MB |

### 未通过提交代码
#### 提交 675444049 · Wrong Answer · 2025-11-03 10:52:19 CST · python3

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        口述要点：用一个双端队列记录索引并保持对应值递减，队首始终是窗口内的最大值，每当新来一个元素时就将队尾所有小于等于这个元素值的索引剔除，剔除后将这个新的元素值索引入队，再将已经滑出窗口左边界的元素弹出，窗口形成手队首就是当前窗口最大值。
        """
        n = len(nums)
        if n == 0 or k <= 0:
            return []
        if n == 1:
            return nums
        if k > n:
            return max(nums)
        dq = deque()
        res = []
        for i, num in enumerate(nums):
            # 清理队尾：删除所有小于等于当前值的索引
            # 使用小于等于可以去重，相等时保留“更靠右”的索引，便于过期判断
            while dq and nums[dq[-1]] <= num:
                dq.pop()
            # 将当前值的索引追加到队尾
            dq.append(i)

            # 判断队首是否过期，如果过期就移出窗口
            if dq[0] <= i-k:
                dq.popleft()

            # 记录答案：从 k-1 位置开始，每前进一步都能得到一个最大值
            if i >= k-1:
                res.append(nums[dq[0]])
            return res
```

#### 提交 675444577 · Wrong Answer · 2025-11-03 10:53:54 CST · python3

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        口述要点：用一个双端队列记录索引并保持对应值递减，队首始终是窗口内的最大值，每当新来一个元素时就将队尾所有小于等于这个元素值的索引剔除，剔除后将这个新的元素值索引入队，再将已经滑出窗口左边界的元素弹出，窗口形成手队首就是当前窗口最大值。
        """
        n = len(nums)
        if n == 0 or k <= 0:
            return []
        if k == 1:
            return nums
        if k > n:
            return max(nums)
        dq = deque()
        res = []
        for i, num in enumerate(nums):
            # 清理队尾：删除所有小于等于当前值的索引
            # 使用小于等于可以去重，相等时保留“更靠右”的索引，便于过期判断
            while dq and nums[dq[-1]] <= num:
                dq.pop()
            # 将当前值的索引追加到队尾
            dq.append(i)

            # 判断队首是否过期，如果过期就移出窗口
            if dq[0] <= i-k:
                dq.popleft()

            # 记录答案：从 k-1 位置开始，每前进一步都能得到一个最大值
            if i >= k-1:
                res.append(nums[dq[0]])
            return res
```

#### 提交 675445017 · Wrong Answer · 2025-11-03 10:55:15 CST · python3

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        口述要点：用一个双端队列记录索引并保持对应值递减，队首始终是窗口内的最大值，每当新来一个元素时就将队尾所有小于等于这个元素值的索引剔除，剔除后将这个新的元素值索引入队，再将已经滑出窗口左边界的元素弹出，窗口形成手队首就是当前窗口最大值。
        """
        n = len(nums)
        if n == 0 or k <= 0:
            return []
        if k == 1:
            return nums
        if k > n:
            return [max(nums)]
        dq = deque()
        res = []
        for i, num in enumerate(nums):
            # 清理队尾：删除所有小于等于当前值的索引
            # 使用小于等于可以去重，相等时保留“更靠右”的索引，便于过期判断
            while dq and nums[dq[-1]] <= num:
                dq.pop()
            # 将当前值的索引追加到队尾
            dq.append(i)

            # 判断队首是否过期，如果过期就移出窗口
            if dq[0] <= i-k:
                dq.popleft()

            # 记录答案：从 k-1 位置开始，每前进一步都能得到一个最大值
            if i >= k-1:
                res.append(nums[dq[0]])
            return res
```

#### 提交 676480799 · Wrong Answer · 2025-11-07 14:16:16 CST · python3

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 1:
            return nums
        dq = collections.deque()
        res = []
        for i, num in enumerate(nums):
            while dq and dq[-1] <= num:
                dq.pop()
            dq.append(i)

            if dq[0] <= i-k:
                dq.popleft()
            
            if i >= k-1:
                res.append(dq[0])
        
        return res
```

#### 提交 676481318 · Wrong Answer · 2025-11-07 14:18:25 CST · python3

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 1:
            return nums
        dq = collections.deque()
        res = []
        for i, num in enumerate(nums):
            while dq and dq[-1] <= num:
                dq.pop()
            dq.append(i)

            if dq[0] <= i-k:
                dq.popleft()
            
            if i >= k-1:
                res.append(nums[dq[0]])
        
        return res
```

#### 提交 677332756 · Wrong Answer · 2025-11-11 14:35:55 CST · python3

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 1:
            return nums
        dq = collections.deque()
        res = []

        for i, num in enumerate(nums):
            while dq and nums[dq[-1]] <= num:
                dq.pop()
            dq.append(i)

            if dq[0] <= i-k:
                dq.popleft()
            
            if i >= k-1:
                res.append(nums[dq[0]])
```

#### 提交 677584205 · Runtime Error · 2025-11-12 13:54:31 CST · python3

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 1:
            return nums
        dq = collections.deque()
        res = []
        for i, num in nums:
            while dq and nums[dq[-1]] <= num:
                dq.pop()
            dq.append(i)

            # 从队列中弹出已经从窗口左侧滑出去的元素
            if dq[0] <= i-k:
                dq.popleft()

            if i >= k-1:
                res.append(dq[0])

        return res
```

#### 提交 677584285 · Wrong Answer · 2025-11-12 13:54:52 CST · python3

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 1:
            return nums
        dq = collections.deque()
        res = []
        for i, num in enumerate(nums):
            while dq and nums[dq[-1]] <= num:
                dq.pop()
            dq.append(i)

            # 从队列中弹出已经从窗口左侧滑出去的元素
            if dq[0] <= i-k:
                dq.popleft()

            if i >= k-1:
                res.append(dq[0])

        return res
```


## 二叉树的最近公共祖先 (`lowest-common-ancestor-of-a-binary-tree`)

- 题目链接：https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/
- 难度：Medium
- 标签：树, 深度优先搜索, 二叉树
- 总提交次数：2
- 最近提交时间：2025-11-25 12:50:41 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 680221648 | 2025-11-24 10:00:58 CST | python3 | Accepted | 56 ms | 21.6 MB |
| 680522250 | 2025-11-25 12:50:41 CST | python3 | Accepted | 81 ms | 21.7 MB |

### 未通过提交代码
(所有提交均已通过)

## 删除有序数组中的重复项 II (`remove-duplicates-from-sorted-array-ii`)

- 题目链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/
- 难度：Medium
- 标签：数组, 双指针
- 总提交次数：4
- 最近提交时间：2025-11-21 11:18:54 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 665553970 | 2025-09-24 11:15:34 CST | python3 | Accepted | 89 ms | 20.2 MB |
| 679647782 | 2025-11-21 11:18:07 CST | python3 | Runtime Error | N/A | N/A |
| 679647852 | 2025-11-21 11:18:23 CST | python3 | Runtime Error | N/A | N/A |
| 679648011 | 2025-11-21 11:18:54 CST | python3 | Accepted | 99 ms | 20.1 MB |

### 未通过提交代码
#### 提交 679647782 · Runtime Error · 2025-11-21 11:18:07 CST · python3

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return nums
        slow = 2 # 下一个写入位置
        for fast in in range(2, n):
            if nums[fast] != nums[slow-2]:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

#### 提交 679647852 · Runtime Error · 2025-11-21 11:18:23 CST · python3

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return nums
        slow = 2 # 下一个写入位置
        for fast in range(2, n):
            if nums[fast] != nums[slow-2]:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```


## 删除有序数组中的重复项 (`remove-duplicates-from-sorted-array`)

- 题目链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-array/
- 难度：Easy
- 标签：数组, 双指针
- 总提交次数：4
- 最近提交时间：2025-11-21 11:15:50 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-21 11:13:08 CST

```markdown
有序数组，相等的元素都挨着。
slow = 当前有效长度 = 下一个写入位置
fast指针先走，当nums[fast] 和 nums[slow-1] 不相等时，则说明遇到一个新元素，就将这个元素放到 slow的位置
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 649584282 | 2025-08-04 07:41:54 CST | python3 | Accepted | 0 ms | 18.8 MB |
| 679639481 | 2025-11-21 10:49:12 CST | python3 | Accepted | 6 ms | 18.5 MB |
| 679647021 | 2025-11-21 11:15:20 CST | python3 | Wrong Answer | N/A | N/A |
| 679647143 | 2025-11-21 11:15:50 CST | python3 | Accepted | 0 ms | 18.8 MB |

### 未通过提交代码
#### 提交 679647021 · Wrong Answer · 2025-11-21 11:15:20 CST · python3

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        slow = 1
        for fast in range(1, len(nums)):
            if nums[fast] != nums[slow-1]:
                nums[slow] = nums[fast]
        return slow
```


## 替换后的最长重复字符 (`longest-repeating-character-replacement`)

- 题目链接：https://leetcode.cn/problems/longest-repeating-character-replacement/
- 难度：Medium
- 标签：哈希表, 字符串, 滑动窗口
- 总提交次数：7
- 最近提交时间：2025-11-20 09:30:02 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-20 09:40:10 CST

```markdown
不变量：窗口长度-历史最大频次 <= k -> 窗口合法条件

for 循环中用 if 就可以，不需要用 while 的原因：因为我们要找的是‘最长’，只要达到一个长度，即使不合法，也不需要缩小，只需要碰运气看见后面能否遇到更长的

我们要在一个一维序列里找“满足某个约束条件的最长连续子串”，滑动窗口刚好就是“用两个指针在数组上滑动、随时维护当前窗口状态并判断是否合法”的通用模板，既能保证 O(N) 一次遍历，又能实时维护约束，所以特别适合这种“最长/最短连续区间 + 约束条件”的题。

滑动窗口的核心思想：
* 用 [left, right] 表示当前考察的子串；
* 右指针负责“扩张候选答案”；
* 左指针负责“修正不满足条件的情况”；
* 中间只维护 O(1) 级别的统计信息（频次、最大值、窗口长度等）。

“凡是涉及‘连续子数组/子串’且求‘最大/最小/定长’的问题，本质上都是在求一个区间。滑动窗口利用‘增量更新’的思想，避免了暴力解法中大量的重复计算，将 $O(N^2)$ 的搜索过程优化为了 $O(N)$ 的线性扫描。”
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 653150313 | 2025-08-15 07:45:22 CST | python3 | Accepted | 131 ms | 17.9 MB |
| 661261800 | 2025-09-10 10:37:06 CST | python3 | Wrong Answer | N/A | N/A |
| 661264449 | 2025-09-10 10:43:06 CST | python3 | Wrong Answer | N/A | N/A |
| 661264580 | 2025-09-10 10:43:23 CST | python3 | Accepted | 83 ms | 17.7 MB |
| 661275344 | 2025-09-10 11:07:18 CST | python3 | Accepted | 123 ms | 17.9 MB |
| 663040591 | 2025-09-15 21:47:55 CST | python3 | Accepted | 149 ms | 17.9 MB |
| 679380595 | 2025-11-20 09:30:02 CST | python3 | Accepted | 103 ms | 17.6 MB |

### 未通过提交代码
#### 提交 661261800 · Wrong Answer · 2025-09-10 10:37:06 CST · python3

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        max_history_count = 1
        char_counts = [0] * 26
        left = 0
        # 怎么确定该替换哪个？肯定是替换成窗口内出现次数最多的那个字符最划算
        # 怎么找窗口内出现次数最多的那个字符？其实不需要每次都找，只需要记录合法窗口内历史字符出现次数最多是多少就行
        for right, r_char in enumerate(s):
            char_index = ord(r_char) - ord('A')
            char_counts[char_index] += 1
            max_history_count = max(max_history_count, char_counts[char_index])
            while (right - left + 1) - max_history_count > k:
                l_char = s[left]
                l_char_index = ord(l_char) - ord('A')
                char_counts[l_char_index] -= 1
                left += 1
        return max_history_count + k if max_history_count != 1 else 1
```

#### 提交 661264449 · Wrong Answer · 2025-09-10 10:43:06 CST · python3

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        max_history_count = 1
        char_counts = [0] * 26
        left = 0
        # 怎么确定该替换哪个？肯定是替换成窗口内出现次数最多的那个字符最划算
        # 怎么找窗口内出现次数最多的那个字符？其实不需要每次都找，只需要记录合法窗口内历史字符出现次数最多是多少就行
        for right, r_char in enumerate(s):
            char_index = ord(r_char) - ord('A')
            char_counts[char_index] += 1
            max_history_count = max(max_history_count, char_counts[char_index])
            while (right - left + 1) - max_history_count > k:
                l_char = s[left]
                l_char_index = ord(l_char) - ord('A')
                char_counts[l_char_index] -= 1
                left += 1
        return min(max_history_count, len(s))
```


## 添加与搜索单词 - 数据结构设计 (`design-add-and-search-words-data-structure`)

- 题目链接：https://leetcode.cn/problems/design-add-and-search-words-data-structure/
- 难度：Medium
- 标签：深度优先搜索, 设计, 字典树, 字符串
- 总提交次数：21
- 最近提交时间：2025-11-19 14:15:17 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-15 14:01:48 CST

```markdown
为什么这样做（本质思路）
* 朴素做法：把所有单词丢在一个列表里，每次 search 全量匹配，时间 O(N*L)，当 N 大时面试官不会满意。
* 正解：字典树（Trie）。它把“共同前缀”合并在一起，检索某个模式串时能沿路径快速下钻；遇到 '.' 时做分支 DFS。这样大部分路径能被提前剪枝，平均性能好，数据结构也清晰易维护。

本质是“在动态词典上做带单字符通配符的前缀匹配”。Trie 把词表变成一个按前缀分层的状态机，精确匹配就是沿边走 O(L)。遇到 '.' 仅在当前节点对所有子边做局部 DFS，相当于小范围的 NFA （不确定的有穷自动机）扩展，能强力剪枝。这样查询复杂度主要依赖模式长度而非词库规模，插入/删除也都是 O(L)。相比列表/哈希集合需要对同长词逐一比对，Trie 在功能、复杂度和工程可维护性上更契合这道题。

不太熟练的是 dfs 的写法

总是写错 node.childs.values() 和 node.childs[ch]
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 677018551 | 2025-11-10 10:18:18 CST | python3 | Runtime Error | N/A | N/A |
| 677018616 | 2025-11-10 10:18:33 CST | python3 | Accepted | 1214 ms | 64.2 MB |
| 677065731 | 2025-11-10 13:55:55 CST | python3 | Wrong Answer | N/A | N/A |
| 677066413 | 2025-11-10 13:59:08 CST | python3 | Accepted | 1213 ms | 64.3 MB |
| 678231281 | 2025-11-15 12:13:52 CST | python3 | Runtime Error | N/A | N/A |
| 678231450 | 2025-11-15 12:15:11 CST | python3 | Runtime Error | N/A | N/A |
| 678231529 | 2025-11-15 12:15:50 CST | python3 | Runtime Error | N/A | N/A |
| 678231648 | 2025-11-15 12:16:47 CST | python3 | Runtime Error | N/A | N/A |
| 678232061 | 2025-11-15 12:20:12 CST | python3 | Runtime Error | N/A | N/A |
| 678232243 | 2025-11-15 12:21:43 CST | python3 | Accepted | 1158 ms | 64.4 MB |
| 678250210 | 2025-11-15 13:54:50 CST | python3 | Runtime Error | N/A | N/A |
| 678250252 | 2025-11-15 13:55:04 CST | python3 | Runtime Error | N/A | N/A |
| 678250518 | 2025-11-15 13:57:00 CST | python3 | Runtime Error | N/A | N/A |
| 678250788 | 2025-11-15 13:59:03 CST | python3 | Runtime Error | N/A | N/A |
| 678250819 | 2025-11-15 13:59:12 CST | python3 | Runtime Error | N/A | N/A |
| 678250997 | 2025-11-15 14:00:26 CST | python3 | Accepted | 1409 ms | 64.4 MB |
| 679188260 | 2025-11-19 14:11:23 CST | python3 | Runtime Error | N/A | N/A |
| 679188610 | 2025-11-19 14:12:36 CST | python3 | Runtime Error | N/A | N/A |
| 679188822 | 2025-11-19 14:13:20 CST | python3 | Runtime Error | N/A | N/A |
| 679188922 | 2025-11-19 14:13:39 CST | python3 | Wrong Answer | N/A | N/A |
| 679189400 | 2025-11-19 14:15:17 CST | python3 | Accepted | 1301 ms | 64.3 MB |

### 未通过提交代码
#### 提交 677018551 · Runtime Error · 2025-11-10 10:18:18 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.child:
                node.child[ch] = TrieNode()
            node = node.child[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, idx: int):
            if idx == len(word):
                return node.is_end
            ch = word[idx]
            if ch == '.':
                for child in node.child.values():
                    if dfs(child, idx+1):
                        return True
                return False
            else:
                if ch not in node.child:
                    return False
                return dfs(node.child[ch], idx+1)
        return dfs(self.root, 0)

        


class TrieNode(self):
    def __init__(self):
        self.child: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
```

#### 提交 677065731 · Wrong Answer · 2025-11-10 13:55:55 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        # 从根节点（空字符串）开始添加
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True


    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, idx: int):
            if idx == len(word):
                return node.is_end
            ch = word[idx]
            if ch == '.':
                for child in node.childs.values():
                    if dfs(child, idx+1):
                        return True
            else:
                if ch not in node.childs:
                    return False
                return dfs(node.childs[ch], idx+1)
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end: bool = False
```

#### 提交 678231281 · Runtime Error · 2025-11-15 12:13:52 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                self.childs[ch] = TrieNode()
            node = self.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    return dfs(child, idx+1)
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs[node[ch], idx+1]
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end: bool = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678231450 · Runtime Error · 2025-11-15 12:15:11 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = self.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    return dfs(child, idx+1)
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs[node[ch], idx+1]
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end: bool = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678231529 · Runtime Error · 2025-11-15 12:15:50 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    return dfs(child, idx+1)
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs[node[ch], idx+1]
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end: bool = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678231648 · Runtime Error · 2025-11-15 12:16:47 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    return dfs(child, idx+1)
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs(node.childs[ch], idx+1)
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end: bool = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678232061 · Runtime Error · 2025-11-15 12:20:12 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    if dfs(child, idx+1):
                        return True
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs(node.childs[ch], idx+1)
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end: bool = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678250210 · Runtime Error · 2025-11-15 13:54:50 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not int node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return True
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    if dfs(child, idx+1):
                        return True
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs(node, idx+1)
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678250252 · Runtime Error · 2025-11-15 13:55:04 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return True
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    if dfs(child, idx+1):
                        return True
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs(node, idx+1)
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678250518 · Runtime Error · 2025-11-15 13:57:00 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return True
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    if dfs(child, idx+1):
                        return True
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs(node, idx+1)
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678250788 · Runtime Error · 2025-11-15 13:59:03 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return True
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    if dfs(child, idx+1):
                        return True
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs(node.child[ch], idx+1)
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 678250819 · Runtime Error · 2025-11-15 13:59:12 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
            node = node.childs[ch]
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return True
            ch = word[idx]
            if ch == '.':
                for child in node.childs:
                    if dfs(child, idx+1):
                        return True
                return False
            else:
                if ch not in node.childs:
                    return False
                return dfs(node.childs[ch], idx+1)
        return dfs(self.root, 0)


class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end = False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 679188260 · Runtime Error · 2025-11-19 14:11:23 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in root:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            if word[idx] == '.':
                for child in node.childs.values():
                    if dfs(child, idx + 1):
                        return True
            else:
                if word[idx] not in node.childs:
                    return False
                return dfs(node.childs[word[idx]], idx + 1)
            
        return dfs(self.root, 0)
        

class TrieNode:
    self.childs: [str, TrieNode] = {}
    self.is_end = False


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 679188610 · Runtime Error · 2025-11-19 14:12:36 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in root:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            if word[idx] == '.':
                for child in node.childs.values():
                    if dfs(child, idx + 1):
                        return True
            else:
                if word[idx] not in node.childs:
                    return False
                return dfs(node.childs[word[idx]], idx + 1)
            
        return dfs(self.root, 0)
        

class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end = False


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 679188822 · Runtime Error · 2025-11-19 14:13:20 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in node:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            if word[idx] == '.':
                for child in node.childs.values():
                    if dfs(child, idx + 1):
                        return True
            else:
                if word[idx] not in node.childs:
                    return False
                return dfs(node.childs[word[idx]], idx + 1)
            
        return dfs(self.root, 0)
        

class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end = False


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 提交 679188922 · Wrong Answer · 2025-11-19 14:13:39 CST · python3

```python
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.childs:
                node.childs[ch] = TrieNode()
        node.is_end = True
        

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            if word[idx] == '.':
                for child in node.childs.values():
                    if dfs(child, idx + 1):
                        return True
            else:
                if word[idx] not in node.childs:
                    return False
                return dfs(node.childs[word[idx]], idx + 1)
            
        return dfs(self.root, 0)
        

class TrieNode:
    def __init__(self):
        self.childs: [str, TrieNode] = {}
        self.is_end = False


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```


## 设计地铁系统 (`design-underground-system`)

- 题目链接：https://leetcode.cn/problems/design-underground-system/
- 难度：Medium
- 标签：设计, 哈希表, 字符串
- 总提交次数：1
- 最近提交时间：2025-11-19 10:42:20 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-19 10:32:36 CST

```markdown
30 秒口述要点
* 这是在线统计的状态建模题。我用两个哈希表：in_map 记录在途乘客 id→(起点, 时间)，stats 记录(起点, 终点)→(总时长, 次数)。checkIn 写 in_map；checkOut 取出配对算时长并累加到 stats，同时删除 in_map。getAverage 直接总时长/次数。全部 O(1)。不存明细，靠充分统计量控制空间，工程上也便于扩展到方差或分位数。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 679141368 | 2025-11-19 10:42:20 CST | python3 | Accepted | 55 ms | 28.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 位1的个数 (`number-of-1-bits`)

- 题目链接：https://leetcode.cn/problems/number-of-1-bits/
- 难度：Easy
- 标签：位运算, 分治
- 总提交次数：1
- 最近提交时间：2025-11-19 09:31:48 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 679119811 | 2025-11-19 09:31:48 CST | python3 | Accepted | 1 ms | 17.4 MB |

### 未通过提交代码
(所有提交均已通过)

## 位 1 的个数 (`er-jin-zhi-zhong-1de-ge-shu-lcof`)

- 题目链接：https://leetcode.cn/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/
- 难度：Easy
- 标签：位运算
- 总提交次数：1
- 最近提交时间：2025-11-19 09:30:14 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 679119507 | 2025-11-19 09:30:14 CST | python3 | Accepted | 50 ms | 17.5 MB |

### 未通过提交代码
(所有提交均已通过)

## O(1) 时间插入、删除和获取随机元素 - 允许重复 (`insert-delete-getrandom-o1-duplicates-allowed`)

- 题目链接：https://leetcode.cn/problems/insert-delete-getrandom-o1-duplicates-allowed/
- 难度：Hard
- 标签：设计, 数组, 哈希表, 数学, 随机化
- 总提交次数：8
- 最近提交时间：2025-11-18 11:11:34 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-18 11:15:07 CST

```markdown
面试口述要点（30 秒）
* 用数组存全部副本，字典映射 值 -> 出现下标集合。
* 插入：push 到数组，集合里加新下标；返回是否首次出现。
* 删除：从集合取一个下标，与数组末尾交换，更新末尾元素的集合，再弹尾；集合空则删键。
* 随机：数组随机下标。所有操作均摊 O(1)，空间 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 678879700 | 2025-11-18 10:51:22 CST | python3 | Wrong Answer | N/A | N/A |
| 678880816 | 2025-11-18 10:54:39 CST | python3 | Wrong Answer | N/A | N/A |
| 678882301 | 2025-11-18 10:59:03 CST | python3 | Accepted | 149 ms | 72.6 MB |
| 678883313 | 2025-11-18 11:02:06 CST | python3 | Runtime Error | N/A | N/A |
| 678883802 | 2025-11-18 11:03:42 CST | python3 | Accepted | 158 ms | 72.6 MB |
| 678884452 | 2025-11-18 11:05:25 CST | python3 | Accepted | 125 ms | 72.5 MB |
| 678885415 | 2025-11-18 11:08:19 CST | python3 | Wrong Answer | N/A | N/A |
| 678886454 | 2025-11-18 11:11:34 CST | python3 | Accepted | 124 ms | 72.6 MB |

### 未通过提交代码
#### 提交 678879700 · Wrong Answer · 2025-11-18 10:51:22 CST · python3

```python
class RandomizedCollection:

    def __init__(self):
        self.vals: list[int] = []
        self.vals_to_idx: dict[int, set] = collections.defaultdict(set)
        

    def insert(self, val: int) -> bool:
        exist = val in self.vals_to_idx
        self.vals.append(val)
        self.vals_to_idx[val].add(len(self.vals) - 1)
        return exist        


    def remove(self, val: int) -> bool:
        if val not in self.vals_to_idx:
            return False
        idx_to_remove = self.vals_to_idx[val].pop()
        last_idx = len(self.vals) - 1
        last_val = self.vals[last_idx]
        if val != last_val:
            self.vals[idx_to_remove] = last_val
            self.vals_to_idx[last_val].add(idx_to_remove)
            self.vals_to_idx[last_val].discard(last_idx)
        self.vals.pop()
        if not self.vals_to_idx[val]:
            del self.vals_to_idx[val]
        return True


    def getRandom(self) -> int:
        return random.choice(self.vals)

        


# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

#### 提交 678880816 · Wrong Answer · 2025-11-18 10:54:39 CST · python3

```python
class RandomizedCollection:

    def __init__(self):
        self.vals: list[int] = []
        self.vals_to_idx: dict[int, set] = collections.defaultdict(set)
        

    def insert(self, val: int) -> bool:
        exist = val in self.vals_to_idx
        self.vals.append(val)
        self.vals_to_idx[val].add(len(self.vals) - 1)
        return exist        


    def remove(self, val: int) -> bool:
        if val not in self.vals_to_idx:
            return False
        idx_to_remove = self.vals_to_idx[val].pop()
        last_idx = len(self.vals) - 1
        last_val = self.vals[last_idx]
        if idx_to_remove != last_idx:
            self.vals[idx_to_remove] = last_val
            self.vals_to_idx[last_val].add(idx_to_remove)
            self.vals_to_idx[last_val].discard(last_idx)
        self.vals.pop()
        if not self.vals_to_idx[val]:
            del self.vals_to_idx[val]
        return True


    def getRandom(self) -> int:
        return random.choice(self.vals)

        


# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

#### 提交 678883313 · Runtime Error · 2025-11-18 11:02:06 CST · python3

```python
class RandomizedCollection:

    def __init__(self):
        self.vals: list[int] = []
        self.vals_to_idx: dict[int, set] = collections.defaultdict(set)
        

    def insert(self, val: int) -> bool:
        exist = val not in self.vals_to_idx
        self.vals.append(val)
        self.vals_to_idx[val].add(len(self.vals) - 1)
        return exist        


    def remove(self, val: int) -> bool:
        if val not in self.vals_to_idx:
            return False
        idx_to_remove = self.vals_to_idx[val].pop()
        last_idx = len(self.vals) - 1
        last_val = self.vals[last_idx]
        if last_val != val:
        # if idx_to_remove != last_idx:
            self.vals[idx_to_remove] = last_val
            self.vals_to_idx[last_val].discard(last_idx)
            self.vals_to_idx[last_val].add(idx_to_remove)
        self.vals.pop()
        if not self.vals_to_idx[val]:
            del self.vals_to_idx[val]
        return True


    def getRandom(self) -> int:
        return random.choice(self.vals)

        


# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

#### 提交 678885415 · Wrong Answer · 2025-11-18 11:08:19 CST · python3

```python
class RandomizedCollection:

    def __init__(self):
        self.vals: list[int] = []
        self.vals_to_idx: dict[int, set] = collections.defaultdict(set)
        

    def insert(self, val: int) -> bool:
        exist = val in self.vals_to_idx
        self.vals.append(val)
        self.vals_to_idx[val].add(len(self.vals) - 1)
        return exist        


    def remove(self, val: int) -> bool:
        if val not in self.vals_to_idx:
            return False
        idx_to_remove = self.vals_to_idx[val].pop()
        last_idx = len(self.vals) - 1
        last_val = self.vals[last_idx]
        if idx_to_remove != last_idx:
            self.vals[idx_to_remove] = last_val
            self.vals_to_idx[last_val].add(idx_to_remove)
            self.vals_to_idx[last_val].discard(last_idx)
        self.vals.pop()
        if not self.vals_to_idx[val]:
            del self.vals_to_idx[val]
        return True


    def getRandom(self) -> int:
        return random.choice(self.vals)

        


# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```


## O(1) 时间插入、删除和获取随机元素 (`insert-delete-getrandom-o1`)

- 题目链接：https://leetcode.cn/problems/insert-delete-getrandom-o1/
- 难度：Medium
- 标签：设计, 数组, 哈希表, 数学, 随机化
- 总提交次数：1
- 最近提交时间：2025-11-18 10:07:26 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-18 10:09:11 CST

```markdown
面试口述要点（30 秒内）
* 用数组存值、字典存值到下标的映射。
* 插入：不存在就 append，并记录下标到字典。
* 删除：定位下标，把它和数组最后一个元素交换，更新最后一个元素的新下标，然后 pop 尾并从字典删键。
* 随机：从数组随机取一个下标返回。
所有操作均摊 O(1)，空间 O(n)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 678866242 | 2025-11-18 10:07:26 CST | python3 | Accepted | 153 ms | 56.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 按序打印 (`print-in-order`)

- 题目链接：https://leetcode.cn/problems/print-in-order/
- 难度：Easy
- 标签：多线程
- 总提交次数：4
- 最近提交时间：2025-11-17 17:10:12 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-16 16:46:25 CST

```markdown
面试可直接口述（30 秒内）
* 这是并发有序问题，本质是 second 依赖 first，third 依赖 second。
* 用两个 Event 当闸门：first 打印后 set 第一个事件，second 等第一个事件后打印并 set 第二个事件，third 等第二个事件再打印。
* 这样每步只依赖前一步，无忙等，无睡眠，时间和空间都是 O(1)。
* 在 Python 里 Event 最贴合“等待某事发生”的语义，代码最简洁。

除了使用 Event，这道题也可以用 Semaphore 来解决。Event 像一个状态开关，非常适合这种简单的‘已完成’信号通知。而 Semaphore 本质是一个计数器，用初始值为0的 Semaphore，通过 release 和 acquire 操作，可以实现完全相同的‘一对一’通知效果。

之所以也考虑 Semaphore，是因为它是一个更通用的工具。比如，如果需求变成控制对某个资源的并发访问数量，Semaphore 就能直接通过设定初始计数值来解决，而 Event 则不适用。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 678466871 | 2025-11-16 16:26:23 CST | python3 | Runtime Error | N/A | N/A |
| 678467020 | 2025-11-16 16:26:53 CST | python3 | Accepted | 52 ms | 17.9 MB |
| 678719832 | 2025-11-17 17:08:53 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 678720642 | 2025-11-17 17:10:12 CST | python3 | Accepted | 55 ms | 18 MB |

### 未通过提交代码
#### 提交 678466871 · Runtime Error · 2025-11-16 16:26:23 CST · python3

```python
class Foo:
    def __init__(self):
        self.gate1 = threading.Semaphore(0)
        self.gate2 = threading.Semaphore(0)


    def first(self, printFirst: 'Callable[[], None]') -> None:
        
        # printFirst() outputs "first". Do not change or remove this line.
        printFirst()
        self.gate1.release()


    def second(self, printSecond: 'Callable[[], None]') -> None:
        
        # printSecond() outputs "second". Do not change or remove this line.
        self.gate1.require()
        printSecond()
        self.gate2.release()


    def third(self, printThird: 'Callable[[], None]') -> None:
        
        # printThird() outputs "third". Do not change or remove this line.
        self.gate2.require()
        printThird()
```

#### 提交 678719832 · Time Limit Exceeded · 2025-11-17 17:08:53 CST · python3

```python
class Foo:
    def __init__(self):
        self.second_sem = threading.Semaphore(0)
        self.third_sem = threading.Semaphore(0)


    def first(self, printFirst: 'Callable[[], None]') -> None:
        
        # printFirst() outputs "first". Do not change or remove this line.
        printFirst()
        self.second_sem.release()


    def second(self, printSecond: 'Callable[[], None]') -> None:
        self.second.acquire()
        
        # printSecond() outputs "second". Do not change or remove this line.
        printSecond()
        self.third_sem.release()


    def third(self, printThird: 'Callable[[], None]') -> None:
        self.third_sem.acquire()
        
        # printThird() outputs "third". Do not change or remove this line.
        printThird()
```


## H2O 生成 (`building-h2o`)

- 题目链接：https://leetcode.cn/problems/building-h2o/
- 难度：Medium
- 标签：多线程
- 总提交次数：4
- 最近提交时间：2025-11-17 17:05:49 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-17 10:15:37 CST

```markdown
之所以让 Oxygen 线程去释放 Hydrogen 的信号量，是为了保证‘成组’的原子性。

Oxygen 线程在这里扮演了‘周期控制器’的角色。它以接收到两个 H 的信号作为‘组装完成’的标志，在自己打印 O 之后，才释放两个 H 的名额。这确保了上一组的 H₂O 完全生成后，下一组的 H 才能开始，从而完美地实现了线程间的同步与分组。

面试口述要点（30秒内）
* 用两个信号量：h_sem=2 限制每轮最多两个 H；o_sem=0 统计 H 的就绪数。
* H：先拿 h_sem、打印 H、再 o_sem.release() 通知 O 有一个 H 就绪。
* O：等待 o_sem 两次后打印 O，最后释放两次 h_sem 开启下一轮。
* 这样每轮恰好 2H+1O，且不会跨轮混输出；实现简单、无忙等、无死锁。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 678602532 | 2025-11-17 10:18:25 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 678606634 | 2025-11-17 10:32:52 CST | python3 | Accepted | 57 ms | 17.9 MB |
| 678609794 | 2025-11-17 10:43:00 CST | python3 | Accepted | 54 ms | 18.1 MB |
| 678718743 | 2025-11-17 17:05:49 CST | python3 | Accepted | 49 ms | 17.9 MB |

### 未通过提交代码
#### 提交 678602532 · Time Limit Exceeded · 2025-11-17 10:18:25 CST · python3

```python
class H2O:
    def __init__(self):
        self.h_sema = threading.Semaphore(2)
        self.o_sema = threading.Semaphore(0)


    def hydrogen(self, releaseHydrogen: 'Callable[[], None]') -> None:

        self.h_sema.acquire()        
        # releaseHydrogen() outputs "H". Do not change or remove this line.
        releaseHydrogen()
        self.o_sema.release()


    def oxygen(self, releaseOxygen: 'Callable[[], None]') -> None:
        self.o_sema.acquire()
        self.o_sema.acquire()
        
        # releaseOxygen() outputs "O". Do not change or remove this line.
        releaseOxygen()

        self.h_sema.release()
        self.h_sema.release()
```


## 数据流的中位数 (`find-median-from-data-stream`)

- 题目链接：https://leetcode.cn/problems/find-median-from-data-stream/
- 难度：Hard
- 标签：设计, 双指针, 数据流, 排序, 堆（优先队列）
- 总提交次数：6
- 最近提交时间：2025-11-16 15:18:37 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 678277102 | 2025-11-15 16:05:59 CST | python3 | Runtime Error | N/A | N/A |
| 678277170 | 2025-11-15 16:06:18 CST | python3 | Runtime Error | N/A | N/A |
| 678277257 | 2025-11-15 16:06:39 CST | python3 | Runtime Error | N/A | N/A |
| 678277990 | 2025-11-15 16:09:50 CST | python3 | Accepted | 195 ms | 40.1 MB |
| 678449167 | 2025-11-16 15:16:40 CST | python3 | Wrong Answer | N/A | N/A |
| 678449721 | 2025-11-16 15:18:37 CST | python3 | Accepted | 225 ms | 40.6 MB |

### 未通过提交代码
#### 提交 678277102 · Runtime Error · 2025-11-15 16:05:59 CST · python3

```python
class MedianFinder:

    def __init__(self):
        self.small = []
        self.large = []
        
    def addNum(self, num: int) -> None:
        # 先将 num 放入大顶堆 small
        heapq.heappush(self.small, -num)
        # 再将大顶堆中最大的数放入小顶堆
        heapq.heappush(self.large, -heapq.heapppop(self.small))
        # 如果小顶堆中的元素数量 大于 大顶堆中的数量，则弹出一个最小的元素放入大顶堆
        if len(small) > len(large):
            heapq.heappush(self.small, -heapq.heapppop(self.large))

    def findMedian(self) -> float:
        # 经过addNum 之后，两个堆的元素数量要么相等，要么大顶堆比小顶堆多 1
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

#### 提交 678277170 · Runtime Error · 2025-11-15 16:06:18 CST · python3

```python
class MedianFinder:

    def __init__(self):
        self.small = []
        self.large = []
        
    def addNum(self, num: int) -> None:
        # 先将 num 放入大顶堆 small
        heapq.heappush(self.small, -num)
        # 再将大顶堆中最大的数放入小顶堆
        heapq.heappush(self.large, -heapq.heappop(self.small))
        # 如果小顶堆中的元素数量 大于 大顶堆中的数量，则弹出一个最小的元素放入大顶堆
        if len(small) > len(large):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        # 经过addNum 之后，两个堆的元素数量要么相等，要么大顶堆比小顶堆多 1
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

#### 提交 678277257 · Runtime Error · 2025-11-15 16:06:39 CST · python3

```python
class MedianFinder:

    def __init__(self):
        self.small = []
        self.large = []
        
    def addNum(self, num: int) -> None:
        # 先将 num 放入大顶堆 small
        heapq.heappush(self.small, -num)
        # 再将大顶堆中最大的数放入小顶堆
        heapq.heappush(self.large, -heapq.heappop(self.small))
        # 如果小顶堆中的元素数量 大于 大顶堆中的数量，则弹出一个最小的元素放入大顶堆
        if len(self.small) > len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        # 经过addNum 之后，两个堆的元素数量要么相等，要么大顶堆比小顶堆多 1
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

#### 提交 678449167 · Wrong Answer · 2025-11-16 15:16:40 CST · python3

```python
class MedianFinder:

    def __init__(self):
        self.small = []
        self.large = []
        

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
        

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return self.small[0]
        return (self.small[0] + self.large[0]) / 2
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```


## 每日温度 (`daily-temperatures`)

- 题目链接：https://leetcode.cn/problems/daily-temperatures/
- 难度：Medium
- 标签：栈, 数组, 单调栈
- 总提交次数：4
- 最近提交时间：2025-11-15 11:31:56 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-14 11:09:19 CST

```markdown
本质上，单调栈就是用空间换时间，把多次重复的向后查找，变成了一次入栈和一次出栈的摊销操作。

本质是 Next Greater Element。答案由右侧更大值的出现触发，用“存索引的递减栈”压缩所有还没等到更大值的元素。每来一个新值，把栈顶那些更小的连续元素一口气弹出并结算距离；没被弹出的要么更高要么相等，当前不可能是它们答案。每个元素最多入栈出栈一次，O(n)。相比堆/排序，单调栈同时保证“值比较”和“最近性”，是这类题的最优结构匹配。

单调栈同时保留了“相对位置”和“值的单调性”，恰好兼顾“更大”和“最近”。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 678036389 | 2025-11-14 11:02:50 CST | python3 | Wrong Answer | N/A | N/A |
| 678036799 | 2025-11-14 11:04:23 CST | python3 | Accepted | 91 ms | 26.4 MB |
| 678225268 | 2025-11-15 11:31:43 CST | python3 | Runtime Error | N/A | N/A |
| 678225302 | 2025-11-15 11:31:56 CST | python3 | Accepted | 95 ms | 26 MB |

### 未通过提交代码
#### 提交 678036389 · Wrong Answer · 2025-11-14 11:02:50 CST · python3

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        # 初始化，默认没有更高温度
        res = [0] * n
        # 使用单调递减栈
        stack = []

        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:
                idx = stack.pop()
                res.append(temperatures[idx])
            stack.append(i)
        
        return res
```

#### 提交 678225268 · Runtime Error · 2025-11-15 11:31:43 CST · python3

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        res = [0] * temperatures
        stack = []

        for i, num in enumerate(temperatures):
            while stack and num > temperatures[stack[-1]]:
                res[stack[-1]] = i-stack[-1]
                stack.pop()
            stack.append(i)
        
        return res
```


## LFU 缓存 (`lfu-cache`)

- 题目链接：https://leetcode.cn/problems/lfu-cache/
- 难度：Hard
- 标签：设计, 哈希表, 链表, 双向链表
- 总提交次数：7
- 最近提交时间：2025-11-15 11:20:49 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-14 14:04:56 CST

```markdown
我用双哈希 + LRU 桶。key 映射到 value 和 freq；freq 映射到一个 OrderedDict，维护该频次下的 LRU 顺序。get 时把 key 从旧频次桶删掉，freq+1 后插入新桶尾；旧桶空且是最小频次则 min_freq++。put 时满了就到 min_freq 的桶里按 LRU 淘汰最旧的 key；新 key 放入 freq=1，并把 min_freq 设为 1。所有操作都是 O(1)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 678076125 | 2025-11-14 14:32:42 CST | python3 | Accepted | 140 ms | 78.4 MB |
| 678221867 | 2025-11-15 11:12:49 CST | python3 | Runtime Error | N/A | N/A |
| 678221954 | 2025-11-15 11:13:15 CST | python3 | Runtime Error | N/A | N/A |
| 678222358 | 2025-11-15 11:15:32 CST | python3 | Runtime Error | N/A | N/A |
| 678222542 | 2025-11-15 11:16:32 CST | python3 | Wrong Answer | N/A | N/A |
| 678223223 | 2025-11-15 11:20:02 CST | python3 | Wrong Answer | N/A | N/A |
| 678223370 | 2025-11-15 11:20:49 CST | python3 | Accepted | 151 ms | 78.6 MB |

### 未通过提交代码
#### 提交 678221867 · Runtime Error · 2025-11-15 11:12:49 CST · python3

```python
class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key2value = {}
        self.key2freq = {}
        self.freq2keys = defaultdict(OrderedDict)
        self.min_freq = 0
        

    def get(self, key: int) -> int:
        if key not in self.key2value:
            return -1
        self.increase_freq(key)
        return self.key2value[key]
        

    def put(self, key: int, value: int) -> None:
        if key in self.key2value:
            self.increase_freq(key)
            return
        if len(key2value) >= self.capacity:
            self.evict_one()
        self.key2value[key] = value
        self.key2freq[key] = 1
        self.freq2keys[1][key] = None
        self.min_freq = 1


    def evict_one(self):
        key_to_del = self.freq2keys[self.min_freq].popitem(last=False)
        del self.key2value[key_to_del]
        del self.key2freq[key_to_del]
    

    def increase_freq(self, key):
        freq = self.key2freq.get(key)
        del self.freq2keys[freq][key]
        if not self.freq2keys[freq]:
            if freq == self.min_freq:
                self.min_freq += 1
        new_freq = freq+1
        self.key2freq[key] = new_freq
        self.freq2keys[new_freq].append(key)        


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 提交 678221954 · Runtime Error · 2025-11-15 11:13:15 CST · python3

```python
class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key2value = {}
        self.key2freq = {}
        self.freq2keys = defaultdict(OrderedDict)
        self.min_freq = 0
        

    def get(self, key: int) -> int:
        if key not in self.key2value:
            return -1
        self.increase_freq(key)
        return self.key2value[key]
        

    def put(self, key: int, value: int) -> None:
        if key in self.key2value:
            self.increase_freq(key)
            return
        if len(self.key2value) >= self.capacity:
            self.evict_one()
        self.key2value[key] = value
        self.key2freq[key] = 1
        self.freq2keys[1][key] = None
        self.min_freq = 1


    def evict_one(self):
        key_to_del = self.freq2keys[self.min_freq].popitem(last=False)
        del self.key2value[key_to_del]
        del self.key2freq[key_to_del]
    

    def increase_freq(self, key):
        freq = self.key2freq.get(key)
        del self.freq2keys[freq][key]
        if not self.freq2keys[freq]:
            if freq == self.min_freq:
                self.min_freq += 1
        new_freq = freq+1
        self.key2freq[key] = new_freq
        self.freq2keys[new_freq].append(key)        


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 提交 678222358 · Runtime Error · 2025-11-15 11:15:32 CST · python3

```python
class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key2value = {}
        self.key2freq = {}
        self.freq2keys = defaultdict(OrderedDict)
        self.min_freq = 0
        

    def get(self, key: int) -> int:
        if key not in self.key2value:
            return -1
        self.increase_freq(key)
        return self.key2value[key]
        

    def put(self, key: int, value: int) -> None:
        if key in self.key2value:
            self.increase_freq(key)
            return
        if len(self.key2value) >= self.capacity:
            self.evict_one()
        self.key2value[key] = value
        self.key2freq[key] = 1
        self.freq2keys[1][key] = None
        self.min_freq = 1


    def evict_one(self):
        key_to_del = self.freq2keys[self.min_freq].popitem(last=False)
        del self.key2value[key_to_del]
        del self.key2freq[key_to_del]
    

    def increase_freq(self, key):
        freq = self.key2freq.get(key)
        del self.freq2keys[freq][key]
        if not self.freq2keys[freq]:
            if freq == self.min_freq:
                self.min_freq += 1
        new_freq = freq+1
        self.key2freq[key] = new_freq
        self.freq2keys[new_freq][key] = None        


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 提交 678222542 · Wrong Answer · 2025-11-15 11:16:32 CST · python3

```python
class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key2value = {}
        self.key2freq = {}
        self.freq2keys = defaultdict(OrderedDict)
        self.min_freq = 0
        

    def get(self, key: int) -> int:
        if key not in self.key2value:
            return -1
        self.increase_freq(key)
        return self.key2value[key]
        

    def put(self, key: int, value: int) -> None:
        if key in self.key2value:
            self.increase_freq(key)
            return
        if len(self.key2value) >= self.capacity:
            self.evict_one()
        self.key2value[key] = value
        self.key2freq[key] = 1
        self.freq2keys[1][key] = None
        self.min_freq = 1


    def evict_one(self):
        key_to_del, _ = self.freq2keys[self.min_freq].popitem(last=False)
        del self.key2value[key_to_del]
        del self.key2freq[key_to_del]
    

    def increase_freq(self, key):
        freq = self.key2freq.get(key)
        del self.freq2keys[freq][key]
        if not self.freq2keys[freq]:
            if freq == self.min_freq:
                self.min_freq += 1
        new_freq = freq+1
        self.key2freq[key] = new_freq
        self.freq2keys[new_freq][key] = None        


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### 提交 678223223 · Wrong Answer · 2025-11-15 11:20:02 CST · python3

```python
class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key2value = {}
        self.key2freq = {}
        self.freq2keys = defaultdict(OrderedDict)
        self.min_freq = 0
        

    def get(self, key: int) -> int:
        if key not in self.key2value:
            return -1
        self.increase_freq(key)
        return self.key2value[key]
        

    def put(self, key: int, value: int) -> None:
        if key in self.key2value:
            self.increase_freq(key)
            return
        if len(self.key2value) >= self.capacity:
            self.evict_one()
        self.key2value[key] = value
        self.key2freq[key] = 1
        self.freq2keys[1][key] = None
        self.min_freq = 1


    def evict_one(self):
        key_to_del, _ = self.freq2keys[self.min_freq].popitem(last=False)
        del self.key2value[key_to_del]
        del self.key2freq[key_to_del]
    

    def increase_freq(self, key):
        freq = self.key2freq.get(key)
        del self.freq2keys[freq][key]
        if not self.freq2keys[freq]:
            if freq == self.min_freq:
                self.min_freq = freq+1
        new_freq = freq+1
        self.key2freq[key] = new_freq
        self.freq2keys[new_freq][key] = None        


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```


## 插入区间 (`insert-interval`)

- 题目链接：https://leetcode.cn/problems/insert-interval/
- 难度：Medium
- 标签：数组
- 总提交次数：3
- 最近提交时间：2025-11-13 08:05:58 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-13 08:09:40 CST

```markdown
分为 3 部分：完全在左边的、完全在右边的，有重叠的

重叠的部分是将重叠的区间不断的合并扩充到新的区间

重叠部分的 append 是在 while 循环之外
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 677764025 | 2025-11-13 08:04:08 CST | python3 | Runtime Error | N/A | N/A |
| 677764041 | 2025-11-13 08:04:25 CST | python3 | Wrong Answer | N/A | N/A |
| 677764095 | 2025-11-13 08:05:58 CST | python3 | Accepted | 0 ms | 19.4 MB |

### 未通过提交代码
#### 提交 677764025 · Runtime Error · 2025-11-13 08:04:08 CST · python3

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        n = len(intervals)
        new_start, new_end = newInterval
        i = 0

        # 完全在左边的
        while i < n and intervals[i][1] < new_start:
            res.append(intervals[i])
            i += 1

        # 有重叠的
        while i < n and intervals[i][0] <= new_end:
            new_start = min(intervals[i][0], new_start)
            new_end = max(intervals[i][1], new_end)
            res.append([new_start, new_end])
            i += 1

        # 完全在右边的
        while i < n:
            ress.append(intervals[i])
            i += 1

        return res
```

#### 提交 677764041 · Wrong Answer · 2025-11-13 08:04:25 CST · python3

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        n = len(intervals)
        new_start, new_end = newInterval
        i = 0

        # 完全在左边的
        while i < n and intervals[i][1] < new_start:
            res.append(intervals[i])
            i += 1

        # 有重叠的
        while i < n and intervals[i][0] <= new_end:
            new_start = min(intervals[i][0], new_start)
            new_end = max(intervals[i][1], new_end)
            res.append([new_start, new_end])
            i += 1

        # 完全在右边的
        while i < n:
            res.append(intervals[i])
            i += 1

        return res
```


## 最小覆盖子串 (`minimum-window-substring`)

- 题目链接：https://leetcode.cn/problems/minimum-window-substring/
- 难度：Hard
- 标签：哈希表, 字符串, 滑动窗口
- 总提交次数：19
- 最近提交时间：2025-11-12 14:16:39 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-04 09:31:54 CST

```markdown
这是“连续子串 + 频次下限”的典型滑动窗口。可行性对右端是单调的，右扩只会更容易满足；对左端是反向单调的，左移只会更难满足。
因此用双指针：右指针扩张直到覆盖 t；一旦覆盖，左指针极限收缩得到该右端下的最短窗口并更新答案，然后让窗口失效继续右移。
计数用 need/window，加上 missing 或 valid 做 O(1) 判定。每个指针只前进不回退，时间 O(n)，空间 O(字母种类)。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 650277735 | 2025-08-06 08:20:47 CST | python3 | Wrong Answer | N/A | N/A |
| 650289033 | 2025-08-06 09:35:27 CST | python3 | Accepted | 63 ms | 17.9 MB |
| 650614334 | 2025-08-07 07:33:30 CST | python3 | Accepted | 71 ms | 18.1 MB |
| 658730052 | 2025-09-02 14:05:10 CST | python3 | Runtime Error | N/A | N/A |
| 658730208 | 2025-09-02 14:05:42 CST | python3 | Runtime Error | N/A | N/A |
| 658730404 | 2025-09-02 14:06:17 CST | python3 | Wrong Answer | N/A | N/A |
| 658731149 | 2025-09-02 14:08:34 CST | python3 | Wrong Answer | N/A | N/A |
| 658734227 | 2025-09-02 14:17:29 CST | python3 | Runtime Error | N/A | N/A |
| 658735712 | 2025-09-02 14:21:28 CST | python3 | Wrong Answer | N/A | N/A |
| 658740985 | 2025-09-02 14:35:06 CST | python3 | Accepted | 111 ms | 18.1 MB |
| 660601783 | 2025-09-08 13:59:57 CST | python3 | Wrong Answer | N/A | N/A |
| 660610029 | 2025-09-08 14:25:10 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 660611366 | 2025-09-08 14:28:30 CST | python3 | Wrong Answer | N/A | N/A |
| 660613691 | 2025-09-08 14:34:27 CST | python3 | Accepted | 90 ms | 18.1 MB |
| 662841333 | 2025-09-15 12:36:24 CST | python3 | Accepted | 69 ms | 17.8 MB |
| 675679440 | 2025-11-04 09:26:49 CST | python3 | Accepted | 63 ms | 18.2 MB |
| 677587958 | 2025-11-12 14:12:09 CST | python3 | Wrong Answer | N/A | N/A |
| 677588261 | 2025-11-12 14:13:26 CST | python3 | Wrong Answer | N/A | N/A |
| 677589092 | 2025-11-12 14:16:39 CST | python3 | Accepted | 94 ms | 17.9 MB |

### 未通过提交代码
#### 提交 650277735 · Wrong Answer · 2025-08-06 08:20:47 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(s) < len(t):
            return ''
        need = collections.Counter(t)
        window = collections.defaultdict(int)
        valid = 0
        left = right = 0
        
        # 记录最小覆盖子串的起始索引和长度
        start = 0
        min_len = float('inf')

        # 不断向右扩张
        for right, ch in enumerate(s):
            # 只有在ch属于need时才更新window
            if ch in need:
                window[ch] += 1
                if window[ch] == need[ch]:
                    valid += 1
            
            while valid == len(need):  # 找到一个可行解，尝试优化
                # 更新最短字符
                window_len = right - left + 1
                if window_len < min_len:
                    min_len = window_len
                    start = left
                
                # 准备移出左边字符
                left_char = s[left]
                if left_char in need:
                    if window[left_char] == need[ch]:
                        valid -= 1
                    window[left_char] -= 1
                left += 1

        return '' if min_len == float('inf') else s[start : start + min_len]
```

#### 提交 658730052 · Runtime Error · 2025-09-02 14:05:10 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(s)
        window = collections.defaultdict(int)
        valid = 0
        start = 0
        min_len = 0
        left = 0

        for right, r_char in enumerate(s):
            window[r_char] += 1
            if window[r_char] == need[r_char]:
                valid += 1
            while valid == len(need):
                l_char = s[left]
                min_len = min(right - left + 1, left)
                start = left
                if window[l_char] != 0:
                    window[l_char] -= 1
                    if window[l_char] != need[l_char]:
                        valid -= 1
                    left += 1
        return s[start, start + min_len]
```

#### 提交 658730208 · Runtime Error · 2025-09-02 14:05:42 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(s)
        window = collections.defaultdict(int)
        valid = 0
        start = 0
        min_len = 0
        left = 0

        for right, r_char in enumerate(s):
            window[r_char] += 1
            if window[r_char] == need[r_char]:
                valid += 1
            while valid == len(need):
                l_char = s[left]
                min_len = min(right - left + 1, min_len)
                start = left
                if window[l_char] != 0:
                    window[l_char] -= 1
                    if window[l_char] != need[l_char]:
                        valid -= 1
                    left += 1
        return s[start, start + min_len]
```

#### 提交 658730404 · Wrong Answer · 2025-09-02 14:06:17 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(s)
        window = collections.defaultdict(int)
        valid = 0
        start = 0
        min_len = 0
        left = 0

        for right, r_char in enumerate(s):
            window[r_char] += 1
            if window[r_char] == need[r_char]:
                valid += 1
            while valid == len(need):
                l_char = s[left]
                min_len = min(right - left + 1, min_len)
                start = left
                if window[l_char] != 0:
                    window[l_char] -= 1
                    if window[l_char] != need[l_char]:
                        valid -= 1
                    left += 1
        return s[start : start + min_len]
```

#### 提交 658731149 · Wrong Answer · 2025-09-02 14:08:34 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(s)
        window = collections.defaultdict(int)
        valid = 0
        start = 0
        min_len = 0
        left = 0

        for right, r_char in enumerate(s):
            window[r_char] += 1
            if window[r_char] == need[r_char]:
                valid += 1
            while valid == len(need):
                l_char = s[left]
                min_len = min(right - left + 1, min_len)
                start = left
                if window[l_char] != 0:
                    if window[l_char] != need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                    left += 1
        return s[start : start + min_len]
```

#### 提交 658734227 · Runtime Error · 2025-09-02 14:17:29 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        window = collections.defaultdict(int)
        valid = 0
        start = 0
        min_len = float('inf')
        left = 0

        for right, r_char in enumerate(s):
            window[r_char] += 1
            if window[r_char] == need[r_char]:
                valid += 1
            while valid == len(need):
                l_char = s[left]
                min_len = min(right - left + 1, min_len)
                start = left
                if window[l_char] != 0:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                    left += 1
        return s[start : start + min_len]
```

#### 提交 658735712 · Wrong Answer · 2025-09-02 14:21:28 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        window = collections.defaultdict(int)
        valid = 0
        start = 0
        min_len = float('inf')
        left = 0

        for right, r_char in enumerate(s):
            window[r_char] += 1
            if r_char in need and window[r_char] == need[r_char]:
                valid += 1
            while valid == len(need):
                l_char = s[left]
                min_len = min(right - left + 1, min_len)
                start = left
                if window[l_char] != 0:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                    left += 1
        return s[start : start + min_len] if min_len != float('inf') else ''
```

#### 提交 660601783 · Wrong Answer · 2025-09-08 13:59:57 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        window = collections.defaultdict(int)
        start = valid = 0
        min_len = float('inf')
        left = 0

        for right, r_char in enumerate(s):
            window[r_char] += 1
            if window[r_char] == need[r_char]:
                valid += 1
            while valid == len(need):
                start = left
                min_len = min(min_len, right - left + 1)
                l_char = s[left]
                if window[l_char] == need[l_char]:
                    valid -= 1
                window[l_char] -= 1
                left += 1
        
        return s[start:start + min_len] if min_len != float('inf') else ''
```

#### 提交 660610029 · Time Limit Exceeded · 2025-09-08 14:25:10 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        window = collections.defaultdict(int)
        start = valid = 0
        min_len = float('inf')
        left = 0

        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            while valid == len(need):
                start = left
                min_len = min(min_len, right - left + 1)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                    left += 1
        
        return s[start:start + min_len] if min_len != float('inf') else ''
```

#### 提交 660611366 · Wrong Answer · 2025-09-08 14:28:30 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        window = collections.defaultdict(int)
        start = valid = 0
        min_len = float('inf')
        left = 0

        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            while valid == len(need):
                start = left
                min_len = min(min_len, right - left + 1)
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        
        return s[start:start + min_len] if min_len != float('inf') else ''
```

#### 提交 677587958 · Wrong Answer · 2025-11-12 14:12:09 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        window = collections.defaultdict(int)
        left = 0
        start = 0
        min_len = float('inf')
        valid = 0
        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            while valid == len(need):
                current_len = right - left + 1
                min_len = min(min_len, current_len)
                start = left
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
        return s[start:start+min_len] if min_len != float('inf') else ''
```

#### 提交 677588261 · Wrong Answer · 2025-11-12 14:13:26 CST · python3

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        window = collections.defaultdict(int)
        left = 0
        start = 0
        min_len = float('inf')
        valid = 0
        for right, r_char in enumerate(s):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            while valid == len(need):
                current_len = right - left + 1
                min_len = min(min_len, current_len)
                start = left
                l_char = s[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        return s[start:start+min_len] if min_len != float('inf') else ''
```


## 基于时间的键值存储 (`time-based-key-value-store`)

- 题目链接：https://leetcode.cn/problems/time-based-key-value-store/
- 难度：Medium
- 标签：设计, 哈希表, 字符串, 二分查找
- 总提交次数：8
- 最近提交时间：2025-11-12 13:47:26 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-04 10:19:51 CST

```markdown
30 秒面试口述要点
* 把问题抽象为“每个 key 的按时间递增版本链 + 前驱查询”。
* 用字典把 key 分桶；每个 key 维护两个并行数组：timestamps 和 values。
* set 利用时间戳严格递增，直接 append，摊还 O(1)；get 用 bisect_right 找 ≤ t 的右边界，O(log n)。
* 并行数组让二分只碰时间戳，缓存友好；相对树结构，写更快、实现更简单；没有删除时是更优解。

一句话总结
* 这套组合精准利用了“单调追加、无删除”的题目特性：字典做分桶，数组做有序日志，二分做前驱查询，达到了实现简单、写入 O(1)、查询 O(log n)、常数低的综合最优。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 675319745 | 2025-11-02 17:35:53 CST | python3 | Runtime Error | N/A | N/A |
| 675319848 | 2025-11-02 17:36:22 CST | python3 | Accepted | 71 ms | 70.2 MB |
| 675690530 | 2025-11-04 10:15:09 CST | python3 | Runtime Error | N/A | N/A |
| 675690658 | 2025-11-04 10:15:37 CST | python3 | Runtime Error | N/A | N/A |
| 675690741 | 2025-11-04 10:15:55 CST | python3 | Wrong Answer | N/A | N/A |
| 675691048 | 2025-11-04 10:16:58 CST | python3 | Accepted | 58 ms | 69.3 MB |
| 677582815 | 2025-11-12 13:47:11 CST | python3 | Runtime Error | N/A | N/A |
| 677582859 | 2025-11-12 13:47:26 CST | python3 | Accepted | 60 ms | 70.1 MB |

### 未通过提交代码
#### 提交 675319745 · Runtime Error · 2025-11-02 17:35:53 CST · python3

```python
class TimeMap:

    def __init__(self):
        self._ts: [str, list[int]] = defaultdict(list)
        self._vals: [str, list[str]] = defaultdict(list)
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        self._ts[key].append(timestamp)
        self._vals[key].append(value)
        

    def get(self, key: str, timestamp: int) -> str:
        ts_list = self._ts.get(key)
        if not ts_list:
            return ''
        idx = bisect.bisect_right(ts_list) - 1
        if idx >= 0:
            return self._vals[key][idx]
        return ''


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)
```

#### 提交 675690530 · Runtime Error · 2025-11-04 10:15:09 CST · python3

```python
class TimeMap:

    def __init__(self):
        self._ts = collections.defaultdict(list)
        self._vals = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self._ts.append(timestamp)
        self._vals.append(value)

    def get(self, key: str, timestamp: int) -> str:
        time_list = self._ts.get(key)
        if not time_list:
            return ''
        idx = bisect.bisect_right(time_list) - 1
        return self._vals[key][idx]


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)
```

#### 提交 675690658 · Runtime Error · 2025-11-04 10:15:37 CST · python3

```python
class TimeMap:

    def __init__(self):
        self._ts = collections.defaultdict(list)
        self._vals = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self._ts[key].append(timestamp)
        self._vals[key].append(value)

    def get(self, key: str, timestamp: int) -> str:
        time_list = self._ts.get(key)
        if not time_list:
            return ''
        idx = bisect.bisect_right(time_list) - 1
        return self._vals[key][idx]


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)
```

#### 提交 675690741 · Wrong Answer · 2025-11-04 10:15:55 CST · python3

```python
class TimeMap:

    def __init__(self):
        self._ts = collections.defaultdict(list)
        self._vals = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self._ts[key].append(timestamp)
        self._vals[key].append(value)

    def get(self, key: str, timestamp: int) -> str:
        time_list = self._ts.get(key)
        if not time_list:
            return ''
        idx = bisect.bisect_right(time_list, timestamp) - 1
        return self._vals[key][idx]


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)
```

#### 提交 677582815 · Runtime Error · 2025-11-12 13:47:11 CST · python3

```python
class TimeMap:

    def __init__(self):
        self.ts = collections.defauldict(list)
        self.vals = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.ts[key].append(timestamp)
        self.vals[key].append(value)
        
    def get(self, key: str, timestamp: int) -> str:
        if key not in self.ts:
            return ''
        idx = bisect.bisect_right(self.ts[key], timestamp) - 1
        if idx >= 0:
            return self.vals[key][idx]
        return ''

# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)
```


## 合并区间 (`merge-intervals`)

- 题目链接：https://leetcode.cn/problems/merge-intervals/
- 难度：Medium
- 标签：数组, 排序
- 总提交次数：3
- 最近提交时间：2025-11-12 09:46:27 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 677515737 | 2025-11-12 08:11:58 CST | python3 | Accepted | 11 ms | 21.4 MB |
| 677527309 | 2025-11-12 09:45:32 CST | python3 | Runtime Error | N/A | N/A |
| 677527497 | 2025-11-12 09:46:27 CST | python3 | Accepted | 8 ms | 21.3 MB |

### 未通过提交代码
#### 提交 677527309 · Runtime Error · 2025-11-12 09:45:32 CST · python3

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals_sorted = sorted(intervals, key=lamda x: x[0])
        merged = []
        for start, end in intervals_sorted:
            # merged 为空和无重叠的情况
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            # 有重叠的情况
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return merged
```


## 爱吃香蕉的珂珂 (`koko-eating-bananas`)

- 题目链接：https://leetcode.cn/problems/koko-eating-bananas/
- 难度：Medium
- 标签：数组, 二分查找
- 总提交次数：3
- 最近提交时间：2025-11-10 14:18:31 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-09 17:07:24 CST

```markdown
本质抽象
* 把“求最小/最大某值”的优化问题，改写成“给定一个候选值，是否可行”的判定问题。
* 若存在一个阈值 T，使得小于 T 均不可行、大于等于 T 均可行（或反之），则可用二分在这个阈值处“切开”答案空间。

面试口述版（30 秒）

* 把优化问题转成判定问题：给定速度 k，能否在 H 小时内吃完，即 check(k)。
* 这个可行性随 k 单调（k 越大越容易），因此存在阈值 T，k < T 不可行，k ≥ T 可行。
* 在 [lower, upper] 上用二分找最小可行 k，检查一次是 O(n)，总体 O(n log upper)。
* 这类“二分答案”适用于有边界、有单调、好检查的题，常见在容量/分割/调度类问题。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 666681185 | 2025-09-28 14:56:32 CST | python3 | Accepted | 178 ms | 19 MB |
| 676899501 | 2025-11-09 17:13:42 CST | python3 | Accepted | 139 ms | 19 MB |
| 677070601 | 2025-11-10 14:18:31 CST | python3 | Accepted | 147 ms | 18.8 MB |

### 未通过提交代码
(所有提交均已通过)

## 前 K 个高频元素 (`top-k-frequent-elements`)

- 题目链接：https://leetcode.cn/problems/top-k-frequent-elements/
- 难度：Medium
- 标签：数组, 哈希表, 分治, 桶排序, 计数, 快速选择, 排序, 堆（优先队列）
- 总提交次数：3
- 最近提交时间：2025-11-08 17:24:42 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-11-08 17:59:35 CST

```markdown
本质一句话
* 小顶堆天生适合做“Top-K 最大”的题，因为它把“第 k 名”的门槛放在堆顶，让每个新候选只需和门槛比一次：赢了进、输了走。这样只为“有希望进前 k 的少数元素”支付 O(log k) 的代价，其余大多数元素几乎 O(1) 就被淘汰，整体达到 O(m log k)（m 为不同元素数）。
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 676706168 | 2025-11-08 17:22:40 CST | python3 | Runtime Error | N/A | N/A |
| 676706407 | 2025-11-08 17:23:35 CST | python3 | Runtime Error | N/A | N/A |
| 676706680 | 2025-11-08 17:24:42 CST | python3 | Accepted | 4 ms | 21 MB |

### 未通过提交代码
#### 提交 676706168 · Runtime Error · 2025-11-08 17:22:40 CST · python3

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq_map = collections.Counter(nums)
        min_heap = []
        for num, freq in freq_map.items():
            if len(min_heap) < k:
                min_heap.heappush(freq, num)
            else:
                if freq > min_heap[0][0]:
                    min_heap.heapreplace(freq, num)
        res = [num for _, num in min_heap]
        return res
```

#### 提交 676706407 · Runtime Error · 2025-11-08 17:23:35 CST · python3

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq_map = collections.Counter(nums)
        min_heap = []
        for num, freq in freq_map.items():
            if len(min_heap) < k:
                heapq.heappush(min_heap, (freq, num))
            else:
                if freq > min_heap[0][0]:
                    min_heap.heapreplace(freq, num)
        res = [num for _, num in min_heap]
        return res
```


## 搜索旋转排序数组 (`search-in-rotated-sorted-array`)

- 题目链接：https://leetcode.cn/problems/search-in-rotated-sorted-array/
- 难度：Medium
- 标签：数组, 二分查找
- 总提交次数：1
- 最近提交时间：2025-09-28 13:40:00 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 666660752 | 2025-09-28 13:40:00 CST | python3 | Accepted | 0 ms | 17.8 MB |

### 未通过提交代码
(所有提交均已通过)

## 搜索插入位置 (`search-insert-position`)

- 题目链接：https://leetcode.cn/problems/search-insert-position/
- 难度：Easy
- 标签：数组, 二分查找
- 总提交次数：1
- 最近提交时间：2025-09-28 11:11:33 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 666622245 | 2025-09-28 11:11:33 CST | python3 | Accepted | 0 ms | 18 MB |

### 未通过提交代码
(所有提交均已通过)

## 在排序数组中查找元素的第一个和最后一个位置 (`find-first-and-last-position-of-element-in-sorted-array`)

- 题目链接：https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/
- 难度：Medium
- 标签：数组, 二分查找
- 总提交次数：3
- 最近提交时间：2025-09-26 16:46:11 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 666234568 | 2025-09-26 16:39:51 CST | python3 | Accepted | 0 ms | 18.7 MB |
| 666236832 | 2025-09-26 16:45:38 CST | python3 | Runtime Error | N/A | N/A |
| 666237042 | 2025-09-26 16:46:11 CST | python3 | Accepted | 0 ms | 18.8 MB |

### 未通过提交代码
#### 提交 666236832 · Runtime Error · 2025-09-26 16:45:38 CST · python3

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left_bound = self.bisect.bisect_left(nums, target)
        if left_bound == len(nums) or nums[left_bound] != target:
            return [-1, -1]

        right_bound = self.bisect.bisect_right(nums, target) - 1
        return [left_bound, right_bound]
```


## 二分查找 (`binary-search`)

- 题目链接：https://leetcode.cn/problems/binary-search/
- 难度：Easy
- 标签：数组, 二分查找
- 总提交次数：1
- 最近提交时间：2025-09-26 14:27:06 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 666185368 | 2025-09-26 14:27:06 CST | python3 | Accepted | 0 ms | 18.5 MB |

### 未通过提交代码
(所有提交均已通过)

## 有序数组的平方 (`squares-of-a-sorted-array`)

- 题目链接：https://leetcode.cn/problems/squares-of-a-sorted-array/
- 难度：Easy
- 标签：数组, 双指针, 排序
- 总提交次数：1
- 最近提交时间：2025-09-25 10:29:55 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 665835843 | 2025-09-25 10:29:55 CST | python3 | Accepted | 15 ms | 19.5 MB |

### 未通过提交代码
(所有提交均已通过)

## 两数之和 II - 输入有序数组 (`two-sum-ii-input-array-is-sorted`)

- 题目链接：https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/
- 难度：Medium
- 标签：数组, 双指针, 二分查找
- 总提交次数：2
- 最近提交时间：2025-09-24 13:59:57 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 665591533 | 2025-09-24 13:59:45 CST | python3 | Runtime Error | N/A | N/A |
| 665591574 | 2025-09-24 13:59:57 CST | python3 | Accepted | 3 ms | 18.2 MB |

### 未通过提交代码
#### 提交 665591533 · Runtime Error · 2025-09-24 13:59:45 CST · python3

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
            current_sum = nums[left] + nums[right]
            if current_sum == target:
                return [left+1, right+1]
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        return []
```


## 旋转链表 (`rotate-list`)

- 题目链接：https://leetcode.cn/problems/rotate-list/
- 难度：Medium
- 标签：链表, 双指针
- 总提交次数：2
- 最近提交时间：2025-09-24 08:03:11 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 665504805 | 2025-09-24 08:02:39 CST | python3 | Wrong Answer | N/A | N/A |
| 665504822 | 2025-09-24 08:03:11 CST | python3 | Accepted | 3 ms | 17.5 MB |

### 未通过提交代码
#### 提交 665504805 · Wrong Answer · 2025-09-24 08:02:39 CST · python3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 边界情况处理
        if not head or not head.next or k == 0:
            return head

        # 先遍历出链表的总长度
        cur = head
        n = 1
        while cur.next:
            cur = cur.next
            n += 1
        # 实际有效的旋转次数
        valid_rotate = k % n
        # 链表成环
        cur.next = head
        # 确定从原链表头结点到新链表尾节点需要走的步数
        steps_to_new_tail = n - valid_rotate - 1 
        # 找到新链表的尾节点
        new_tail = head
        for _ in range(steps_to_new_tail):
            new_tail = new_tail.next
        # 确定新链表的头结点
        new_head = new_tail.next
        # 断开环
        new_tail.next = None
        return new_tail
```


## 航班预订统计 (`corporate-flight-bookings`)

- 题目链接：https://leetcode.cn/problems/corporate-flight-bookings/
- 难度：Medium
- 标签：数组, 前缀和
- 总提交次数：8
- 最近提交时间：2025-09-20 18:56:25 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 664197412 | 2025-09-19 14:48:11 CST | python3 | Runtime Error | N/A | N/A |
| 664197682 | 2025-09-19 14:48:55 CST | python3 | Accepted | 47 ms | 28.4 MB |
| 664197883 | 2025-09-19 14:49:30 CST | python3 | Runtime Error | N/A | N/A |
| 664198003 | 2025-09-19 14:49:49 CST | python3 | Accepted | 51 ms | 28.3 MB |
| 664516223 | 2025-09-20 18:52:25 CST | python3 | Runtime Error | N/A | N/A |
| 664516362 | 2025-09-20 18:53:12 CST | python3 | Runtime Error | N/A | N/A |
| 664516692 | 2025-09-20 18:55:15 CST | python3 | Runtime Error | N/A | N/A |
| 664516892 | 2025-09-20 18:56:25 CST | python3 | Accepted | 47 ms | 28.4 MB |

### 未通过提交代码
#### 提交 664197412 · Runtime Error · 2025-09-19 14:48:11 CST · python3

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        res = [0] * n
        diff = Diff(res)
        for booking in bookings:
            diff.increase(booking[0]-1, booking[1]-1, booking[2])
        return diff.result()


class Diff:
    def __init__(self, nums: list[int]) -> list[int]:
        if not nums:
            raise ValueError("输入数组不能为空。")
        self.diff = [0] * len(nums)
        self.diff[0] = nums[0]
        for i in range(1, len(nums)):
            self.diff[i] = nums[i] - nums[i-1]

    def increase(self, i, j, delta):
        if not (0 <= i <= j < len(self.diff)):
            raise IndexError("索引越界")
        self.diff[i] += delta
        if j+1 < len(self.diff):
            self.diff[j+1] -= delta

        def result(self)  -> list[int]:
            res = [0] * len(self.diff)
            res[0] = self.diff[0]
            for i in range(1, len(self.diff)):
                res[i] = res[i-1] + self.diff[i]
            return res
```

#### 提交 664197883 · Runtime Error · 2025-09-19 14:49:30 CST · python3

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        res = [0] * n
        diff = Diff(res)
        for booking in bookings:
            diff.increase(booking[0]-1, booking[1]-1, booking[2])
        return diff.result()


    class Diff:
        def __init__(self, nums: list[int]) -> list[int]:
            if not nums:
                raise ValueError("输入数组不能为空。")
            self.diff = [0] * len(nums)
            self.diff[0] = nums[0]
            for i in range(1, len(nums)):
                self.diff[i] = nums[i] - nums[i-1]
    
        def increase(self, i, j, delta):
            if not (0 <= i <= j < len(self.diff)):
                raise IndexError("索引越界")
            self.diff[i] += delta
            if j+1 < len(self.diff):
                self.diff[j+1] -= delta
    
        def result(self)  -> list[int]:
            res = [0] * len(self.diff)
            res[0] = self.diff[0]
            for i in range(1, len(self.diff)):
                res[i] = res[i-1] + self.diff[i]
            return res
```

#### 提交 664516223 · Runtime Error · 2025-09-20 18:52:25 CST · python3

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        nums = [0] * n
        diff = Difference(nums)
        for booking in bookings:
            diff.increasment(booking[0]-1, booking[1]-1, booking[2])
        return diff.result


class Difference:
    def __init__(self, nums: list[int]):
        if not nums:
            raise ValueError('Input empty')
        self._len = len(nums)
        self._diff = [0] * self._len
        self._diff[0] = nums[0]
        for i in range(1, self._len):
            self._diff[i] = nums[i] - nums[i-1]

    def increasment(self, i, j, delta):
        if not (0 <= i <= j < self._len):
            raise IndexError('Index error')
        self.diff[i] += delta
        if j+1 < self._len:
            self.diff[j+1] -= delta

    def result(self) -> list[int]:
        res = [0] * self._len
        res[0] = self._diff[0]
        for i in range(1, self._len):
            res[i] = res[i-1] + self.diff[i]
        return res
```

#### 提交 664516362 · Runtime Error · 2025-09-20 18:53:12 CST · python3

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        nums = [0] * n
        diff = Difference(nums)
        for booking in bookings:
            diff.increasment(booking[0]-1, booking[1]-1, booking[2])
        return diff.result


class Difference:
    def __init__(self, nums: list[int]):
        if not nums:
            raise ValueError('Input empty')
        self._len = len(nums)
        self._diff = [0] * self._len
        self._diff[0] = nums[0]
        for i in range(1, self._len):
            self._diff[i] = nums[i] - nums[i-1]

    def increasment(self, i, j, delta):
        if not (0 <= i <= j < self._len):
            raise IndexError('Index error')
        self._diff[i] += delta
        if j+1 < self._len:
            self._diff[j+1] -= delta

    def result(self) -> list[int]:
        res = [0] * self._len
        res[0] = self._diff[0]
        for i in range(1, self._len):
            res[i] = res[i-1] + self._diff[i]
        return res
```

#### 提交 664516692 · Runtime Error · 2025-09-20 18:55:15 CST · python3

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        nums = [0] * n
        diff = Difference(nums)
        for booking in bookings:
            diff.increasment(booking[0]-1, booking[1]-1, booking[2])
        return diff.result


class Difference:
    def __init__(self, nums: list[int]):
        if not nums:
            raise ValueError('Input empty')
        self._len = len(nums)
        self._diff = [0] * self._len
        self._diff[0] = nums[0]
        for i in range(1, self._len):
            self._diff[i] = nums[i] - nums[i-1]

    def increasment(self, i, j, delta):
        if not (0 <= i <= j < self._len):
            raise IndexError('Index error')
        self._diff[i] += delta
        if j+1 < self._len:
            self._diff[j+1] -= delta

    def result(self) -> list[int]:
        res = [0] * self._len
        res[0] = self._diff[0]
        for i in range(1, self._len):
            res[i] = res[i-1] + self._diff[i]
        return res
```


## 连续数组 (`contiguous-array`)

- 题目链接：https://leetcode.cn/problems/contiguous-array/
- 难度：Medium
- 标签：数组, 哈希表, 前缀和
- 总提交次数：8
- 最近提交时间：2025-09-18 18:59:53 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 662051475 | 2025-09-12 15:49:44 CST | python3 | Wrong Answer | N/A | N/A |
| 662051824 | 2025-09-12 15:50:30 CST | python3 | Wrong Answer | N/A | N/A |
| 662053043 | 2025-09-12 15:53:17 CST | python3 | Runtime Error | N/A | N/A |
| 662053307 | 2025-09-12 15:53:52 CST | python3 | Accepted | 153 ms | 23.5 MB |
| 662212265 | 2025-09-13 08:23:46 CST | python3 | Runtime Error | N/A | N/A |
| 662212277 | 2025-09-13 08:23:57 CST | python3 | Accepted | 105 ms | 23 MB |
| 663975731 | 2025-09-18 18:58:12 CST | python3 | Wrong Answer | N/A | N/A |
| 663976102 | 2025-09-18 18:59:53 CST | python3 | Accepted | 115 ms | 23 MB |

### 未通过提交代码
#### 提交 662051475 · Wrong Answer · 2025-09-12 15:49:44 CST · python3

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        pre_sum = [0] * (n+1)
        for i in range(1, n+1):
            pre_sum[i] == pre_sum[i-1] + (-1 if nums[n-1] == 0 else 1)

        res = 0
        sum_to_index = {}
        for i in range(n+1):
            if pre_sum[i] not in sum_to_index:
                sum_to_index[pre_sum[i]] = i
            else:
                res = max(res, i-sum_to_index[pre_sum[i]])
        return res
```

#### 提交 662051824 · Wrong Answer · 2025-09-12 15:50:30 CST · python3

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        pre_sum = [0] * (n+1)
        for i in range(1, n+1):
            pre_sum[i] == pre_sum[i-1] + (-1 if nums[i-1] == 0 else 1)

        res = 0
        sum_to_index = {}
        for i in range(n+1):
            if pre_sum[i] not in sum_to_index:
                sum_to_index[pre_sum[i]] = i
            else:
                res = max(res, i-sum_to_index[pre_sum[i]])
        return res
```

#### 提交 662053043 · Runtime Error · 2025-09-12 15:53:17 CST · python3

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        pre_sum = [0] * (n+1)
        for i in range(1, n+1):
            pre_sum[i] == pre_sum[i-1] + (-1 if nums[i] == 0 else 1)

        res = 0
        sum_to_index = {}
        for i in range(n+1):
            if pre_sum[i] not in sum_to_index:
                sum_to_index[pre_sum[i]] = i
            else:
                res = max(res, i-sum_to_index[pre_sum[i]])
        return res
```

#### 提交 662212265 · Runtime Error · 2025-09-13 08:23:46 CST · python3

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        max_len = 0
        pre_sum = 0
        sum_to_index = {0: -1}
        for i, num in enumerate(nums):
            if num == 0:
                pre_sum += -1
            else:
                pre_sum += 1

            if  pre_sum in sum_to_index:
                max_len = max(max_len, i - sum_to_index[pre_sum])
            else:
                sum_to_index[pre_sum] == i
        return max_len
```

#### 提交 663975731 · Wrong Answer · 2025-09-18 18:58:12 CST · python3

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        max_len = 0
        sum_index_map = {0: -1}
        pre_sum = 0
        for i, num in enumerate(nums):
            if num == 0:
                pre_sum += -1
            else:
                pre_sum += 1
            if pre_sum in sum_index_map:
                max_len = max(max_len, i-sum_index_map[pre_sum])
            else:
                sum_index_map[max_len] = i
        return max_len
```


## 除了自身以外数组的乘积 (`product-of-array-except-self`)

- 题目链接：https://leetcode.cn/problems/product-of-array-except-self/
- 难度：Medium
- 标签：数组, 前缀和
- 总提交次数：4
- 最近提交时间：2025-09-18 12:43:22 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 662006248 | 2025-09-12 13:52:37 CST | python3 | Wrong Answer | N/A | N/A |
| 662006311 | 2025-09-12 13:52:49 CST | python3 | Accepted | 31 ms | 26.5 MB |
| 662017082 | 2025-09-12 14:28:02 CST | python3 | Accepted | 19 ms | 22.9 MB |
| 663847982 | 2025-09-18 12:43:22 CST | python3 | Accepted | 27 ms | 23 MB |

### 未通过提交代码
#### 提交 662006248 · Wrong Answer · 2025-09-12 13:52:37 CST · python3

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n

        # 前缀和数组
        prefix = [0] * n
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] * nums[i]        

        # 后缀和数组
        suffix = [0] * n
        suffix[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix[i] = suffix[i+1] * nums[i]

        res[0] = suffix[1]
        res[n-1] = prefix[n-2]
        for i in range(1, n-1):
            res[i] = prefix[i-1] * suffix[i+1]
```


## 连续的子数组和 (`continuous-subarray-sum`)

- 题目链接：https://leetcode.cn/problems/continuous-subarray-sum/
- 难度：Medium
- 标签：数组, 哈希表, 数学, 前缀和
- 总提交次数：4
- 最近提交时间：2025-09-17 21:37:45 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 662218715 | 2025-09-13 09:34:03 CST | python3 | Accepted | 67 ms | 37.4 MB |
| 663709780 | 2025-09-17 21:36:02 CST | python3 | Wrong Answer | N/A | N/A |
| 663709889 | 2025-09-17 21:36:20 CST | python3 | Wrong Answer | N/A | N/A |
| 663710370 | 2025-09-17 21:37:45 CST | python3 | Accepted | 87 ms | 37.1 MB |

### 未通过提交代码
#### 提交 663709780 · Wrong Answer · 2025-09-17 21:36:02 CST · python3

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        pre_sum = 0
        remainder_map = {0: -1}
        for i, num in enumerate(nums):
            pre_sum += num
            remainder = pre_sum % num
            if remainder in remainder_map:
                if remainder_map[remainder] - i >= 2:
                    return True
            else:
                remainder_map[remainder] = i
        return False
```

#### 提交 663709889 · Wrong Answer · 2025-09-17 21:36:20 CST · python3

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        pre_sum = 0
        remainder_map = {0: -1}
        for i, num in enumerate(nums):
            pre_sum += num
            remainder = pre_sum % k
            if remainder in remainder_map:
                if remainder_map[remainder] - i >= 2:
                    return True
            else:
                remainder_map[remainder] = i
        return False
```


## 乘积小于 K 的子数组 (`subarray-product-less-than-k`)

- 题目链接：https://leetcode.cn/problems/subarray-product-less-than-k/
- 难度：Medium
- 标签：数组, 二分查找, 前缀和, 滑动窗口
- 总提交次数：11
- 最近提交时间：2025-09-16 18:46:00 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-08-13 09:15:37 CST

```markdown
错误1：更新结果的位置写错，应该写在窗口收缩之后
错误2：算res时只算了window_product<k的情况，没算出以right结尾的所有子数组的数量
错误3：收缩窗口的条件应该是while window_product >= k and left <= right
错误4：应该用地板除，而不是普通除法
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 652474128 | 2025-08-13 07:55:14 CST | python3 | Wrong Answer | N/A | N/A |
| 652474201 | 2025-08-13 07:57:21 CST | python3 | Wrong Answer | N/A | N/A |
| 652483130 | 2025-08-13 09:15:56 CST | python3 | Runtime Error | N/A | N/A |
| 652483240 | 2025-08-13 09:16:20 CST | python3 | Accepted | 67 ms | 19.5 MB |
| 660978517 | 2025-09-09 14:26:06 CST | python3 | Wrong Answer | N/A | N/A |
| 660981830 | 2025-09-09 14:34:11 CST | python3 | Wrong Answer | N/A | N/A |
| 660985429 | 2025-09-09 14:42:38 CST | python3 | Runtime Error | N/A | N/A |
| 660987884 | 2025-09-09 14:48:15 CST | python3 | Accepted | 67 ms | 19.5 MB |
| 663318591 | 2025-09-16 18:41:49 CST | python3 | Runtime Error | N/A | N/A |
| 663319434 | 2025-09-16 18:45:40 CST | python3 | Runtime Error | N/A | N/A |
| 663319518 | 2025-09-16 18:46:00 CST | python3 | Accepted | 83 ms | 19.5 MB |

### 未通过提交代码
#### 提交 652474128 · Wrong Answer · 2025-08-13 07:55:14 CST · python3

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        left = 0
        window_product = 1
        res = 0

        for right, r_num in enumerate(nums):
            window_product *= r_num
            if window_product < k:
                res += 1
            while window_product > k:
                l_num = nums[left]
                window_product = window_product / l_num
                l_num += 1
        
        return res
```

#### 提交 652474201 · Wrong Answer · 2025-08-13 07:57:21 CST · python3

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        left = 0
        window_product = 1
        res = 0

        for right, r_num in enumerate(nums):
            window_product *= r_num
            if window_product < k:
                res += 1
            while window_product > k:
                l_num = nums[left]
                window_product = window_product / l_num
                left += 1
        
        return res
```

#### 提交 652483130 · Runtime Error · 2025-08-13 09:15:56 CST · python3

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        left = 0
        window_product = 1
        res = 0

        for right, r_num in enumerate(nums):
            window_product *= r_num
            while window_product >= k:
                l_num = nums[left]
                window_product //=  l_num
                left += 1
            res += right - left + 1
        
        return res
```

#### 提交 660978517 · Wrong Answer · 2025-09-09 14:26:06 CST · python3

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        res = 0
        left = 0
        prod = 1
        for right, r_num in enumerate(nums):
            prod *= r_num
            while prod >= k:
                l_num = nums[left]
                prod /= l_num
                left += 1
            if prod < k:
                res += 1
        return prod
```

#### 提交 660981830 · Wrong Answer · 2025-09-09 14:34:11 CST · python3

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        res = 0
        left = 0
        prod = 1
        for right, r_num in enumerate(nums):
            prod *= r_num
            while prod >= k:
                l_num = nums[left]
                prod /= l_num
                left += 1
            res += 1
        return res
```

#### 提交 660985429 · Runtime Error · 2025-09-09 14:42:38 CST · python3

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        res = 0
        left = 0
        prod = 1
        for right, r_num in enumerate(nums):
            prod *= r_num
            while prod >= k:
                l_num = nums[left]
                prod /= l_num
                left += 1
            res += (right - left + 1)
        return res
```

#### 提交 663318591 · Runtime Error · 2025-09-16 18:41:49 CST · python3

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        product = 1
        res_cnt = 0
        left = 0
        for right, r_num in enumerate(nums):
            product *= r_num
            while product >= k:
                l_num = nums[left]
                product /= l_num
                left += 1
            res_cnt += right - left + 1
        return res_cnt
```

#### 提交 663319434 · Runtime Error · 2025-09-16 18:45:40 CST · python3

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        product = 1
        res_cnt = 0
        left = 0
        for right, r_num in enumerate(nums):
            product *= r_num
            while product >= k:
                l_num = nums[left]
                product /= l_num
                left += 1
            res_cnt += (right - left + 1)
        return res_cnt
```


## 表现良好的最长时间段 (`longest-well-performing-interval`)

- 题目链接：https://leetcode.cn/problems/longest-well-performing-interval/
- 难度：Medium
- 标签：栈, 数组, 哈希表, 前缀和, 单调栈
- 总提交次数：3
- 最近提交时间：2025-09-14 09:06:03 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 662484097 | 2025-09-14 09:05:12 CST | python3 | Wrong Answer | N/A | N/A |
| 662484166 | 2025-09-14 09:05:51 CST | python3 | Accepted | 31 ms | 18.2 MB |
| 662484189 | 2025-09-14 09:06:03 CST | python3 | Accepted | 19 ms | 18.1 MB |

### 未通过提交代码
#### 提交 662484097 · Wrong Answer · 2025-09-14 09:05:12 CST · python3

```python
class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        # 问题归一化，大于 8 小时记为1，小于等于 8 小时记为-1，问题转化为 最长的和大于 0 的子数组
        # 前缀和 + 哈希表：用哈希表记录前缀和值第一次出现的索引位置（因为我们希望子数组长度最大，索引j 尽可能的小）
        first_sum_map = {}
        current_sum = 0
        max_len = 0
        for i, num in enumerate(hours):
            score = 1 if num > 8 else -1
            current_sum += score
            if current_sum > 0: # 此时说明从数组开头到当前索引i，都是表现好的
                max_len = max(max_len, i+1)
            else:
                need = current_sum-1
                if need in first_sum_map:  # 前缀和的步长变化幅度为 1，因此current_sum-1一定是离i 最远的
                    max_len = max(max_len, i-(first_sum_map.get(need)))
            
            first_sum_map[current_sum] = i
        return max_len
```


## 和可被 K 整除的子数组 (`subarray-sums-divisible-by-k`)

- 题目链接：https://leetcode.cn/problems/subarray-sums-divisible-by-k/
- 难度：Medium
- 标签：数组, 哈希表, 前缀和
- 总提交次数：1
- 最近提交时间：2025-09-13 10:56:07 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 662237835 | 2025-09-13 10:56:07 CST | python3 | Accepted | 23 ms | 20.6 MB |

### 未通过提交代码
(所有提交均已通过)

## 最后 K 个数的乘积 (`product-of-the-last-k-numbers`)

- 题目链接：https://leetcode.cn/problems/product-of-the-last-k-numbers/
- 难度：Medium
- 标签：设计, 数组, 数学, 数据流, 前缀和
- 总提交次数：1
- 最近提交时间：2025-09-12 14:56:13 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 662028263 | 2025-09-12 14:56:13 CST | python3 | Accepted | 55 ms | 31.2 MB |

### 未通过提交代码
(所有提交均已通过)

## 矩阵区域和 (`matrix-block-sum`)

- 题目链接：https://leetcode.cn/problems/matrix-block-sum/
- 难度：Medium
- 标签：数组, 矩阵, 前缀和
- 总提交次数：2
- 最近提交时间：2025-09-12 10:11:43 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 661940310 | 2025-09-12 10:11:25 CST | python3 | Runtime Error | N/A | N/A |
| 661940425 | 2025-09-12 10:11:43 CST | python3 | Accepted | 24 ms | 18.7 MB |

### 未通过提交代码
#### 提交 661940310 · Runtime Error · 2025-09-12 10:11:25 CST · python3

```python
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        num_mat = NumMatrix(mat)
        res = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                x1 = max(i-k, 0)
                y1 = max(j-k, 0)
                x2 = min(i+k, m-1)
                y2 = min(j+k, n-1)
                res[i][j] = num_mat.sub_sum(x1, y1, x2, y2)
        return res
        

class NumMatrix:
    def __init__(self. matrix: list[list[int]]):
        m = len(matrix)
        n = len(matrix[0])
        self.pre_sum = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                self.pre_sum[i][j] = self.pre_sum[i-1][j] + self.pre_sum[i][j-1] + matrix[i-1][j-1] - self.pre_sum[i-1][j-1]
    
    def sub_sum(self, x1: int, y1: int, x2: int, y2: int) -> int:
        return self.pre_sum[x2+1][y2+1] - self.pre_sum[x2+1][y1] - self.pre_sum[x1][y2+1] + self.pre_sum[x1][y1]
```


## 二维区域和检索 - 矩阵不可变 (`range-sum-query-2d-immutable`)

- 题目链接：https://leetcode.cn/problems/range-sum-query-2d-immutable/
- 难度：Medium
- 标签：设计, 数组, 矩阵, 前缀和
- 总提交次数：9
- 最近提交时间：2025-09-12 09:42:52 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 661664134 | 2025-09-11 13:37:21 CST | python3 | Runtime Error | N/A | N/A |
| 661665131 | 2025-09-11 13:41:31 CST | python3 | Wrong Answer | N/A | N/A |
| 661666526 | 2025-09-11 13:46:54 CST | python3 | Accepted | 113 ms | 30.7 MB |
| 661913792 | 2025-09-12 07:34:40 CST | python3 | Runtime Error | N/A | N/A |
| 661913795 | 2025-09-12 07:34:48 CST | python3 | Wrong Answer | N/A | N/A |
| 661913833 | 2025-09-12 07:36:11 CST | python3 | Wrong Answer | N/A | N/A |
| 661913849 | 2025-09-12 07:36:58 CST | python3 | Wrong Answer | N/A | N/A |
| 661915426 | 2025-09-12 08:16:57 CST | python3 | Accepted | 130 ms | 30.7 MB |
| 661930600 | 2025-09-12 09:42:52 CST | python3 | Accepted | 100 ms | 30.7 MB |

### 未通过提交代码
#### 提交 661664134 · Runtime Error · 2025-09-11 13:37:21 CST · python3

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        self.pre_sum = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                self.pre_sum[i][j] = self.pre_sum[i-1][j] + self.pre_sum[i][j-1] + matrix[i][j] - self.pre_sum[i-1][j-1]
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row1+1][col1+1] - self.pre_sum[row1][col1+1] - self.pre_sum[row1+1][col1] + self.pre_sum[row2][col2]
        
        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```

#### 提交 661665131 · Wrong Answer · 2025-09-11 13:41:31 CST · python3

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        self.pre_sum = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                self.pre_sum[i][j] = self.pre_sum[i-1][j] + self.pre_sum[i][j-1] + matrix[i-1][j-1] - self.pre_sum[i-1][j-1]
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row1+1][col1+1] - self.pre_sum[row1][col1+1] - self.pre_sum[row1+1][col1] + self.pre_sum[row2][col2]

        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```

#### 提交 661913792 · Runtime Error · 2025-09-12 07:34:40 CST · python3

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        self.pre_sum = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1)j:
            for j in range(1, n+1):
                self.pre_sum[i][j] = self.pre_sum[i-1][j] + self.pre_sum[i][j-1] + matrix[i-1][j-1] - self.pre_sum[i-1][j-1]
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row2+1][col2+1] - self.pre_sum[row2+1][col1] - self.pre_sum[row2][col2+1] - self.pre_sum[col1][row1] 

        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```

#### 提交 661913795 · Wrong Answer · 2025-09-12 07:34:48 CST · python3

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        self.pre_sum = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                self.pre_sum[i][j] = self.pre_sum[i-1][j] + self.pre_sum[i][j-1] + matrix[i-1][j-1] - self.pre_sum[i-1][j-1]
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row2+1][col2+1] - self.pre_sum[row2+1][col1] - self.pre_sum[row2][col2+1] - self.pre_sum[col1][row1] 

        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```

#### 提交 661913833 · Wrong Answer · 2025-09-12 07:36:11 CST · python3

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        self.pre_sum = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                self.pre_sum[i][j] = self.pre_sum[i-1][j] + self.pre_sum[i][j-1] + matrix[i-1][j-1] - self.pre_sum[i-1][j-1]
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row2+1][col2+1] - self.pre_sum[row2+1][col1] - self.pre_sum[row1][col2+1] - self.pre_sum[col1][row1] 

        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```

#### 提交 661913849 · Wrong Answer · 2025-09-12 07:36:58 CST · python3

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        self.pre_sum = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                self.pre_sum[i][j] = self.pre_sum[i-1][j] + self.pre_sum[i][j-1] + matrix[i-1][j-1] - self.pre_sum[i-1][j-1]
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row2+1][col2+1] - self.pre_sum[row2+1][col1] - self.pre_sum[row1][col2+1] + self.pre_sum[col1][row1] 

        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```


## 寻找数组的中心下标 (`find-pivot-index`)

- 题目链接：https://leetcode.cn/problems/find-pivot-index/
- 难度：Easy
- 标签：数组, 前缀和
- 总提交次数：3
- 最近提交时间：2025-09-11 15:53:00 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 661683743 | 2025-09-11 14:41:04 CST | python3 | Wrong Answer | N/A | N/A |
| 661692568 | 2025-09-11 15:00:04 CST | python3 | Accepted | 11 ms | 18.3 MB |
| 661717474 | 2025-09-11 15:53:00 CST | python3 | Accepted | 3 ms | 18.3 MB |

### 未通过提交代码
#### 提交 661683743 · Wrong Answer · 2025-09-11 14:41:04 CST · python3

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        pre_sum = [0] * (len(nums) + 1)
        for i in range(1, len(pre_sum)):
            pre_sum[i] = pre_sum[i-1] + nums[i-1]
        
        if pre_sum[-1] == 0:
            return 0
        
        for i in range(len(nums)):
            if nums[i] + 2*pre_sum[i] == pre_sum[-1]:
                return i
        
        return -1
```


## 区域和检索 - 数组不可变 (`range-sum-query-immutable`)

- 题目链接：https://leetcode.cn/problems/range-sum-query-immutable/
- 难度：Easy
- 标签：设计, 数组, 前缀和
- 总提交次数：2
- 最近提交时间：2025-09-11 10:56:46 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 661615954 | 2025-09-11 10:56:23 CST | python3 | Runtime Error | N/A | N/A |
| 661616160 | 2025-09-11 10:56:46 CST | python3 | Accepted | 3 ms | 21.4 MB |

### 未通过提交代码
#### 提交 661615954 · Runtime Error · 2025-09-11 10:56:23 CST · python3

```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.pre_sum = [0] * (len(nums)+1)
        for i in range(1, len(self.pre_sum)):
            pre_sum[i] = pre_sum[i-1] + nums[i-1]
        

    def sumRange(self, left: int, right: int) -> int:
        return self.pre_sum[right+1] - self.pre_sum[left]
        


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)
```


## 至少有 K 个重复字符的最长子串 (`longest-substring-with-at-least-k-repeating-characters`)

- 题目链接：https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/
- 难度：Medium
- 标签：哈希表, 字符串, 分治, 滑动窗口
- 总提交次数：7
- 最近提交时间：2025-09-10 16:41:27 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-08-22 14:34:22 CST

```markdown
卡点1：缩小完窗口后写结果更新时不会写了，窗口内字符的种类数==h，且每种字符的出现次数等于k次，和valid_count的关系是什么 -> valid_count不就应该等于h吗！！
错误1：缩小窗口时又忘记left+=1
错误2：left指针错误的写在了外循环的外面
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 655479008 | 2025-08-22 14:24:38 CST | python3 | Runtime Error | N/A | N/A |
| 655479093 | 2025-08-22 14:24:51 CST | python3 | Time Limit Exceeded | N/A | N/A |
| 655479443 | 2025-08-22 14:25:52 CST | python3 | Wrong Answer | N/A | N/A |
| 655480630 | 2025-08-22 14:29:23 CST | python3 | Accepted | 47 ms | 17.4 MB |
| 661391600 | 2025-09-10 16:33:35 CST | python3 | Runtime Error | N/A | N/A |
| 661391803 | 2025-09-10 16:33:58 CST | python3 | Wrong Answer | N/A | N/A |
| 661395750 | 2025-09-10 16:41:27 CST | python3 | Accepted | 55 ms | 17.8 MB |

### 未通过提交代码
#### 提交 655479008 · Runtime Error · 2025-08-22 14:24:38 CST · python3

```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        max_len = 0
        left = 0
        # 因为都是小写字母，所以窗口内最多有26种不同的字母，所以把字母的种类数从1到26都遍历一遍
        for h in range(1, 27):
            windows = collections.defaultdict(int)
            uniq_count = 0
            valid_count = 0

            for right, right_char in enumerate(s):
                if windows[right_char] == 0:
                    uniq_count += 1
                windows[right_char] += 1
                if windows[right_char] == k:
                    valid_count += 1
                
                while uniq_count > h:
                    left_char = s[left]
                    if windows[left_char] == k:
                        valid_count -= 1
                    windows[left_char] -= 1
                    if windows[left_char] = 0:
                        uniq_count -= 1
                
                if uniq_count == h and valid_count == h:
                    max_len = max(max_len, right - left + 1)
        
        return max_len
```

#### 提交 655479093 · Time Limit Exceeded · 2025-08-22 14:24:51 CST · python3

```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        max_len = 0
        left = 0
        # 因为都是小写字母，所以窗口内最多有26种不同的字母，所以把字母的种类数从1到26都遍历一遍
        for h in range(1, 27):
            windows = collections.defaultdict(int)
            uniq_count = 0
            valid_count = 0

            for right, right_char in enumerate(s):
                if windows[right_char] == 0:
                    uniq_count += 1
                windows[right_char] += 1
                if windows[right_char] == k:
                    valid_count += 1
                
                while uniq_count > h:
                    left_char = s[left]
                    if windows[left_char] == k:
                        valid_count -= 1
                    windows[left_char] -= 1
                    if windows[left_char] == 0:
                        uniq_count -= 1
                
                if uniq_count == h and valid_count == h:
                    max_len = max(max_len, right - left + 1)
        
        return max_len
```

#### 提交 655479443 · Wrong Answer · 2025-08-22 14:25:52 CST · python3

```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        max_len = 0
        left = 0
        # 因为都是小写字母，所以窗口内最多有26种不同的字母，所以把字母的种类数从1到26都遍历一遍
        for h in range(1, 27):
            windows = collections.defaultdict(int)
            uniq_count = 0
            valid_count = 0

            for right, right_char in enumerate(s):
                if windows[right_char] == 0:
                    uniq_count += 1
                windows[right_char] += 1
                if windows[right_char] == k:
                    valid_count += 1
                
                while uniq_count > h:
                    left_char = s[left]
                    if windows[left_char] == k:
                        valid_count -= 1
                    windows[left_char] -= 1
                    if windows[left_char] == 0:
                        uniq_count -= 1
                    left += 1
                
                if uniq_count == h and valid_count == h:
                    max_len = max(max_len, right - left + 1)
        
        return max_len
```

#### 提交 661391600 · Runtime Error · 2025-09-10 16:33:35 CST · python3

```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        # 字符一共就26种，加一个约束条件，要求子串中字符的种类正好是 h，每种字符出现次数都不少于 k
        max_len = 0
        for h in range(1, 27):
            left = valid_count = uniq_count = 0
            window = collections.defaultdict(int)
            for right, r_char in enumerate(s):
                if window[r_char] == 0:
                    uniq_count += 1
                window[r_char] += 1
                if window[r_char] == k:
                    valid += 1
                while uniq_count > h:
                    l_char = s[left]
                    if window[l_char] == k:
                        valid -= 1
                    window[l_char] -= 1
                    if window[l_char] == 0:
                        uniq_count -= 1
                    left += 1
                max_len = max(max_len, right - left + 1)
        return max_len
```

#### 提交 661391803 · Wrong Answer · 2025-09-10 16:33:58 CST · python3

```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        # 字符一共就26种，加一个约束条件，要求子串中字符的种类正好是 h，每种字符出现次数都不少于 k
        max_len = 0
        for h in range(1, 27):
            left = valid_count = uniq_count = 0
            window = collections.defaultdict(int)
            for right, r_char in enumerate(s):
                if window[r_char] == 0:
                    uniq_count += 1
                window[r_char] += 1
                if window[r_char] == k:
                    valid_count += 1
                while uniq_count > h:
                    l_char = s[left]
                    if window[l_char] == k:
                        valid_count -= 1
                    window[l_char] -= 1
                    if window[l_char] == 0:
                        uniq_count -= 1
                    left += 1
                max_len = max(max_len, right - left + 1)
        return max_len
```


## 长度最小的子数组 (`minimum-size-subarray-sum`)

- 题目链接：https://leetcode.cn/problems/minimum-size-subarray-sum/
- 难度：Medium
- 标签：数组, 二分查找, 前缀和, 滑动窗口
- 总提交次数：4
- 最近提交时间：2025-09-10 15:07:04 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 659092759 | 2025-09-03 14:38:49 CST | python3 | Wrong Answer | N/A | N/A |
| 659095384 | 2025-09-03 14:45:06 CST | python3 | Accepted | 11 ms | 27.9 MB |
| 661347716 | 2025-09-10 15:06:11 CST | python3 | Wrong Answer | N/A | N/A |
| 661348138 | 2025-09-10 15:07:04 CST | python3 | Accepted | 8 ms | 27.7 MB |

### 未通过提交代码
#### 提交 659092759 · Wrong Answer · 2025-09-03 14:38:49 CST · python3

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_len = float('inf')
        total_sum = 0
        left = 0
        for right, r_num in enumerate(nums):
            total_sum += r_num
            while total_sum >= target:
                total_sum -= nums[left]
                left += 1
            min_len = min(right - left + 1, min_len)
        return min_len if min_len != float('inf') else 0
```

#### 提交 661347716 · Wrong Answer · 2025-09-10 15:06:11 CST · python3

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_len = float('inf')
        left = 0
        current_sum = 0
        for right, r_num in enumerate(nums):
            current_sum += r_num
            while current_sum >= target:
                min_len = min(min_len, right - left + 1)
                l_num = nums[left]
                current_sum -= l_num
                left += 1
        return min_len
```


## 最大连续1的个数 III (`max-consecutive-ones-iii`)

- 题目链接：https://leetcode.cn/problems/max-consecutive-ones-iii/
- 难度：Medium
- 标签：数组, 二分查找, 前缀和, 滑动窗口
- 总提交次数：3
- 最近提交时间：2025-09-09 15:30:52 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-08-14 07:49:18 CST

```markdown
错误1：更新结果时错误的写成了res = max(res, right - left + 1 - zero_count)
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 652815748 | 2025-08-14 07:45:27 CST | python3 | Wrong Answer | N/A | N/A |
| 652815826 | 2025-08-14 07:48:12 CST | python3 | Accepted | 83 ms | 20.1 MB |
| 661008148 | 2025-09-09 15:30:52 CST | python3 | Accepted | 150 ms | 19.9 MB |

### 未通过提交代码
#### 提交 652815748 · Wrong Answer · 2025-08-14 07:45:27 CST · python3

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        res = 0
        zero_count = 0

        for right, r_num in enumerate(nums):
            if r_num == 0:
                zero_count += 1
            while zero_count > k:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1
            res = max(res, right - left + 1 - zero_count)
        
        return res
```


## 将 x 减到 0 的最小操作数 (`minimum-operations-to-reduce-x-to-zero`)

- 题目链接：https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/
- 难度：Medium
- 标签：数组, 哈希表, 二分查找, 前缀和, 滑动窗口
- 总提交次数：8
- 最近提交时间：2025-09-09 13:33:31 CST

### 题目笔记
#### 笔记 1 · 更新于 2025-08-12 09:02:52 CST

```markdown
错误1：扩大窗口时应该是先更新windown_sum再right+1，而不是先right += 1再window_sum += nums[right]
 错误2：更新结果时的条件判断写错，应该是if window_sum == target:
                max_len = max(max_len, right-left)
错误3：窗口缩小时错误的写成了left-=1
```

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 652125177 | 2025-08-12 08:58:41 CST | python3 | Wrong Answer | N/A | N/A |
| 652125693 | 2025-08-12 09:02:17 CST | python3 | Accepted | 116 ms | 28.8 MB |
| 660955477 | 2025-09-09 13:02:06 CST | python3 | Wrong Answer | N/A | N/A |
| 660956350 | 2025-09-09 13:06:29 CST | python3 | Wrong Answer | N/A | N/A |
| 660956388 | 2025-09-09 13:06:44 CST | python3 | Wrong Answer | N/A | N/A |
| 660962117 | 2025-09-09 13:32:43 CST | python3 | Wrong Answer | N/A | N/A |
| 660962242 | 2025-09-09 13:33:15 CST | python3 | Runtime Error | N/A | N/A |
| 660962304 | 2025-09-09 13:33:31 CST | python3 | Accepted | 97 ms | 28.6 MB |

### 未通过提交代码
#### 提交 652125177 · Wrong Answer · 2025-08-12 08:58:41 CST · python3

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # 题目转化为 寻找 和为sum(nums) - x 的最长子数组
        left = right = 0
        window_sum = 0
        target = sum(nums) - x
        max_len = float('-inf')

        while right < len(nums):
            window_sum += nums[right]
            right += 1

            while window_sum > target and left < right:
                window_sum -= nums[left]
                left -= 1
                
            if window_sum == target:
                max_len = max(max_len, right-left)
        
        return len(nums) - max_len if max_len != float('-inf') else -1
```

#### 提交 660955477 · Wrong Answer · 2025-09-09 13:02:06 CST · python3

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # 问题转化成 查找和为 total_sum - x 的最长子数组
        target = sum(nums) - x
        max_len = 0
        current_sum = 0
        left = 0

        for right, r_num in enumerate(nums):
            current_sum += r_num
            if current_sum == target:
                max_len = max(max_len, right-left+1)
            while current_sum > target:
                l_num = nums[left]
                current_sum -= l_num
                left += 1
        
        return len(nums) - max_len if max_len != 0 else -1
```

#### 提交 660956350 · Wrong Answer · 2025-09-09 13:06:29 CST · python3

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # 问题转化成 查找和为 total_sum - x 的最长子数组
        target = sum(nums) - x
        max_len = 0
        current_sum = 0
        left = 0

        for right, r_num in enumerate(nums):
            if current_sum == target:
                max_len = max(max_len, right-left+1)
            current_sum += r_num
            while current_sum > target:
                l_num = nums[left]
                current_sum -= l_num
                left += 1
        
        return len(nums) - max_len if max_len != 0 else -1
```

#### 提交 660956388 · Wrong Answer · 2025-09-09 13:06:44 CST · python3

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # 问题转化成 查找和为 total_sum - x 的最长子数组
        target = sum(nums) - x
        max_len = 0
        current_sum = 0
        left = 0

        for right, r_num in enumerate(nums):
            current_sum += r_num
            if current_sum == target:
                max_len = max(max_len, right-left+1)
            while current_sum > target:
                l_num = nums[left]
                current_sum -= l_num
                left += 1
        
        return len(nums) - max_len if max_len != 0 else -1
```

#### 提交 660962117 · Wrong Answer · 2025-09-09 13:32:43 CST · python3

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # 问题转化成 查找和为 total_sum - x 的最长子数组
        target = sum(nums) - x
        max_len = -1
        current_sum = 0
        left = 0

        for right, r_num in enumerate(nums):
            current_sum += r_num
            while current_sum > target:
                l_num = nums[left]
                current_sum -= l_num
                left += 1
            if current_sum == target:
                max_len = max(max_len, right-left+1)
        
        return len(nums) - max_len if max_len != 0 else -1
```

#### 提交 660962242 · Runtime Error · 2025-09-09 13:33:15 CST · python3

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        # 问题转化成 查找和为 total_sum - x 的最长子数组
        target = sum(nums) - x
        max_len = -1
        current_sum = 0
        left = 0

        for right, r_num in enumerate(nums):
            current_sum += r_num
            while current_sum > target:
                l_num = nums[left]
                current_sum -= l_num
                left += 1
            if current_sum == target:
                max_len = max(max_len, right-left+1)
        
        return len(nums) - max_len if max_len != -1 else -1
```


## 字符串的排列 (`permutation-in-string`)

- 题目链接：https://leetcode.cn/problems/permutation-in-string/
- 难度：Medium
- 标签：哈希表, 双指针, 字符串, 滑动窗口
- 总提交次数：6
- 最近提交时间：2025-09-09 08:31:19 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 650625014 | 2025-08-07 09:27:20 CST | python3 | Accepted | 27 ms | 17.6 MB |
| 660782710 | 2025-09-08 21:27:10 CST | python3 | Wrong Answer | N/A | N/A |
| 660789212 | 2025-09-08 21:42:02 CST | python3 | Wrong Answer | N/A | N/A |
| 660793399 | 2025-09-08 21:51:50 CST | python3 | Wrong Answer | N/A | N/A |
| 660797800 | 2025-09-08 22:02:13 CST | python3 | Accepted | 27 ms | 17.9 MB |
| 660867188 | 2025-09-09 08:31:19 CST | python3 | Accepted | 23 ms | 17.9 MB |

### 未通过提交代码
#### 提交 660782710 · Wrong Answer · 2025-09-08 21:27:10 CST · python3

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        need = collections.Counter(s1)
        window = collections.defaultdict(int)
        left = valid = 0

        for right, r_char in enumerate(s2):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            if len(window) == len(s1) and valid == len(need):
                return True
            l_char = s2[left]
            if l_char in need:
                if window[l_char] == need[l_char]:
                    valid -= 1
            left += 1
        return False
```

#### 提交 660789212 · Wrong Answer · 2025-09-08 21:42:02 CST · python3

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        need = collections.Counter(s1)
        window = collections.defaultdict(int)
        left = valid = 0

        for right, r_char in enumerate(s2):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            if valid == len(need):
                return True
            if right - left + 1 > len(s1):
                l_char = s2[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                left += 1
        return False
```

#### 提交 660793399 · Wrong Answer · 2025-09-08 21:51:50 CST · python3

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        need = collections.Counter(s1)
        window = collections.defaultdict(int)
        left = valid = 0

        for right, r_char in enumerate(s2):
            if r_char in need:
                window[r_char] += 1
                if window[r_char] == need[r_char]:
                    valid += 1
            if right - left + 1 == len(s1) and valid == len(need):
                return True
            if right - left + 1 > len(s1):
                l_char = s2[left]
                if l_char in need:
                    if window[l_char] == need[l_char]:
                        valid -= 1
                    window[l_char] -= 1
                left += 1
        return False
```


## 水果成篮 (`fruit-into-baskets`)

- 题目链接：https://leetcode.cn/problems/fruit-into-baskets/
- 难度：Medium
- 标签：数组, 哈希表, 滑动窗口
- 总提交次数：1
- 最近提交时间：2025-09-08 11:48:39 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 660572954 | 2025-09-08 11:48:39 CST | python3 | Accepted | 212 ms | 23.2 MB |

### 未通过提交代码
(所有提交均已通过)

## 子数组最大平均数 I (`maximum-average-subarray-i`)

- 题目链接：https://leetcode.cn/problems/maximum-average-subarray-i/
- 难度：Easy
- 标签：数组, 滑动窗口
- 总提交次数：7
- 最近提交时间：2025-09-01 21:44:01 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 658393270 | 2025-09-01 14:21:45 CST | python3 | Wrong Answer | N/A | N/A |
| 658393766 | 2025-09-01 14:23:13 CST | python3 | Wrong Answer | N/A | N/A |
| 658396244 | 2025-09-01 14:30:06 CST | python3 | Accepted | 165 ms | 27 MB |
| 658404145 | 2025-09-01 14:49:24 CST | python3 | Accepted | 124 ms | 26.9 MB |
| 658559348 | 2025-09-01 21:39:48 CST | python3 | Wrong Answer | N/A | N/A |
| 658560019 | 2025-09-01 21:41:31 CST | python3 | Wrong Answer | N/A | N/A |
| 658560994 | 2025-09-01 21:44:01 CST | python3 | Accepted | 176 ms | 27 MB |

### 未通过提交代码
#### 提交 658393270 · Wrong Answer · 2025-09-01 14:21:45 CST · python3

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        max_avg = -float('inf')
        window_sum = 0
        left = 0
        for right, r_num in enumerate(nums):
            window_sum += r_num
            if right - left + 1 == k:
                max_avg = max(window_sum / k, max_avg)
            if right - left + 1 > k:
                window_sum - nums[left]
                left += 1

        return max_avg if max_avg != -float('inf') else 0
```

#### 提交 658393766 · Wrong Answer · 2025-09-01 14:23:13 CST · python3

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        max_avg = -float('inf')
        window_sum = 0
        left = 0
        for right, r_num in enumerate(nums):
            window_sum += r_num
            if right - left + 1 == k:
                max_avg = max(window_sum / k, max_avg)
            if right - left + 1 > k:
                window_sum -= nums[left]
                left += 1

        return max_avg if max_avg != -float('inf') else 0
```

#### 提交 658559348 · Wrong Answer · 2025-09-01 21:39:48 CST · python3

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        left = 0
        max_avg = -float('inf')
        window_sum = 0
        for right, r_num in enumerate(nums):
            window_sum += r_num
            if right > k-1:
                max_avg = max(window_sum / k, max_avg)
                left += 1
        return max_avg if max_avg != -float('inf') else 0
```

#### 提交 658560019 · Wrong Answer · 2025-09-01 21:41:31 CST · python3

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        left = 0
        max_avg = -float('inf')
        window_sum = 0
        for right, r_num in enumerate(nums):
            window_sum += r_num
            if right > k-1:
                max_avg = max(window_sum / k, max_avg)
                window_sum -= nums[left]
                left += 1
        return max_avg if max_avg != -float('inf') else 0
```


## 移除元素 (`remove-element`)

- 题目链接：https://leetcode.cn/problems/remove-element/
- 难度：Easy
- 标签：数组, 双指针
- 总提交次数：1
- 最近提交时间：2025-08-04 09:12:04 CST

### 题目笔记
(暂无笔记)

### 提交记录
| 提交ID | 提交时间 (UTC+08) | 语言 | 结果 | 耗时 | 内存 |
| --- | --- | --- | --- | --- | --- |
| 649591060 | 2025-08-04 09:12:04 CST | python3 | Accepted | 0 ms | 17.5 MB |

### 未通过提交代码
(所有提交均已通过)
