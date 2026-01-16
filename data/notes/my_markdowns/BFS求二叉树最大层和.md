# BFS求二叉树最大层和

## 🧑‍💻 User

用 BFS 解决力扣1161. 最大层内元素和

---
## 🤖 Assistant

题目：给你一棵二叉树，求「节点值之和最大的那一层的层号」，层号从 1 开始。如果有多层和一样大，返回层号最小的那一层。

---

## 一、为什么用 BFS（层序遍历）是最自然的解法？

需求：**按“层”统计每一层的和**。

- BFS 的本质：用队列，一层一层往外“扩散”；
- 我们在处理每一层时，队列里刚好只装着这一层的所有节点；
- 只要：
 1. 记录当前层的节点个数 `level_size`；
 2. 从队列里弹出这 `level_size` 个节点，把 `val` 累加；
 3. 把它们的左右孩子入队（就是下一层）；
- 这样就可以**自然地一层一层算出每层的和**，同时顺手维护“最大和”和“对应层号”。

时间复杂度：
- 每个节点只进队、出队一次：**O(n)**

空间复杂度：
- 队列里最多存一层的节点：**O(width)**，最坏 O(n)

---

## 二、面试用的 30 秒口述思路（可直接背）

> 用队列做层序遍历。从根节点开始入队，维护当前层号 `level`，当前层和 `level_sum`，以及全局最大和 `max_sum` 和答案层 `ans_level`。
> 每一轮 while 里，先取当前队列长度 `size`，表示这一层有多少个节点，然后循环 `size` 次把节点依次出队，累加它们的 `val`，同时把非空的左右孩子入队。
> 一层处理完后，用 `level_sum` 和 `max_sum` 比较，如果更大，就更新 `max_sum` 和 `ans_level`。然后层号加一，继续下一层。
> 最后返回 `ans_level`。整个过程每个节点只访问一次，时间 O(n)，空间是队列的大小 O(width)。

---

## 三、代码实现（BFS，Python，含中文注释）

```python
from collections import deque
from typing import Optional


# LeetCode 已经给出的二叉树节点定义
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def maxLevelSum(self, root: Optional["TreeNode"]) -> int:
        """返回二叉树中，节点值之和最大的层号（层号从 1 开始）。

        使用 BFS 层序遍历，一层一层计算每层的节点和，并记录最大和及对应层号。

        Args:
            root: 二叉树根节点。

        Returns:
            最大层内元素和对应的层号（从 1 开始编号）。
        """
        if root is None:
            # 按题意一般不会为 None，这里防御性返回 0 或者 0 层
            return 0

        # 队列用于 BFS 层序遍历
        queue = deque([root])

        # 当前层号，从 1 开始
        level = 1
        # 记录最大层和及其层号
        max_sum = float("-inf")
        max_level = 1

        # 当队列非空时，一直处理
        while queue:
            # 当前这一层的节点个数
            level_size = len(queue)
            # 当前层的节点值之和
            level_sum = 0

            # 处理当前层的所有节点
            for _ in range(level_size):
                node = queue.popleft()
                level_sum += node.val

                # 将下一层的左右孩子节点加入队列
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)

            # 用当前层的和去更新最大值
            if level_sum > max_sum:
                max_sum = level_sum
                max_level = level

            # 准备处理下一层
            level += 1

        return max_level
```

---

## 四、面试中可能追加的问题（简单准备）

- Q：为什么 BFS 比 DFS 更自然？
A：因为题目按“层”统计，BFS 本身就是层序遍历；BFS 用队列，天然按层处理，统计每层的和很直接。DFS 也能做，但需要在递归里额外记录深度并用数组按深度累加，相对绕一点。

- Q：复杂度是多少？
A：每个节点只进队出队一次，时间 O(n)；队列最多存一层的节点，空间 O(width)，最坏情况 O(n)。
