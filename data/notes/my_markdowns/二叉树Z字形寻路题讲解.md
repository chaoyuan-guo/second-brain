# 二叉树Z字形寻路题讲解

## 🧑‍💻 User

讲解力扣题目1104. 二叉树寻路

---
## 🤖 Assistant

题目：Leet扣 1104. 二叉树寻路

---

## 一、题目本质

- 有一棵**完全二叉树**，第 1 层只有节点 `1`。
- 每一层节点数是 2^k 个（完美二叉树），**编号是连续的**。
- 编号方式是“**之字形**”：
 - 奇数层：从左到右 递增（正常）
 - 偶数层：从右到左 递增（反着）
- 给你一个节点的 `label`，要求返回：**从根节点 1 到该节点的路径（包含根和该节点）**。

---

## 二、关键观察 & 为什么这么做

1. 完全二叉树（正常编号）里，如果一个节点的编号是 `x`，
那么它父节点的编号就是 `x // 2` —— 这是最简单的规律。

2. 难点在于：现在每层是“之字形”编号，**父节点不再是简单的 `label // 2`**。
但是：
> 同一层的“正常顺序编号”和“之字形编号”其实只是**左右翻转（镜像）**的关系。

3. 对任意一层（假设是第 k 层，**0-based level = floor(log2(label))**）：
 - 这一层的编号范围是：
 `start = 2^level`
 `end = 2^(level+1) - 1`
 - 如果这一层是正常从左到右的编号，则为 `[start, ..., end]`
 - 如果这一层是从右到左编号，则为 `[end, ..., start]`
 - 这两个序列是**左右对称**的，所以：
 - 当前层一个节点的“正常编号”与“之字形编号”之间的关系是：
```text
       normal = start + end - zigzag
       ```

4. 在正常编号的完全二叉树里：
   - 子节点 normal 编号 = `normal_child`
   - 父节点 normal 编号 = `normal_child // 2`

5. 把这三步串起来（不用死记推导，记结论就行）：
   - 当前层：`normal_child = start + end - zigzag_child`
   - 父层 normal：`normal_parent = normal_child // 2`
   - 再把 `normal_parent` 映射为上一层的 zigzag 编号，  
     可以合并成一个更简单的公式：
     ```text
     parent = (start + end - zigzag_child) // 2
     ```
   - 这个 `parent` 就是**上一层的 zigzag 编号**，可以直接作为下一轮的 `label` 使用。

6. 所以整体策略：
   - 从给定 `label` 一路往上跳父亲，直到 1。
   - 每次用当前层的 `[start, end]` 和上面的公式算父亲。
   - 把所有经过的节点记录下来，最后反转（因为是从下往上算的）。

时间复杂度：  
树的高度是 `O(log label)`，每层算一次，整体 `O(log label)`。  
空间复杂度：路径长度也是 `O(log label)`。

---

## 三、解题步骤（面试中可说的逻辑）

1. 先算出当前 `label` 所在层的层号（用 `level = int(log2(label))`）。
2. 这一层的编号区间是 `[start, end] = [2^level, 2^(level+1)-1]`。
3. 因为这一层是 zigzag 编号，我们先把它映射回“正常编号”的对称点，再除以 2 得到父亲；  
   可以合并成：  
   `parent = (start + end - label) // 2`。
4. 循环：从目标 `label` 一直用上面的公式跳到 1，每次把节点加入数组。
5. 最后把数组反转，就是从根到目标节点的路径。

---

## 四、Python 代码（可直接背）

```python
import math
from typing import List


class Solution:
 def pathInZigZagTree(self, label: int) -> List[int]:
 """返回从根节点到给定节点 label 的路径（之字形标号的完全二叉树）。

 思路：
 - 从 label 一路往上找父节点直到根节点 1。
 - 在每一层，利用当前层的区间 [start, end]，通过
 parent = (start + end - label) // 2
 计算上一层的父节点（该父节点已经是 zigzag 编号）。
 - 将路径从下到上记录，最后反转即可。
 """
 path = []
 while label >= 1:
 path.append(label)

 # 当前层的 0-based level，例如：label 在 [2^level, 2^(level+1)-1]
 level = int(math.log2(label))
 level_start = 1 << level # 2 ** level
 level_end = (1 << (level + 1)) - 1 # 2 ** (level + 1) - 1

 # 根据推导公式，直接算上一层父节点的 zigzag 编号
 label = (level_start + level_end - label) // 2

 # 我们是从目标节点一直往上走的，所以需要反转
 return path[::-1]
```

---

## 五、面试中 30 秒口述要点（可直接复述）

- 这是一个完全二叉树，只是每一层的标号是之字形的。
- 在正常完全二叉树里，编号为 x 的节点父亲是 x // 2；
较难的是如何从之字形编号算父亲。
- 对每一层来说，它的编号区间是 `[2^level, 2^(level+1)-1]`，之字形和正常编号只是在这一段内做了一次左右翻转，因此：
当前 label 对应的“正常编号”是 `start + end - label`。
- 正常编号的父亲是 `normal // 2`，结合翻转关系，可以得到一个统一公式：
`parent = (start + end - label) // 2`。
- 所以我从目标 label 开始，循环：
计算当前层的 start/end，用这个公式跳到父亲，直到 1，
把路径记录下来再反转即可，时间复杂度是 `O(log label)`。

---
## 🤖 Assistant

好的，没问题。我们开始吧。

作为一名经验丰富的工程师，你的优势在于强大的工程能力和逻辑思维，弱点是对于特定算法题型的“套路”不熟。我的目标就是帮你快速识别题型，掌握核心“套路”，并能清晰地表达出来。

我们今天来看一道非常经典、面试中也可能遇到的题目：**力扣 1104. 二叉树寻路**。

这道题属于中等难度，但它考察的不是复杂的算法，而是你对**问题规律的洞察能力**。

---

### 1. 问题分析：这道题到底在考什么？

首先看题，它描述了一棵“Z字形”编号的满二叉树。

* 标准的满二叉树：每一层的节点都是从左到右依次编号。它的父子关系非常简单：`父节点 = 子节点 // 2`。
* 这道题的树：奇数层（1, 3, 5...）从左到右编号，偶数层（2, 4, 6...）从右到左编号。

![Z字形二叉树](https://assets.leetcode.com/uploads/2019/06/24/tree.png)

**核心矛盾**：由于偶数层的编号是反的，`父节点 = 子节点 // 2` 这个公式就失效了。

**我们的任务**：给定一个节点的编号 `label`，找出从根节点 `1` 到 `label` 的路径。

### 2. 核心思路：找到不变的规律

解决这类路径问题，最直接的方法就是**从目标节点反向往回走到根节点**。因为每个节点的父节点是唯一的，这条路径也是唯一的。

既然 `parent = child // 2` 不好使了，我们就必须找到一个新的、能正确计算出父节点编号的方法。

**突破口在哪？**

让我们观察一下这棵树。虽然**编号**是乱的，但树的**物理结构**和标准的满二叉树是一模一样的。这意味着，一个节点在树上的**位置**是固定的。

关键点来了：**对称性**。

在任何一层，我们看看一个节点 `label` 和它在物理位置上“对称”的那个节点有什么关系。

* **第4层**：节点范围是 `[8, 15]`。
 * `15` 和 `8` 是对称的，和为 `23`。
 * `14` 和 `9` 是对称的，和为 `23`。
 * `13` 和 `10` 是对称的，和为 `23`。
* **第3层**：节点范围是 `[4, 7]`。
 * `4` 和 `7` 是对称的，和为 `11`。
 * `5` 和 `6` 是对称的，和为 `11`。
* **第2层**：节点范围是 `[2, 3]`。
 * `2` 和 `3` 是对称的，和为 `5`。

**规律找到了**：在同一层中，任意一个节点 `label` 和它“对称位置”的节点 `symmetric_label` 的值加起来是一个**恒定的值**。这个值等于**该层的最大编号 + 该层的最小编号**。

即：`label + symmetric_label = level_min + level_max`

**如何利用这个规律找爸爸？**

1. 假设我们当前在节点 `label`。
2. 我们先找到它对称位置的节点值：`symmetric_label = level_min + level_max - label`。
3. **最关键的一步**：`label` 这个节点的父节点，在**物理位置**上，其实就是 `symmetric_label` 这个节点在**标准满二叉树**中的父节点！
4. 在标准满二叉树中找父节点就太简单了，直接 `parent = symmetric_label // 2`。

所以，我们推导出了**核心公式**：
`parent_label = (level_min + level_max - current_label) // 2`

有了这个公式，我们就可以从 `label` 一路反推到 `1` 了。

### 3. 步骤拆解

1. **初始化**：创建一个空列表 `path` 用于存放路径。
2. **循环**：从给定的 `label` 开始，只要它不等于 `1`，就一直循环。
 a. **记录当前节点**：将当前的 `label` 加入 `path` 列表。
 b. **确定当前层级**：如何快速确定一个数 `n` 在满二叉树的第几层？用对数 `log2(n)` 向下取整再加1。在 Python 中，有个更简洁高效的方法：`n.bit_length()`，一个数的二进制表示的位数刚好就是它所在的层级。例如 `14` 的二进制是 `1110`，长度为 `4`，就在第 `4` 层。
 c. **计算层级范围**：知道了层级 `level`，该层的最小值是 `2^(level-1)`，最大值是 `2^level - 1`。
 d. **计算父节点**：应用我们的核心公式 `parent = (level_min + level_max - label) // 2`，并将 `label` 更新为计算出的 `parent` 值。
3. **收尾**：循环结束后，`label` 已经变成了 `1`（根节点）。把 `1` 也加入 `path` 列表。
4. **反转**：因为我们是从后往前找的，所以最后需要把 `path` 列表反转一下，得到从根到目标的正确路径。

### 4. Python 代码实现

```python
import math

def path_in_zigzag_tree(label: int) -> list[int]:
    """
    在一棵Z字形编号的二叉树中，找到从根到指定节点的路径。

    Args:
      label: 目标节点的编号。

    Returns:
      一个整数列表，表示从根节点到目标节点的路径。
    """
    # 1. 确定 label 所在的层级
    # 一个数 n 的二进制位数，就是它在满二叉树中的层级
    level = label.bit_length()
    
    path = []
    
    # 2. 从 label 开始，反向循环直到根节点
    current_label = label
    while level > 0:
        # 将当前节点加入路径
        path.append(current_label)
        
        # 计算当前层的数值范围
        # 使用位运算 1 << n 等价于 2**n，效率更高
        level_min = 1 << (level - 1)
        level_max = (1 << level) - 1
        
        # 核心公式：计算父节点
        # 当前节点的父节点，等于它“对称节点”在标准满二叉树里的父节点
        parent_label = (level_min + level_max - current_label) // 2
        
        # 更新当前节点为父节点，继续向上寻找
        current_label = parent_label
        level -= 1
        
    # 3. 因为是反向添加的，所以需要反转列表
    return path[::-1]

# 示例
print(f"14 的路径是: {path_in_zigzag_tree(14)}")  # 预期输出: [1, 3, 4, 14]
print(f"26 的路径是: {path_in_zigzag_tree(26)}")  # 预期输出: [1, 2, 6, 10, 26]

```

### 5. 面试口述思路要点 (30秒内讲完)

> 面试官你好，对于这道题，我的思路是**从目标节点反向遍历到根节点**来构建路径。
>
> 这棵树的关键在于它的Z字形编号，导致常规的 `父节点 = 子节点 // 2` 失效了。
>
> 为了解决这个问题，我发现了一个**对称性的规律**：在任何一层，一个节点和它物理位置对称的节点的编号之和，等于这层最大值和最小值的和。
>
> 利用这个规律，我们可以推导出父节点的计算公式：**`父节点 = (当前层最小值 + 当前层最大值 - 当前节点) // 2`**。
>
> 所以，我的算法就是从目标节点开始，循环利用这个公式计算出父节点，记录路径，直到根节点1，最后把路径反转即可。这个算法的时间复杂度是 O(logN)，因为树的高度就是 logN。

---
## 🧑‍💻 User

什么叫“当前层的 0-based level”

---
## 🤖 Assistant

“0-based level” 这个说法，其实就是**层号从 0 开始数**，而不是从 1 开始数。

### 1. 直观理解

一棵二叉树的层数通常我们会这么说：

- 根节点那层：第 1 层
- 它的孩子那层：第 2 层
- 再下面：第 3 层
……

这叫 **1-based**（从 1 开始计数）。

所谓 **0-based level** 就是把同样的层改成：

- 根节点那层：level = 0
- 它的孩子那层：level = 1
- 再下面：level = 2
……

也就是：**原来第 k 层 → 现在的 level = k-1**。

### 2. 为什么在这题里用 0-based level

在这题里我们写了类似这样一段：

```python
level = int(math.log2(label))
level_start = 1 << level           # 2 ** level
level_end = (1 << (level + 1)) - 1 # 2 ** (level + 1) - 1
```

这个 `level` 就是 **0-based level**，它有一个很方便的性质：

- level = 0：这一层的 label 范围是 [1, 1]
- level = 1：这一层的 label 范围是 [2, 3]
- level = 2：这一层的 label 范围是 [4, 7]
- level = 3：这一层的 label 范围是 [8, 15]
……

统一公式就是：
**第 level 层的范围是 `[2^level, 2^(level+1) - 1]`**

而 `int(log2(label))` 刚好算出的就是这个 0-based level，
比如：

- label = 1 → log2(1)=0 → level=0
- label = 2 或 3 → log2 在 [1,1.x) → 取整 1 → level=1
- label = 4~7 → log2 在 [2,2.x) → 取整 2 → level=2

所以我说“当前层的 0-based level”，就是指：

> 用从 0 开始计数的层号，表示当前这个节点所在的高度。

仅此而已，只是一个从 0 开始的层索引，并没有别的特别含义。

---
## 🧑‍💻 User

再讲一下力扣题目662. 二叉树最大宽度，要求使用 BFS

---
## 🤖 Assistant

题目：Leet扣 662. 二叉树最大宽度（要求用 BFS）

---

## 一、题目本质

**宽度的定义**（关键）：
某一层的宽度 =
> 这一层最左、最右**非空节点之间**的节点数（**中间的空节点也要算进去**）

所以不能只数这一层有多少个节点，而是要考虑“**空位**”。

本质上，就是把树当成一棵**完全二叉树**来看，给每个节点一个“下标位置”，
然后一层一层 BFS，算每一层“最左下标”和“最右下标”的差。

---

## 二、为什么要给节点编号（位置 index）

完全二叉树里，如果给节点一个 index（0-based）：

- 根节点 index = 0
- 左孩子 index = `parent_index * 2`
- 右孩子 index = `parent_index * 2 + 1`

这样，即使中间有空节点，我们也可以通过 index 看出“跨度”：

例如某层有两个节点，下标是 4 和 7：
那这层宽度 = `7 - 4 + 1 = 4`，说明两者之间“占了 4 个位置”，
哪怕中间只存在 2 个真实节点，也要算成 4。

**所以做法：**

- 用 BFS 按层遍历；
- 队列里存 `(节点指针, index)`；
- 对每一层：
 - 记录这一层最左的 index（first）和最右的 index（last）；
 - 宽度 = `last - first + 1`；
- 所有层宽度取最大值。

---

## 三、index 可能很大怎么办？（为什么要做偏移）

如果树很深，index 会变得特别大（比如 2^30 级别），
虽然 Python 能顶住，但面试中一般会顺手写一个**小优化**：

在每一层开始时，把该层的 index 做一次“平移”，让最左节点变成 0：

- 这一层队列的第一个节点 index = `head_index`
- 这一层其它节点的“相对 index” = `index - head_index`
- 这样这一层 index 都是从 0 开始，数值更小
- 宽度不会变，因为我们统一减去同一个数（差值不变）

---

## 四、BFS 解法步骤（按层）

1. 若 root 为空，返回 0。
2. 用队列保存 `(node, index)`，根结点从 index = 0 开始。
3. while 队列不空：
 - 当前层节点个数 `level_size = len(queue)`；
 - 当前层最左 index = `head_index = queue[^0][^1]`；
 - 遍历这一层的所有节点（循环 level_size 次）：
 - 弹出 `(node, idx)`；
 - 计算当前节点在本层的相对下标：`cur_index = idx - head_index`；
 - 在循环末尾（或用 i == level_size-1 判断最后一个节点）记录 `cur_index`，用来算本层宽度；
 - 把左右孩子入队，index 分别是 `2*cur_index` 和 `2*cur_index + 1`；
 - 本层宽度 = `last_cur_index + 1`（因为这一层最左被平移成 0 了）；
 - 更新全局最大宽度。
4. 返回最大宽度。

时间复杂度：
整棵树每个节点只入队出队一次 → `O(N)`
空间复杂度：
队列最多存一层的所有节点 → `O(N)`（最坏情况）

---

## 五、Python 代码（BFS，可直接背）

```python
from collections import deque
from typing import Optional


class TreeNode:
    """二叉树节点定义。"""
    def __init__(self, val: int = 0,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """返回二叉树的最大宽度（按完全二叉树位置编号计算）。

        使用 BFS 按层遍历，用 index 模拟完全二叉树的下标：
        - 根节点 index = 0
        - 左孩子 index = parent_index * 2
        - 右孩子 index = parent_index * 2 + 1

        对每一层，记录最左和最右节点的 index，宽度为 right - left + 1。
        为避免 index 过大，在每层开始时将 index 平移，使最左节点 index 为 0。
        """
        if not root:
            return 0

        max_width = 0
        queue = deque()
        # 队列中元素为 (节点, index)，根节点 index 从 0 开始
        queue.append((root, 0))

        while queue:
            level_size = len(queue)
            # 当前层最左节点的 index，用于做偏移
            _, head_index = queue[0]

            # 用于记录当前层最后一个节点的相对 index
            last_index = 0

            for i in range(level_size):
                node, idx = queue.popleft()
                # 将当前节点 index 归一到从 0 开始，避免数字过大
                cur_index = idx - head_index
                last_index = cur_index  # 每次更新，最后一次就是本层最右

                if node.left:
                    # 左孩子在完全二叉树中的位置
                    queue.append((node.left, cur_index * 2))
                if node.right:
                    # 右孩子在完全二叉树中的位置
                    queue.append((node.right, cur_index * 2 + 1))

            # 本层宽度 = 最右(相对 index) - 最左(0) + 1 = last_index + 1
            max_width = max(max_width, last_index + 1)

        return max_width
```

---

## 六、面试中 30 秒口述思路（可直接说）

- 题目要求的宽度是：一层中最左和最右非空节点之间的总位置数，中间的空位也要算。
- 我把树看成一棵完全二叉树，用 BFS 按层遍历，同时给节点一个位置下标：
根是 0，左孩子是 2*i，右孩子是 2*i+1。
- 对每一层，我记录队头的下标作为基准，所有节点的下标都减去这个基准，
这样本层最左下标变为 0，最后一个节点的下标就是这一层的宽度减 1。
- 本层宽度就是 `last_index + 1`，在遍历过程中维护一个最大值即可。
- 时间复杂度 O(N)，空间复杂度 O(N)。

---
## 🧑‍💻 User

索引能从 1 开始吗

---
## 🤖 Assistant

可以，索引从 1 开始完全没问题，而且**从 1 开始其实更常见**，因为它和“完全二叉树用数组存储”的套路是一致的。

---

## 1. 两种写法对比（核心点）

只要你满足“父子下标关系一致”，从 0 开始、从 1 开始都可以：

### 写法 A：从 1 开始（更常见）

- 根节点：index = 1
- 左孩子：`left_index = parent_index * 2`
- 右孩子：`right_index = parent_index * 2 + 1`

### 写法 B：从 0 开始（我之前写的）

- 根节点：index = 0
- 左孩子：`left_index = parent_index * 2`
- 右孩子：`right_index = parent_index * 2 + 1`

只要**父子都用同一套规则**，同一层的下标之间的**差值**就是层宽度（包含中间空位），起点是 0 还是 1 不影响结果。

---

## 2. 用 1 开始的 BFS 写法示例

下面是一个“从 1 开始编号”的版本，你可以和之前从 0 开始的版本对比一下：

```python
from collections import deque
from typing import Optional


class TreeNode:
    """二叉树节点定义。"""
    def __init__(self, val: int = 0,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """BFS，根节点下标从 1 开始的版本。"""
        if not root:
            return 0

        max_width = 0
        # 队列中保存 (节点, 完全二叉树下标)，根从 1 开始
        queue = deque([(root, 1)])

        while queue:
            level_size = len(queue)
            # 当前层最左节点的下标
            _, first_index = queue[0]
            last_index = first_index

            for _ in range(level_size):
                node, idx = queue.popleft()
                last_index = idx  # 当前层遍历到的最右节点下标

                if node.left:
                    # 左孩子在完全二叉树中的下标
                    queue.append((node.left, idx * 2))
                if node.right:
                    # 右孩子在完全二叉树中的下标
                    queue.append((node.right, idx * 2 + 1))

            # 当前层宽度 = 最右下标 - 最左下标 + 1
            max_width = max(max_width, last_index - first_index + 1)

        return max_width
```

这版就彻底是“经典的完全二叉树数组下标”写法了。

---

## 3. 面试口述（你可以这么说）

> 索引我可以从 1 开始，用完全二叉树数组下标的方式：根是 1，左孩子 2*i，右孩子 2*i+1。
> BFS 按层遍历，队列里存 (节点，下标)。
> 对于每一层，记录队头和队尾节点的下标，宽度就是 last - first + 1，整个过程中维护最大值。
> 下标从 0 还是从 1 开始都可以，本质是利用下标模拟完全二叉树，空位用下标之间的差值体现出来。

---
## 🧑‍💻 User

你的 for 循环上面写了一行last_index = first_index，然后 for 循环里面又写了一行last_index = idx，这让我看起来有点困惑

---
## 🤖 Assistant

你抓得很细，这里确实容易让人疑惑，我来拆开说一下。

原来的代码是这样的（1-based 版本）：

```python
while queue:
    level_size = len(queue)
    _, first_index = queue[0]
    last_index = first_index      # 这里先赋值一次

    for _ in range(level_size):
        node, idx = queue.popleft()
        last_index = idx          # 这里每次循环又覆盖一次

        if node.left:
            queue.append((node.left, idx * 2))
        if node.right:
            queue.append((node.right, idx * 2 + 1))

    max_width = max(max_width, last_index - first_index + 1)
```

### 1. 这两行的本意是什么？

- `first_index`：当前层**最左节点**的下标（队头）。
- `last_index`：当前层**最右节点**的下标。

逻辑是：

- 一开始先假设“当前层最右的 = 最左的”，所以写了
`last_index = first_index`（只是一个初始化）。
- 在 `for` 里按顺序把这一层的节点 pop 出来：
 - 每 pop 一个，就把 `last_index = idx`，
 - 最后一次 pop 的 `idx`，就是这一层**最右节点**的下标。
- 循环结束后，`last_index` 就是这一层真正的最右下标。

所以这两行代表的是：

- 上面那行：先给 `last_index` 一个初始值，防止变量未定义；
- 下面那行：实际在遍历时不断更新，**最后的值才是我们要的**。

你可以把它理解为：

```python
last_index = first_index  # 占个位，下面循环一定会覆盖
for ...:
    ...
    last_index = idx      # 最终以最后一个节点的 idx 为准
```

这在逻辑上是没问题的，但**从可读性上确实有点迷惑、显得多余**。

---

### 2. 更清晰的写法（推荐你记这个）

你可以改成“只在最后一个节点时赋值”，看起来更直观：

```python
while queue:
    level_size = len(queue)
    _, first_index = queue[0]

    for i in range(level_size):
        node, idx = queue.popleft()

        if i == level_size - 1:
            # 当前层最后一个节点的下标
            last_index = idx

        if node.left:
            queue.append((node.left, idx * 2))
        if node.right:
            queue.append((node.right, idx * 2 + 1))

    max_width = max(max_width, last_index - first_index + 1)
```

这样逻辑完全等价，而且语义非常清晰：

- `first_index`：队头的 index。
- `last_index`：这一层遍历到的**最后一个节点**的 index（用 `i == level_size - 1` 判断）。

---

### 3. 甚至可以不用在循环里维护 last_index

还有一种更“干净”的写法：在进入这一层前，队列里**只包含这一层的节点**，
所以队头、队尾就是这一层的最左、最右：

```python
while queue:
    level_size = len(queue)
    _, first_index = queue[0]
    _, last_index = queue[-1]     # 注意：在本层入队孩子之前取

    for _ in range(level_size):
        node, idx = queue.popleft()

        if node.left:
            queue.append((node.left, idx * 2))
        if node.right:
            queue.append((node.right, idx * 2 + 1))

    max_width = max(max_width, last_index - first_index + 1)
```

这里的关键点是：
`level_size = len(queue)` 这行已经把当前层大小固定住了，
后面虽然会往队列里 append 孩子，但这一层的节点先被全部 pop 掉，
不会影响我们在循环前从 `queue[^0]`、`queue[-1]` 取到的 first / last。

---

### 4. 面试时你可以怎么说（关于这个细节）

> 我在 BFS 每一层时，只关心这一层最左节点和最右节点的 index。
> 代码里我会在进入层循环前，从队列的第一个和最后一个元素拿到 first_index 和 last_index，
> 最后宽度就是 last_index - first_index + 1。
> 具体是在循环里更新 last_index，还是一开始直接从 queue[-1] 拿，都可以，核心思想是一样的。

总结一句：
你看到的那种“先设成 first_index 再在循环里覆盖”的写法，**只是为了初始化变量，不是算法必须**。
可以按上面推荐的两种重写方式改得更清晰。

---
## 🧑‍💻 User

按最后的这种写法重新整理下 30 秒的口述思路

---
## 🤖 Assistant

- 这题宽度的定义是：一层中最左和最右非空节点之间的所有位置个数，中间缺的也要算，所以要用“位置索引”来模拟完全二叉树。
- 我用 BFS 按层遍历，队列里存 `(节点, 下标)`，下标按完全二叉树规则：根是 1，左子 2*i，右子 2*i+1。这样即使中间有空位，也会体现在下标差里。
- 每一层开始时，当前层的所有节点都在队列中，我先取
`first_index = queue[^0][^1]`，`last_index = queue[-1][^1]`，
这一层的宽度就是 `last_index - first_index + 1`，用它更新全局最大值。
- 然后再用 `level_size = len(queue)` 限定这一层的节点个数，循环 `level_size` 次：每次 pop 一个节点，用它的下标生成左右孩子的下标，再把孩子入队。
- 整体是标准 BFS，一次遍历所有节点，时间复杂度 O(N)，空间 O(N)，关键点就是“用完全二叉树下标模拟空位，再用当前层队头和队尾下标之差当宽度”。
