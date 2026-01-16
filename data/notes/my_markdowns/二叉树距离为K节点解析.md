# 二叉树距离为K节点解析

## 🧑‍💻 User

讲解力扣题目863. 二叉树中所有距离为 K 的结点

---
## 🤖 Assistant

好的，没问题！作为你的“算法速成教练”，我们现在开始。你的背景非常典型，很多资深工程师在职业发展后期都会遇到算法这道坎。别担心，我的目标就是帮你用最少的时间、最直接的方式搞定它，让你在面试中游刃有余。

我们今天来攻克 **LeetCode 863. 二叉树中所有距离为 K 的结点**。这道题非常经典，因为它考察了树、图、BFS（广度优先搜索）这几个核心知识点的结合。

---

### 题目解析：863. 二叉树中所有距离为 K 的结点

给定一个二叉树的根节点 `root`，一个目标节点 `target`，和一个整数 `K`。返回所有与 `target` 距离为 `K` 的节点的值的列表。

**核心难点是什么？**

二叉树的指针是单向的，我们只能从父节点访问子节点（`node.left`, `node.right`）。但是“距离”是双向的，从 `target` 节点出发，我们不仅要能**向下**走到它的子孙节点，还要能**向上**走到它的父节点，再从父节点拐到另一个子树去。

**怎么解决这个“向上走”的问题？**

直接的想法就是：**把树变成图**。

* **树**：有方向、有层级、无环。
* **图**：节点之间可以任意连接，没有方向限制。

一旦我们把树的结构转换成图，那从 `target` 节点出发，到任何节点的距离问题，就变成了一个经典的图遍历问题。而求图中“距离为K”的所有点，**广度优先搜索（BFS）** 是最完美的工具。

所以，我们的解题思路就清晰地分成了两步：

1. **建图**：遍历整棵树，构建一个图的表示。这个图需要能体现出节点之间的双向连接关系。最常用的方式是使用**哈希表（字典）** 来存储每个节点的邻接点。
2. **搜索**：从 `target` 节点开始，在新建的图上进行广度优先搜索（BFS），找到所有距离为 `K` 的节点。

---

### 步骤拆解与代码实现

#### **第一步：建图 (将 Tree 转换为 Graph)**

我们需要一个数据结构来存储图。Python 的字典（哈希表）非常适合，key 是节点本身，value 是一个列表，存放所有与它相邻的节点。

我们通过一次树的遍历（深度优先 DFS 或者广度优先 BFS 都可以）来填充这个字典。对于每个父子关系 `parent -> child`，我们都要添加两条边：

1. `parent` 的邻接表里加上 `child`。
2. `child` 的邻接表里加上 `parent`。（**这就是实现“向上走”的关键**）

```python
# 假设我们有一个建图函数
def build_graph(root):
    # 使用字典来表示邻接表
    adj_list = collections.defaultdict(list)
    
    # 使用栈进行深度优先遍历来构建图
    stack = [root]
    
    while stack:
        node = stack.pop()
        if node:
            if node.left:
                # 添加双向边
                adj_list[node].append(node.left)
                adj_list[node.left].append(node)
                stack.append(node.left)
            if node.right:
                # 添加双向边
                adj_list[node].append(node.right)
                adj_list[node.right].append(node)
                stack.append(node.right)
    return adj_list
```

#### **第二步：广度优先搜索 (BFS)**

BFS 的特性就是一层一层地向外扩展。从 `target` 开始，第一层就是距离为 1 的节点，第二层就是距离为 2 的节点，...，第 `K` 层就是我们想要的答案。

我们需要：
* 一个 `queue`（队列）：存放 `(待访问节点, 当前距离)`。
* 一个 `visited`（集合）：记录已经访问过的节点，防止走回头路，陷入死循环。

```python
# 假设我们已经有了邻接表 adj_list 和目标 target, K
def bfs_search(adj_list, target, K):
    # 初始化队列，放入起始节点和初始距离0
    queue = collections.deque([(target, 0)])
    # 初始化已访问集合，放入起始节点
    visited = {target}
    # 结果列表
    result = []
    
    while queue:
        # 从队列头部取出一个节点和它对应的距离
        current_node, current_dist = queue.popleft()
        
        # 如果当前距离等于K，说明找到了一个答案
        if current_dist == K:
            result.append(current_node.val)
            # 找到一层后可以继续找同层的，但没必要再往深了
            continue

        # 如果距离已经大于K，后面的也不可能满足，可以剪枝（但在这里不是必须）
        if current_dist > K:
            break

        # 遍历当前节点的所有邻居
        if current_node in adj_list:
            for neighbor in adj_list[current_node]:
                # 如果邻居没被访问过
                if neighbor not in visited:
                    # 标记为已访问
                    visited.add(neighbor)
                    # 将邻居和更新后的距离放入队列
                    queue.append((neighbor, current_dist + 1))
                    
    return result
```

#### **完整代码**

遵循 Google Python Style Guide，我们将代码整合到一个类中。

```python
import collections

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    """
    解决二叉树中所有距离为 K 的结点问题。
    """

    def distanceK(self, root: 'TreeNode', target: 'TreeNode', k: int) -> list[int]:
        """
        查找所有与 target 节点距离为 k 的节点值。

        该方法分为两步：
        1. 将二叉树转换为一个无向图（使用邻接表表示），以便于双向遍历。
        2. 从 target 节点开始，使用广度优先搜索（BFS）找到所有距离为 k 的节点。

        Args:
            root: 二叉树的根节点。
            target: 目标节点。
            k: 目标距离。

        Returns:
            一个包含所有符合条件的节点值的列表。
        """
        if not root:
            return []

        # 1. 建图：使用 DFS 将树转换为邻接表表示的无向图
        adj_list = collections.defaultdict(list)
        
        # 使用栈进行深度优先遍历
        stack = [(root, None)] # (当前节点, 父节点)
        while stack:
            node, parent = stack.pop()
            if parent:
                # 添加双向边
                adj_list[node].append(parent)
                adj_list[parent].append(node)
            if node.left:
                stack.append((node.left, node))
            if node.right:
                stack.append((node.right, node))
        
        # 2. 搜索：从 target 节点开始进行 BFS
        # 队列中存放元组 (节点, 当前距离)
        queue = collections.deque([(target, 0)])
        # 集合用于记录已访问过的节点，防止重复访问
        visited = {target}
        result = []

        while queue:
            current_node, current_dist = queue.popleft()

            # 如果当前距离等于 k，则将节点值加入结果列表
            if current_dist == k:
                result.append(current_node.val)
                continue  # 继续处理队列中同层级的其他节点

            # 如果距离已超过k，无需再向外扩展
            if current_dist > k:
                break
                
            # 遍历所有邻居节点
            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_dist + 1))
        
        return result

```
*代码说明：建图部分我用了一个更简洁的 DFS 写法，通过 `(node, parent)` 元组来追踪父节点，从而建立双向连接。效果是一样的。*

---

### 为什么要这样做（Why & How）

* **为什么要转换成图？**
 * **本质：** 这是为了解决“逆向/向上”移动的问题。树的原生结构限制了我们只能向下。通过建立一个包含父节点信息的邻接表，我们实际上是创建了一个通用的图结构，赋予了每个节点“向上走”到父节点的能力。
 * **降维打击：** 这是一种“转化思想”，把一个在特定数据结构（树）上很难处理的问题，转化到一个更通用、有现成经典算法（图的BFS）可以解决的领域。

* **为什么用 BFS 而不是 DFS？**
 * **BFS 的特性：** BFS 按“层”或“距离”进行遍历。它从起点开始，先访问完所有距离为 1 的节点，然后是所有距离为 2 的节点，以此类推。
 * **完美匹配：** 这个特性与我们的问题“找到所有距离为 K 的节点”完美契合。BFS 遍历到第 K 层时，该层的所有节点就是我们的答案。DFS 是“一条路走到黑”，它找到的第一个节点可能距离很远，不适合按距离查找。

### 面试口述思路要点（30秒版本）

> 好的面试官。这道题的核心挑战在于，我们需要在树上进行双向的距离查找，但树的结构本身只支持从父到子的单向遍历。
>
> 我的思路是分两步走：
>
> **第一，转化问题**：我会先遍历一遍整棵树，构建一个**无向图**。我用一个哈希表来存储每个节点的邻接点，这样对于任意一个节点，我既能找到它的子节点，也能找到它的父节点。
>
> **第二，解决问题**：有了这个图之后，问题就变成了“从 target 节点出发，找到所有距离为 K 的节点”。这是一个典型的**广度优先搜索（BFS）** 的应用场景。我从 target 节点开始进行 BFS，当搜索到第 K 层时，那一层的所有节点就是最终的答案。

这个口述思路清晰、自信，并且准确地使用了“无向图”、“哈希表”、“广度优先搜索/BFS”这些专业术语，能给面试官留下很好的印象。

**练习建议**：
1. **理解**：确保你真的理解了为什么要“树转图”，以及为什么 BFS 在这里是最佳选择。
2. **默写**：关掉讲解，自己在 LeetCode 上把这道题的完整代码写出来。写不出来就回来看，直到能独立、流畅地写出来为止。
3. **举一反三**：思考一下，如果 K 值很大，超过了树的高度，你的代码会返回什么？（空列表，是正确的）。如果 target 节点就是根节点，代码是否依然有效？（是的）。

这道题搞懂了，你对树的遍历、图的思维、BFS 算法的理解就能上一个台阶，达到六七十分的目标又近了一步！加油！

---
## 🤖 Assistant

题目链接：Leet扣 863. 二叉树中所有距离为 K 的结点

---

## 1. 题目理解（先用“图”的视角看）

给定一棵二叉树，一个目标结点 `target`，以及一个整数 `K`，
要求：返回**所有与 `target` 的距离恰好为 K 的结点的值**。

这里的“距离”是**边的条数**：
- 相邻父子之间距离是 1
- 你可以从子到父，也可以从父到子（本质上就是一张无向图）

因此，这个问题可以理解成：
> 把二叉树看成一个无向图，从结点 `target` 出发，找所有距离为 K 的结点。

---

## 2. 核心思路（把树变成“能往上走”的结构，然后 BFS）

### 难点本质

普通二叉树只有 `left` / `right`，**没有父指针**，
但“距离”允许往父节点走，所以必须能从一个节点**走到它的父节点**。

### 核心做法

1. **先 DFS 一遍树，记录每个节点的父节点**：
 - 用一个字典 `parent_map`，键是节点对象，值是它的父节点；根节点父亲为 `None`
 - 这样我们就可以从任意节点，向三个方向走：`left`、`right`、`parent`

2. **然后从 `target` 做 BFS（按层遍历）**：
 - 队列元素为 `(node, dist)`，表示当前节点和它到 target 的距离
 - 从 `target` 出发，初始距离为 0
 - 每一层扩展到：左孩子、右孩子、父节点，前提是没有访问过
 - 当 `dist == K` 时，队列里当前层所有节点就是答案，收集它们的 `val` 即可

3. 用一个 `visited` 集合防止死循环（因为现在有“父边”，图是无向的）

### 时间 / 空间复杂度

- 建父指针 DFS：`O(N)`
- BFS：最多访问每个节点一次，也是 `O(N)`
- 总时间复杂度：**O(N)**
- 辅助空间：父指针字典 + 访问集合 + 队列，**O(N)**

这个解法逻辑清晰，非常适合面试直接说。

---

## 3. 面试口述思路（30 秒版本）

> 这题本质是从目标节点出发，在一张无向图里找距离为 K 的所有节点。
> 二叉树本身只能往下走，但距离允许往父节点走，所以我第一步用 DFS 遍历整棵树，给每个节点建立一个父指针映射。这样从一个节点可以走到它的左孩子、右孩子和父节点。
> 第二步从 target 节点开始做 BFS，队列里存当前节点和它到 target 的距离，用一个 visited 集合避免重复访问。当 BFS 扩展到距离为 K 的那一层时，这一层所有节点的值就是答案，直接返回即可。整体时间复杂度 O(N)，空间 O(N)。

---

## 4. 代码实现（Python，符合 Google Style，带中文注释）

```python
# Definition for a binary tree node.
# LeetCode 会预先定义这个类，这里仅供参考。
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections


class Solution(object):
    def distanceK(self, root, target, k):
        """返回二叉树中所有与目标节点距离为 k 的节点值。

        思路：
            1）先 DFS 整棵树，记录每个节点的父节点，构造“父指针”关系；
            2）再从 target 节点出发做 BFS，按层扩展到距离 k；
            3）距离为 k 的那一层节点值即为答案。

        Args:
            root: TreeNode，二叉树根节点。
            target: TreeNode，目标节点。
            k: int，要求的距离。

        Returns:
            List[int]，所有与 target 距离为 k 的节点值。
        """
        if not root:
            return []

        # 1. DFS 建立每个节点到其父节点的映射
        parent_map = {}

        def dfs(node, parent):
            """深度优先遍历，记录节点的父亲。

            Args:
                node: 当前遍历到的节点。
                parent: 当前节点的父节点。
            """
            if not node:
                return
            parent_map[node] = parent
            dfs(node.left, node)
            dfs(node.right, node)

        dfs(root, None)

        # 2. 从 target 出发做 BFS，找到距离为 k 的所有节点
        queue = collections.deque()
        queue.append((target, 0))  # (当前节点, 与 target 的距离)
        visited = set()
        visited.add(target)

        result = []

        while queue:
            node, dist = queue.popleft()

            # 如果当前距离已经等于 k，收集这一层以及队列中同层剩余的节点
            if dist == k:
                # 当前节点
                result.append(node.val)
                # 队列里剩下的节点如果距离也是 k，一并收集
                # 这里 BFS 性质保证剩下的要么都是 k，要么更大（我们只要 k）
                while queue:
                    other_node, other_dist = queue.popleft()
                    if other_dist == k:
                        result.append(other_node.val)
                    else:
                        # 理论上不会出现 > k 的，因为我们到 k 就不再扩展下一层
                        break
                return result

            # 否则将相邻节点入队（左、右、父）
            neighbors = [node.left, node.right, parent_map.get(node)]
            for nei in neighbors:
                if nei and nei not in visited:
                    visited.add(nei)
                    queue.append((nei, dist + 1))

        return result
```

---

如果你愿意，下一步可以让你自己先写一遍，再让我帮你对照这个标准解答做“查漏补缺”，这样会更有记忆点。

---
## 🧑‍💻 User

上面代码中建图的过程跟二叉树的前中后序遍历有关系吗

---
## 🤖 Assistant

不算有实质关系。

更准确地说：

- **有关系的点：**
建图这一步本质上就是一次“遍历整棵树”，而**前序 / 中序 / 后序**这些都是“深度优先遍历（DFS）”的一种具体顺序，所以从“分类”上看，建图可以看成是一种 DFS 遍历。

- **没关系的点：**
这道题里，我们建图的目的只是：
> 把每一条父子边都记录成「双向边」。

为了做到这一点，你只需要**每个节点访问一次**，在访问到它的时候知道它的父节点是谁，然后连边即可。
至于你是：
 - 先处理根，再左，再右（前序）
 - 先左，再根，再右（中序）
 - 先左，再右，再根（后序）

对**结果完全没有影响**，因为我们根本不利用“遍历顺序”这个信息。

---

### 对照一下代码里的遍历方式

之前建图用的是这个思路（简化版）：

```python
stack = [(root, None)]  # (当前节点, 父节点)
while stack:
    node, parent = stack.pop()
    if parent:
        adj_list[node].append(parent)
        adj_list[parent].append(node)
    if node.left:
        stack.append((node.left, node))
    if node.right:
        stack.append((node.right, node))
```

- 这其实是一个**DFS**，而且逻辑上更接近“前序遍历”（先处理当前节点，再把子节点压栈）。
- 但这个“前序”仅仅是遍历实现细节，对建图结果没有额外价值。

同样的事情你用 BFS 也可以：

```python
from collections import deque, defaultdict

def build_graph(root):
    adj_list = defaultdict(list)
    queue = deque([(root, None)])
    while queue:
        node, parent = queue.popleft()
        if parent:
            adj_list[node].append(parent)
            adj_list[parent].append(node)
        if node.left:
            queue.append((node.left, node))
        if node.right:
            queue.append((node.right, node))
    return adj_list
```

效果一模一样。

---

### 面试中可以怎么说（20–30 秒版本）

> 建图这一步其实不依赖前中后序这种具体顺序，我只需要遍历一遍树，把每个父子关系记录成无向边就行。
> 我这里用的是一个简单的 DFS（实现上接近前序），用 BFS 或者中序、后序都可以，结果完全一样，因为我们不需要利用遍历顺序本身的性质，只需要把所有边都扫一遍建成邻接表。

记住一句话：
**这题建图只要求“遍历到每条边”，不要求“以什么顺序遍历”**。
