#!/usr/bin/env python3
"""生成 testset.json 评估集

特点：
1. 调整题型分布：减少统计题，增加理解和推理题
2. 新增评分维度字段：retrieval_weight, content_weight, citation_weight
3. 改进 must_have 为分层结构：must_have（必须）、should_have（加分）、must_not_have（扣分）
"""
import json
from pathlib import Path

def generate_testset():
    testset = {
        "meta": {
            "name": "agentic_rag_eval",
            "language": "zh",
            "notes_dir": "data/notes/my_markdowns",
            "total_questions": 25,
            "description": "部分得分机制、多维度评分"
        },
        "questions": []
    }
    
    questions = []
    
    # ========== 第一部分：知识理解类（12题）==========
    
    # Q1: Perfect Tree 节点数
    questions.append({
        "id": "Q01_perfect_tree_node_count",
        "query": "Perfect Binary Tree 的节点数公式是什么？",
        "category": "understanding",
        "difficulty": "easy",
        "expected_sources": ["二叉树 Perfect Complete Full 区别.md"],
        "scoring": {
            "retrieval_weight": 0.3,
            "content_weight": 0.5,
            "citation_weight": 0.2
        },
        "content_rules": {
            "must_have": [
                {"text": "2^h", "weight": 0.3},
                {"text": "节点", "weight": 0.2}
            ],
            "should_have": [
                {"text": "公式", "weight": 0.1}
            ],
            "evidence": [
                {"text": "2^h - 1", "weight": 0.4}
            ]
        },
        "citation_rules": {
            "require_quote": True,
            "require_source": False
        }
    })
    
    # Q2: 迷宫 bug
    questions.append({
        "id": "Q02_maze_exit_bug",
        "query": "在「力扣迷宫最近出口」这道题中，用户代码的主要 bug 是什么？",
        "category": "understanding",
        "difficulty": "medium",
        "expected_sources": ["力扣迷宫最近出口题解.md"],
        "scoring": {
            "retrieval_weight": 0.3,
            "content_weight": 0.5,
            "citation_weight": 0.2
        },
        "content_rules": {
            "must_have": [
                {"text": "入口", "weight": 0.25},
                {"text": "出口", "weight": 0.25}
            ],
            "evidence": [
                {"text": "把入口当成了出口", "weight": 0.5}
            ]
        },
        "citation_rules": {
            "require_quote": True,
            "require_source": False
        }
    })
    
    # Q3: 爬楼梯公式
    questions.append({
        "id": "Q03_climb_stairs_formula",
        "query": "爬楼梯问题的递推公式是什么？",
        "category": "understanding",
        "difficulty": "easy",
        "expected_sources": ["爬楼梯动态规划思路解析.md"],
        "scoring": {
            "retrieval_weight": 0.3,
            "content_weight": 0.5,
            "citation_weight": 0.2
        },
        "content_rules": {
            "must_have": [
                {"text": "f(n)", "weight": 0.2}
            ],
            "evidence": [
                {"text": "f(n) = f(n-1) + f(n-2)", "weight": 0.8}
            ]
        },
        "citation_rules": {
            "require_quote": True,
            "require_source": False
        }
    })
    

    # Q4: 恶意软件传播策略
    questions.append({
        "id": "Q04_malware_strategy",
        "query": "在「尽量减少恶意软件的传播」题目中，删除哪个节点能拯救最多节点？",
        "category": "understanding",
        "difficulty": "medium",
        "expected_sources": ["尽量减少恶意软件的传播.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "连通分量", "weight": 0.3}, {"text": "一个", "weight": 0.2}],
            "evidence": [{"text": "只有一个初始感染点", "weight": 0.5}]
        },
        "citation_rules": {"require_quote": True, "require_source": False}
    })

    # Q5: 腐烂橘子 bug
    questions.append({
        "id": "Q05_oranges_bug",
        "query": "在腐烂橘子题目中，用户代码 'grid[nr][nc] == 2' 这行有什么问题？",
        "category": "understanding",
        "difficulty": "easy",
        "expected_sources": ["BFS练习题 II.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "==", "weight": 0.3}],
            "evidence": [{"text": "用了 == (比较运算)，而不是 = (赋值运算)", "weight": 0.7}]
        },
        "citation_rules": {"require_quote": True, "require_source": False}
    })

    # Q6: 搜索二维矩阵映射
    questions.append({
        "id": "Q06_search_matrix_mapping",
        "query": "在力扣74搜索二维矩阵中，如何将一维索引映射到二维坐标？",
        "category": "understanding",
        "difficulty": "medium",
        "expected_sources": ["力扣74搜索二维矩阵讲解.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "行", "weight": 0.2}, {"text": "列", "weight": 0.2}],
            "evidence": [{"text": "row = mid // n, col = mid % n", "weight": 0.6}]
        },
        "citation_rules": {"require_quote": True, "require_source": False}
    })

    # Q7: 动态规划核心
    questions.append({
        "id": "Q07_dp_essence",
        "query": "动态规划的核心思想是什么？",
        "category": "understanding",
        "difficulty": "easy",
        "expected_sources": ["动态规划.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "状态", "weight": 0.4}, {"text": "转移", "weight": 0.3}],
            "should_have": [{"text": "子问题", "weight": 0.1}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q8: 回溯本质
    questions.append({
        "id": "Q08_backtrack_essence",
        "query": "回溯算法的本质是什么？",
        "category": "understanding",
        "difficulty": "medium",
        "expected_sources": ["回溯算法理解与子集问题本质.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "选择", "weight": 0.3}, {"text": "撤销", "weight": 0.3}],
            "evidence": [{"text": "做选择、递归、撤销选择", "weight": 0.4}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q9: Z字形遍历技巧
    questions.append({
        "id": "Q09_zigzag_trick",
        "query": "二叉树Z字形层序遍历有什么技巧？",
        "category": "understanding",
        "difficulty": "medium",
        "expected_sources": ["二叉树Z字形寻路题讲解.md", "二叉树锯齿形层序遍历解析.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "反转", "weight": 0.4}],
            "should_have": [{"text": "方向", "weight": 0.2}, {"text": "奇数", "weight": 0.2}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q10: 三种二叉树区别
    questions.append({
        "id": "Q10_tree_types_diff",
        "query": "Perfect、Complete、Full 三种二叉树的主要区别是什么？",
        "category": "understanding",
        "difficulty": "medium",
        "expected_sources": ["二叉树 Perfect Complete Full 区别.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [
                {"text": "Perfect", "weight": 0.15},
                {"text": "Complete", "weight": 0.15},
                {"text": "Full", "weight": 0.15}
            ],
            "evidence": [{"text": "Perfect 所有层都满", "weight": 0.3}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q11: 钥匙和房间
    questions.append({
        "id": "Q11_keys_rooms",
        "query": "力扣「钥匙和房间」这道题的核心思路是什么？",
        "category": "understanding",
        "difficulty": "medium",
        "expected_sources": ["力扣钥匙和房间题讲解.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "DFS", "weight": 0.3}],
            "should_have": [{"text": "visited", "weight": 0.2}, {"text": "遍历", "weight": 0.2}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q12: 乘积最大子数组
    questions.append({
        "id": "Q12_max_product",
        "query": "乘积最大子数组问题为什么需要同时维护最大值和最小值？",
        "category": "understanding",
        "difficulty": "hard",
        "expected_sources": ["乘积最大子数组题目讲解.md"],
        "scoring": {"retrieval_weight": 0.3, "content_weight": 0.5, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "负数", "weight": 0.3}, {"text": "最小", "weight": 0.2}],
            "evidence": [{"text": "负数乘以最小值可能变成最大值", "weight": 0.5}]
        },
        "citation_rules": {"require_quote": True, "require_source": False}
    })



    # ========== 第二部分：跨文档推理类（6题）==========

    # Q13: BFS vs DFS 空间复杂度
    questions.append({
        "id": "Q13_bfs_dfs_space",
        "query": "比较 BFS 和 DFS 在树遍历中的空间复杂度差异",
        "category": "reasoning",
        "difficulty": "hard",
        "expected_sources": ["BFS练习题 II.md", "DFS&回溯算法.md", "递归遍历.md"],
        "scoring": {"retrieval_weight": 0.4, "content_weight": 0.4, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [
                {"text": "BFS", "weight": 0.15},
                {"text": "DFS", "weight": 0.15},
                {"text": "空间", "weight": 0.2}
            ],
            "should_have": [{"text": "O(w)", "weight": 0.15}, {"text": "O(h)", "weight": 0.15}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q14: 连通分量相关题目
    questions.append({
        "id": "Q14_component_problems",
        "query": "哪些题目用到了连通分量的概念？列举题目名称",
        "category": "reasoning",
        "difficulty": "hard",
        "expected_sources": ["尽量减少恶意软件的传播.md", "力扣钥匙和房间题讲解.md"],
        "scoring": {"retrieval_weight": 0.4, "content_weight": 0.4, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "恶意软件", "weight": 0.3}, {"text": "连通", "weight": 0.3}],
            "should_have": [{"text": "钥匙", "weight": 0.2}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q15: DP 状态定义示例
    questions.append({
        "id": "Q15_dp_state_examples",
        "query": "在动态规划题目中，如何定义状态？请举2个具体例子",
        "category": "reasoning",
        "difficulty": "hard",
        "expected_sources": ["动态规划.md", "爬楼梯动态规划思路解析.md", "乘积最大子数组题目讲解.md"],
        "scoring": {"retrieval_weight": 0.4, "content_weight": 0.4, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "状态", "weight": 0.3}, {"text": "f(", "weight": 0.2}],
            "should_have": [{"text": "爬楼梯", "weight": 0.15}, {"text": "乘积", "weight": 0.15}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q16: 多源 BFS 题目
    questions.append({
        "id": "Q16_multi_source_bfs",
        "query": "哪些题目使用了多源 BFS？说明其特点",
        "category": "reasoning",
        "difficulty": "hard",
        "expected_sources": ["BFS练习题 II.md"],
        "scoring": {"retrieval_weight": 0.4, "content_weight": 0.4, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "多源", "weight": 0.3}, {"text": "BFS", "weight": 0.2}],
            "evidence": [{"text": "腐烂的橘子", "weight": 0.3}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q17: BFS 层序遍历技巧
    questions.append({
        "id": "Q17_bfs_level_size",
        "query": "在 BFS 层序遍历中，如何确保每次只处理一层的节点？",
        "category": "reasoning",
        "difficulty": "medium",
        "expected_sources": ["BFS练习题 II.md", "BFS求二叉树最大层和.md"],
        "scoring": {"retrieval_weight": 0.4, "content_weight": 0.4, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "len(queue)", "weight": 0.4}, {"text": "层", "weight": 0.2}],
            "evidence": [{"text": "level_size = len(queue)", "weight": 0.4}]
        },
        "citation_rules": {"require_quote": True, "require_source": False}
    })

    # Q18: 二分查找变体对比
    questions.append({
        "id": "Q18_binary_search_variants",
        "query": "笔记中涉及了哪些二分查找的变体或应用？",
        "category": "reasoning",
        "difficulty": "hard",
        "expected_sources": ["力扣74搜索二维矩阵讲解.md", "二分查找.md"],
        "scoring": {"retrieval_weight": 0.4, "content_weight": 0.4, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "二分", "weight": 0.3}],
            "should_have": [{"text": "矩阵", "weight": 0.2}, {"text": "mid", "weight": 0.2}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # ========== 第三部分：Negative Case（5题）==========

    # Q19: Dijkstra（文档未覆盖）
    questions.append({
        "id": "Q19_dijkstra_unknown",
        "query": "Dijkstra 算法的时间复杂度是多少？",
        "category": "negative",
        "difficulty": "easy",
        "expected_sources": [],
        "allow_unknown": True,
        "scoring": {"retrieval_weight": 0.2, "content_weight": 0.6, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [],
            "unknown_indicators": ["不知道", "没有", "未提及", "未找到", "无法确定"]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q20: 红黑树（文档未覆盖）
    questions.append({
        "id": "Q20_rbtree_unknown",
        "query": "如何实现红黑树的插入操作？",
        "category": "negative",
        "difficulty": "medium",
        "expected_sources": [],
        "allow_unknown": True,
        "scoring": {"retrieval_weight": 0.2, "content_weight": 0.6, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [],
            "unknown_indicators": ["不知道", "没有", "未提及", "未找到", "无法确定"]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q21: 快速排序（文档未覆盖）
    questions.append({
        "id": "Q21_quicksort_unknown",
        "query": "快速排序的最坏情况时间复杂度是什么？",
        "category": "negative",
        "difficulty": "easy",
        "expected_sources": [],
        "allow_unknown": True,
        "scoring": {"retrieval_weight": 0.2, "content_weight": 0.6, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [],
            "unknown_indicators": ["不知道", "没有", "未提及", "未找到", "无法确定"]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q22: 哈希表冲突（文档未覆盖）
    questions.append({
        "id": "Q22_hashtable_unknown",
        "query": "哈希表解决冲突的方法有哪些？",
        "category": "negative",
        "difficulty": "easy",
        "expected_sources": [],
        "allow_unknown": True,
        "scoring": {"retrieval_weight": 0.2, "content_weight": 0.6, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [],
            "unknown_indicators": ["不知道", "没有", "未提及", "未找到", "无法确定"]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q23: AVL树旋转（文档未覆盖）
    questions.append({
        "id": "Q23_avl_unknown",
        "query": "AVL树的旋转操作有哪几种？",
        "category": "negative",
        "difficulty": "medium",
        "expected_sources": [],
        "allow_unknown": True,
        "scoring": {"retrieval_weight": 0.2, "content_weight": 0.6, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [],
            "unknown_indicators": ["不知道", "没有", "未提及", "未找到", "无法确定"]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })



    # ========== 第四部分：统计分析类（2题）==========

    # Q24: 提交结果统计
    questions.append({
        "id": "Q24_submissions_result_counts",
        "query": "在 leetcode_submissions.md 中，Accepted 和 Wrong Answer 的提交各有多少次？",
        "category": "statistics",
        "difficulty": "medium",
        "expected_sources": ["leetcode_submissions.md"],
        "scoring": {"retrieval_weight": 0.2, "content_weight": 0.6, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "Accepted", "weight": 0.3}, {"text": "Wrong Answer", "weight": 0.3}],
            "should_have": [{"text": "次", "weight": 0.2}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })

    # Q25: 不同题目数量
    questions.append({
        "id": "Q25_unique_problem_count",
        "query": "leetcode_submissions.md 中一共涉及多少道不同的题目？",
        "category": "statistics",
        "difficulty": "medium",
        "expected_sources": ["leetcode_submissions.md"],
        "scoring": {"retrieval_weight": 0.2, "content_weight": 0.6, "citation_weight": 0.2},
        "content_rules": {
            "must_have": [{"text": "题目", "weight": 0.3}],
            "should_have": [{"text": "道", "weight": 0.2}]
        },
        "citation_rules": {"require_quote": False, "require_source": False}
    })


    print(f"生成了 {len(questions)} 道题目")
    testset["questions"] = questions
    return testset

if __name__ == "__main__":
    testset = generate_testset()
    output_path = Path(__file__).parent.parent / "testsets" / "testset.json"
    output_path.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 评估集已生成: {output_path}")
