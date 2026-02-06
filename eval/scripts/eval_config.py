"""Evaluation configuration loader.

统一配置加载模块，支持从 JSON 文件加载配置，提供默认值回退。

配置优先级：题目配置 > 类别配置 > 默认配置

Usage:
    from eval.scripts.eval_config import load_eval_config, get_config

    # 加载配置
    config = load_eval_config()

    # 获取特定配置
    threshold = get_config().thresholds.perfect_score_threshold
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 配置文件路径
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "eval_config.json"


@dataclass
class ScoringWeights:
    """评分权重配置"""
    retrieval_weight: float = 0.3
    content_weight: float = 0.5
    citation_weight: float = 0.2
    tool_weight: float = 0.0


@dataclass
class ScoringConfig:
    """评分配置"""
    default_weights: ScoringWeights = field(default_factory=ScoringWeights)
    category_weights: Dict[str, ScoringWeights] = field(default_factory=dict)


@dataclass
class ThresholdsConfig:
    """阈值配置"""
    pass_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "understanding": 0.7,
        "reasoning": 0.65,
        "negative": 0.8,
        "statistics": 0.7,
        "skill": 0.7,
        "web_search": 0.6,
        "multi_turn": 0.7,
        "default": 0.7,
    })
    perfect_score_threshold: float = 1.0
    semantic_similarity_threshold: float = 0.75
    semantic_match_weight: float = 0.7


@dataclass
class RecallConfig:
    """召回率配置"""
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    k_for_scoring: int = 5
    use_chunk_level: bool = False


@dataclass
class ToolEvaluationConfig:
    """工具评估配置"""
    redundancy_penalty_rate: float = 0.2
    order_violation_max_penalty: float = 0.3


@dataclass
class UnknownDetectionConfig:
    """未知检测配置"""
    patterns: List[str] = field(default_factory=lambda: [
        r"未知",
        r"不知道",
        r"无法确定",
        r"无相关信息",
        r"文档未覆盖",
        r"没有.*信息",
        r"未提及",
        r"没有.*相关",
        r"笔记.*没有",
        r"未.*找到",  # 改为 未.*找到 以匹配"未在笔记中找到"
        r"无法回答",
        r"没有.*记录",  # 新增：匹配"没有这方面的记录"
    ])
    exclusion_patterns: List[str] = field(default_factory=lambda: [
        r"虽然.*但",
        r"虽然.*不过",
        r"虽然没有.*可以",
        r"虽然未.*但",
    ])
    lazy_rejection_penalty: float = 0.5
    # "说了不知道 + 提供补充选项"的得分（用户体验合理，不应判为 hallucination）
    soft_unknown_score: float = 0.8


@dataclass
class SynonymConfig:
    """同义词配置"""
    global_synonyms: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """评估系统配置"""
    version: str = "1.0"
    description: str = ""
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    tool_evaluation: ToolEvaluationConfig = field(default_factory=ToolEvaluationConfig)
    unknown_detection: UnknownDetectionConfig = field(default_factory=UnknownDetectionConfig)
    synonyms: SynonymConfig = field(default_factory=SynonymConfig)


# 全局配置实例
_config: Optional[EvalConfig] = None


def _parse_scoring_weights(data: Dict[str, Any]) -> ScoringWeights:
    """解析评分权重"""
    return ScoringWeights(
        retrieval_weight=data.get("retrieval_weight", 0.3),
        content_weight=data.get("content_weight", 0.5),
        citation_weight=data.get("citation_weight", 0.2),
        tool_weight=data.get("tool_weight", 0.0),
    )


def _parse_scoring_config(data: Dict[str, Any]) -> ScoringConfig:
    """解析评分配置"""
    default_weights = _parse_scoring_weights(data.get("default_weights", {}))
    category_weights = {}
    for cat, weights in data.get("category_weights", {}).items():
        category_weights[cat] = _parse_scoring_weights(weights)
    return ScoringConfig(default_weights=default_weights, category_weights=category_weights)


def _parse_thresholds_config(data: Dict[str, Any]) -> ThresholdsConfig:
    """解析阈值配置"""
    config = ThresholdsConfig()
    if "pass_thresholds" in data:
        config.pass_thresholds.update(data["pass_thresholds"])
    if "perfect_score_threshold" in data:
        config.perfect_score_threshold = data["perfect_score_threshold"]
    if "semantic_similarity_threshold" in data:
        config.semantic_similarity_threshold = data["semantic_similarity_threshold"]
    if "semantic_match_weight" in data:
        config.semantic_match_weight = data["semantic_match_weight"]
    return config


def _parse_recall_config(data: Dict[str, Any]) -> RecallConfig:
    """解析召回率配置"""
    return RecallConfig(
        k_values=data.get("k_values", [1, 3, 5, 10]),
        k_for_scoring=data.get("k_for_scoring", 5),
        use_chunk_level=data.get("use_chunk_level", False),
    )


def _parse_tool_evaluation_config(data: Dict[str, Any]) -> ToolEvaluationConfig:
    """解析工具评估配置"""
    return ToolEvaluationConfig(
        redundancy_penalty_rate=data.get("redundancy_penalty_rate", 0.2),
        order_violation_max_penalty=data.get("order_violation_max_penalty", 0.3),
    )


def _parse_unknown_detection_config(data: Dict[str, Any]) -> UnknownDetectionConfig:
    """解析未知检测配置"""
    config = UnknownDetectionConfig()
    if "patterns" in data:
        config.patterns = data["patterns"]
    if "exclusion_patterns" in data:
        config.exclusion_patterns = data["exclusion_patterns"]
    if "lazy_rejection_penalty" in data:
        config.lazy_rejection_penalty = data["lazy_rejection_penalty"]
    return config


def _parse_synonym_config(data: Dict[str, Any]) -> SynonymConfig:
    """解析同义词配置"""
    return SynonymConfig(
        global_synonyms=data.get("global", {})
    )


def load_eval_config(config_path: Optional[Path] = None) -> EvalConfig:
    """加载评估配置。

    Args:
        config_path: 配置文件路径，默认为 eval/config/eval_config.json

    Returns:
        EvalConfig 实例

    Notes:
        - 如果配置文件不存在，返回默认配置
        - 配置会被缓存到全局变量，后续 get_config() 调用返回相同实例
        - 传入 config_path 会强制重新加载并更新全局缓存
    """
    global _config

    # 只有在未指定路径且已有缓存时才返回缓存
    if _config is not None and config_path is None:
        return _config

    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        _config = EvalConfig()
        return _config

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError) as e:
        print(f"警告: 配置文件加载失败 ({e})，使用默认配置")
        _config = EvalConfig()
        return _config

    config = EvalConfig(
        version=data.get("version", "1.0"),
        description=data.get("description", ""),
    )

    if "scoring" in data:
        config.scoring = _parse_scoring_config(data["scoring"])
    if "thresholds" in data:
        config.thresholds = _parse_thresholds_config(data["thresholds"])
    if "recall" in data:
        config.recall = _parse_recall_config(data["recall"])
    if "tool_evaluation" in data:
        config.tool_evaluation = _parse_tool_evaluation_config(data["tool_evaluation"])
    if "unknown_detection" in data:
        config.unknown_detection = _parse_unknown_detection_config(data["unknown_detection"])
    if "synonyms" in data:
        config.synonyms = _parse_synonym_config(data["synonyms"])

    # 始终更新全局缓存，确保 get_config() 返回正确的配置
    _config = config

    return config


def get_config() -> EvalConfig:
    """获取当前配置实例。

    如果尚未加载配置，会自动加载默认配置。
    """
    global _config
    if _config is None:
        _config = load_eval_config()
    return _config


def reset_config() -> None:
    """重置配置缓存，用于测试。"""
    global _config
    _config = None


def get_scoring_weights(
    question: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None
) -> ScoringWeights:
    """获取评分权重，支持配置优先级。

    优先级：题目配置 > 类别配置 > 默认配置

    Args:
        question: 题目配置字典，可包含 scoring 字段
        category: 题目类别

    Returns:
        ScoringWeights 实例
    """
    config = get_config()

    # 默认权重
    weights = ScoringWeights(
        retrieval_weight=config.scoring.default_weights.retrieval_weight,
        content_weight=config.scoring.default_weights.content_weight,
        citation_weight=config.scoring.default_weights.citation_weight,
        tool_weight=config.scoring.default_weights.tool_weight,
    )

    # 类别权重覆盖
    if category and category in config.scoring.category_weights:
        cat_weights = config.scoring.category_weights[category]
        weights.retrieval_weight = cat_weights.retrieval_weight
        weights.content_weight = cat_weights.content_weight
        weights.citation_weight = cat_weights.citation_weight
        weights.tool_weight = cat_weights.tool_weight

    # 题目配置覆盖
    if question and "scoring" in question:
        q_scoring = question["scoring"]
        if "retrieval_weight" in q_scoring:
            weights.retrieval_weight = q_scoring["retrieval_weight"]
        if "content_weight" in q_scoring:
            weights.content_weight = q_scoring["content_weight"]
        if "citation_weight" in q_scoring:
            weights.citation_weight = q_scoring["citation_weight"]
        if "tool_weight" in q_scoring:
            weights.tool_weight = q_scoring["tool_weight"]

    return weights


def get_pass_threshold(category: str) -> float:
    """获取指定类别的通过阈值。

    Args:
        category: 题目类别

    Returns:
        通过阈值
    """
    config = get_config()
    return config.thresholds.pass_thresholds.get(
        category,
        config.thresholds.pass_thresholds.get("default", 0.7)
    )


if __name__ == "__main__":
    # 测试配置加载
    config = load_eval_config()
    print(f"配置版本: {config.version}")
    print(f"满分阈值: {config.thresholds.perfect_score_threshold}")
    print(f"语义匹配阈值: {config.thresholds.semantic_similarity_threshold}")
    print(f"通过阈值: {config.thresholds.pass_thresholds}")
    print(f"Recall K 值: {config.recall.k_values}")
    print(f"未知检测模式数: {len(config.unknown_detection.patterns)}")
