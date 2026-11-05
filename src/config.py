"""配置加载 + 运行时覆盖。

消融回测需要在同一进程内跑多组配置，而 load_config 有全局缓存，
所以提供 set_override 注入临时配置，跑完再清除。
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import re

_config_cache = None
_override: Dict[str, Any] = {}


def _coerce_numeric(val: Any) -> Any:
    """把 YAML 里写成字符串的数值转成 float/int。

    YAML 1.1 把 `3e9` 解析成字符串（科学计数法必须带小数点如 `3.0e9`），
    但 universe.filter 里的 `between` 比较需要数值。
    """
    if isinstance(val, str):
        s = val.strip()
        if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", s):
            try:
                num = float(s)
                return int(num) if num.is_integer() else num
            except ValueError:
                pass
    return val


def _coerce_tree(obj: Any) -> Any:
    """递归转换配置树中的所有叶子节点"""
    if isinstance(obj, dict):
        return {k: _coerce_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_tree(v) for v in obj]
    return _coerce_numeric(obj)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件（全局缓存一次）"""
    global _config_cache
    if _config_cache is None:
        with open(path, 'r', encoding='utf-8') as f:
            _config_cache = _coerce_tree(yaml.safe_load(f))
    return _config_cache


def _deep_merge(base: Dict, patch: Dict) -> Dict:
    """把 patch 深合并进 base 的副本"""
    out = {k: v for k, v in base.items()}
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_override(patch: Dict[str, Any]) -> Dict[str, Any]:
    """注入运行时配置覆盖（不写盘，不污染 config.yaml）

    用于消融回测：跑某组配置，跑完调用 clear_override 恢复。
    """
    global _override
    _override = _deep_merge(_override, patch)
    return get_config_dict()


def clear_override() -> Dict[str, Any]:
    """清除运行时覆盖，恢复 config.yaml 原值"""
    global _override
    _override = {}
    return get_config_dict()


def get_config_dict() -> Dict[str, Any]:
    """取当前生效的完整配置（缓存 + 覆盖合并）"""
    return _deep_merge(load_config(), _override)


def get_config(key_path: str, default: Any = None) -> Any:
    """按路径获取配置项，例: get_config("universe.min_price")"""
    cfg = get_config_dict()
    val = cfg
    for k in key_path.split('.'):
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def module_enabled(path: str, default: bool = True) -> bool:
    """读取 modules.<path> 子模块开关（组合策略开关表，见 config.yaml 的 modules 段）。

    查找顺序（先到先得）：
      1. ablation.<path>    —— 消融框架通过 set_override 注入的显式覆盖，优先级最高
      2. modules.<path>     —— config.yaml 中声明的持久开关
      3. default            —— 两者都没有时的安全默认（True = 启用）

    例: module_enabled("risk.enabled") → ablation.risk.enabled 或 modules.risk.enabled。
    """
    for prefix in ("ablation", "modules"):
        v = get_config(f"{prefix}.{path}", None)
        if v is not None:
            return bool(v)
    return default
