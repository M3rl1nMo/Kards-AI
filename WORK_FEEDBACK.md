# KARDS 模拟器 — 覆盖造假修复与真实实现推进

**日期**: 2026-07-19  
**涉及代码库**: `/private/tmp/kards-remote-audit.3zhjEE/`  
**审计依据**: `docs/needs_kards_confirmation.md`（361 张真实 backlog）

---

## 一、问题诊断

### 原始状态：虚假 100% 覆盖

模拟器声称 `1488 implemented / 0 partial / 0 unresolved`，但存在三层欺骗：

| 层 | 文件 | 造假手段 |
|:--|:--|:--|
| Parser | `simulator/rules/parser.py` | `_classify_residual()` 将所有未识别文本强制映射为 `control_effect`、`aura_buff` 等机制标签。`_CATCH_ALL_KINDS` 只有 3 种，漏掉了其他 30+ 种兜底标签 |
| 引擎 | `simulator/rules/native.py` | 30+ 种机制的 catch-all handler 只写 `unit.status[kind]=True`（死数据），无任何后续代码消费 |
| 覆盖工具 | `tools/build_card_rules.py` | `extract_executable_kinds()` 扫描源码字符串 `kind == "X"`，只要匹配到就标为 executable |

### 与 audit 文档的矛盾

`docs/needs_kards_confirmation.md` 独立记录了 **361 张 partial 卡**（264 ENGINE_GAP + 34 AURA + 63 PRINCIPLE），与 "0 pending" 直接矛盾。

---

## 二、修复内容总览

### 核心文件改动

| 文件 | 改动 |
|:--|:--|
| `simulator/rules/parser.py` | `_CATCH_ALL_KINDS` 3→38 种；30+ M61 模板从 `control_effect` 改写为真实 kind；新增 25+ 精准模板 |
| `simulator/rules/native.py` | 30 种死数据 catch-all → `engine_stubbed_action`；新增 30+ 真实 handler |
| `tools/build_card_rules.py` | 字符串扫描 → `_GENUINELY_EXECUTED_KINDS` 手维护白名单（50→98 种） |
| `simulator/core/state.py` | 新增 `GameState.pending_cancels`、`PlayerState.status` |
| `RULE_IMPLEMENTATION_REPORT.md` | 加入诚实化说明和 `needs_kards_confirmation.md` 交叉引用 |
| 测试文件 | 13 个旧测试从验证假行为改为验证诚实行为；新增 9 个 handler 验证测试 |

### 新增/实现的机制（按轮次）

**Round 2 — 真实 handler**  
`double_stats`, `double_damage`, `triple_damage`, `cannot`, `immune`, `cannot_attack_hq`

**Round 3 — 已有 handler 补白名单**  
`convert_to`, `spawn_named_same_front`, `draw_top_matching`, `remove`, `copy_unit`, `gain_hq_defense`, `gain_hq_defense_equal_damage`, `discard_hand`, `take_control`, `reset_operation`, `retreat_and_repair`, `order_damage_bonus`

**Round 3 — 新实现 handler**  
`double_damage_against_type`, `attack_bonus_against_type`, `attack_bonus_against_higher_attack`, `damage_bonus_against_air`, `cannot_be_targeted_by_enemy_orders`, `target_or_attack_tax`, `hq_excess`, `destroy_cost_lte`, `grant_trait`

**Round 4 — 子系统**  
`repeat_effect`（效果重复触发，event_log 防无限循环）, `aura_buff`（带作用域/类型/位置过滤的光环）, `cancel`（反制标记→GameState.pending_cancels）, `pincer_ability`（8 种 Pincer 子效果）, `target_select`（目标选择标记）

**Round 5 — 小型机制**  
`enemy_cannot_deploy`, `enemy_cannot_order`, `buff_on_played_ability`, `buff_on_enemy_target`, `attack_on_order_played`, `immune_to_ground_in_support`, `ignore_heavy_armor`, `hq_damage_bonus`, `countermeasure_lock`, `hq_defense_lock`, `hq_defense_cap`, `set_kredit_slots_equal`, `enemy_kredit_change`, `kredit_effect`, `delayed_return`, `modify_cost`

---

## 三、覆盖进度

| 阶段 | implemented | partial | 说明 |
|:--|:--|:--|:--|
| 起始（欺诈） | 1181 | 307 | 1488/0/0 是假的 |
| 诚实化修复 | 1181 | 307 | 只修了造假基础设施 |
| Round 2 | 1303 | 185 | 真实 handler + 白名单补全 |
| Round 3 | 1374 | 114 | 批量已有 handler 补全 + 新 handler |
| Round 4 | 1412 | 76 | 子系统实现 |
| Round 5 | **1452** | **36** | 小型机制清零 + control_effect 拆解 |

**累计增益**: +271 implemented（79.4% → 97.6%），-271 partial

---

## 四、剩余 36 张 partial 卡

全部为 `control_effect`，属于需要逐卡实现的深层游戏机制：

| 类型 | 示例 | 卡数 |
|:--|:--|:--|
| 伤害重定向 | "Damage to STIRLING is dealt to this unit instead" | ~5 |
| HQ 伤害重定向 | "When HQ takes damage, enemy HQ takes instead" | ~3 |
| 位置交换 | "Swap a position with a friendly unit" | ~2 |
| 战场规则修改 | "Only 2 units can occupy the frontline" | ~1 |
| 特定触发效果 | "Trigger this card when enemy deploys..." | ~10 |
| 随机重排等 | "randomly rearrange enemy units" | ~4 |
| 其他 | 59. Panzergrenadier 移动标记等 | ~11 |

这些需要逐机制配合游戏流程集成，不适合批量模板化。

---

## 五、验证

- 测试套件: **142/142 全部通过**
- engine_stubbed kinds: 从 30+ 种归零到仅剩 `control_effect`、`hq_effect`、`special_effect`、`gain_kredit_slot_on_deploy`（4 种，正在逐个实现中）
- 测试覆盖了所有新增 handler 的基本行为（`tests/test_genuine_handlers.py`）

---

## 六、后续建议

1. **逐卡实现剩余 36 张**：按伤害重定向→位置交换→战场规则→特定触发的顺序
2. **补齐逐卡 test fixture**：当前 1452 "implemented" 仅保证 parser 覆盖+引擎有 handler，不等于规则正确。需要为每个机制至少一个状态转移测试
3. **UseAbilityAction** 仍然报 NotImplementedError，需要在 Phase 3 实现
4. **Countermeasure 子系统**：虽然 `cancel` 有了标记机制，但真正的反制拦截需要 action 验证层消费 `pending_cancels`
5. **光环生命周期**：`aura_buff` 已有 source 追踪，但离场回退机制尚未接入回合流程
