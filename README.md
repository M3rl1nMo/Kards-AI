# KARDS 规则模拟器（仅模拟器，神经网络部分自行设计）

初始模拟器代码提取自 [EvanProgramming/Kards-AI](https://github.com/EvanProgramming/Kards-AI)，保留卡牌数据、规则执行、游戏状态与相关测试，不含原项目的神经网络、训练或搜索模块。后续将自主设计 AI 模型与训练系统。

本项目独立维护，与 KARDS 官方没有隶属关系。上游代码及卡牌数据的许可尚待核实；来源说明不授予额外使用权限。

## 事实核查（本目录已实测）

- `simulator/` **只依赖 Python 标准库**：不 import `ai`、`torch`、`numpy`、`yaml`。
- **Python 要求：`>=3.9`**（无 `match` 语句；全部使用 `from __future__ import annotations`）。
- 测试：2026-10-08 修改 DILEMMA 后，`208` 个模拟器测试全部通过（本机 Python 3.12.14）。
- 历史入场检查：`eligible_cards=1486`，`episodes=32`；默认法国卡池未触发 unsupported effect，不代表全卡池验证通过。
- 扩展测试：修复前五个主国随机对局 50 局中有 23 局触发未支持效果；DILEMMA 修复后，英国/德国简单固定卡组 100 局、373 次交换，未支持效果及交换数值错误均为 0。
- 中文问题说明、原因和修改方法见 [模拟器问题说明](docs/simulator_bug_notes_zh.md)。源码中用 `BUG-01` 至 `BUG-12` 标注目前发现的问题；这不是全卡池审计结论。
- 每次开发进展与待办事项见 [开发日志](docs/development_log_zh.md)。
- BUG-01～09 修复/接口补充后，测试共 220 项全部通过，无预期失败；BUG-09 的费用增减与重复交换已通过定向验证，详见中文文档。
- 默认严格效果检查会拒绝未支持效果并停止该局；此前能运行的随机全卡池对局可能因此报错，需排查效果或缩小到已验证卡组。
- 引擎吞吐：`--no-replay --training-path` 约 **53.6 games/s、5,990 steps/s**（Ryzen 7 9700X）。

## 两个**必需**的数据文件（缺一即报错）

| 文件 | 作用 | 缺失后果 |
| --- | --- | --- |
| `data/source/kards_info_cards.json` | 1,488 张卡目录 | `CardDatabase.from_file` 直接 `FileNotFoundError` |
| `data/rules/card_rules.json` | 逐卡结构化规则 AST；状态标记不等于全部执行路径正确 | `NativeRuleEngine` 构造时 `FileNotFoundError` |

注意第二项：`simulator/rules/native.py` 用 `Path(__file__).parents[2] / "data/rules/card_rules.json"`
**按目录相对位置**加载，所以 `simulator/` 必须与 `data/` 保持同级。
`NativeRuleEngine(..., store=CardRuleStore({}))` 可以绕过该文件并回退到实时解析，
实测解析结果与已存规则**逐卡完全一致**（1488/1488 相同），但这只是逃生舱，不是默认路径。

## 用法

```powershell
python -m unittest discover -s tests        # DILEMMA 修复后基线为 208 项
python -m training.preflight                # eligible_cards=1486
python tools/profile_simulator.py --games 8 --max-steps 300 --seed 71 --no-replay --training-path
python tools/build_card_rules.py            # 仅当卡文本/规则改动时重建 data/rules/card_rules.json
python tools/classify_coverage.py           # 重新生成 docs/rule_coverage.json
```

## 面向神经网络设计者的接口契约

模拟器提供的边界（`simulator/core/game.py`）：

| 接口 | 说明 |
| --- | --- |
| `reset(deck_p1, deck_p2, nations, seed, auto_mulligan=False, ally_nations)` | 起始手牌 p1=4 / p2=5；默认 True 保留全部起手牌；自动/手动都在最后一次换牌结束后启动 p1 首回合（BUG-03 已修复） |
| `get_available_actions()` | **唯一合法动作权威**，返回完整 `Action` 对象列表；已按状态缓存 |
| `step(action)` / `step_fast(action)` | 前者额外克隆一次状态（调试用），后者返回 `(reward, terminal)`，训练请用后者 |
| `get_state()` | 返回状态克隆 |
| `GameState.clone_for_search()` | MCTS 分支用的快速隔离克隆（事件日志浅拷贝，因规则只追加不修改） |
| `get_observation(player_id)` | 玩家视角 dict，**对手手牌为 `None`**、对手克数隐藏 |
| `get_player_actions()` / `step_player(action)` | 网络使用的动作接口，与观测共用不带卡名的单位编号；原始 get_available_actions 返回裁判内部 ID |
| `GameState.to_dict()/from_dict()/to_json()` | 完整裁判状态序列化（禁止进入玩家侧输入） |

自己实现观测/动作编码时必须知道的几件事：

1. **随机数**：规则层一律用 `random.Random(state.rng_seed + len(state.event_log))`，
   所以**给定种子与动作序列，整局可完全确定性重放**；换牌会洗牌（用同一推导）。
2. **隐藏信息**：`PlayerState.hand`/`deck` 是**完整**信息，`ObservationEncoder` 之类
   若吃 `GameState` 就会看到对手手牌与牌库顺序。搜索若直接用全状态 = 确定化作弊。
3. **单位位置不是槽位**：`state.battlefield` 是 `{"frontline": [instance_id...], "support_line": [...]}`
   两个**有序列表**；支援线同时最多 4 个单位（`action.py:112`）。没有固定的 1..N 槽位，
   做“第 i 格”编码需要自己定义顺序并把列表顺序**显式**转为位置语义。
4. **`state.units` 顺序**不等于战场槽位顺序，且 `UnitState.status` 是个大杂烩 dict。
5. **奖励语义**：`step_fast` 的 `reward` 是行动方造成的对手 HQ 掉血，加非平局终局的 +100。
   BUG-04 已修复：行动方胜利 +100、失败 -100、平局无终局加分；掉血奖励仍保留，不等同于纯胜负目标。
6. **动作上限**：上限截断**不是平局**。应记录 `truncated` 并用自举价值，别写 0.0 目标。
7. **自身反制措施**：观测已包含己方 active_countermeasures、费用规则和单位修正；对方秘密仍隐藏。后续新增卡牌机制需同步审查观测。

## 未包含（按设计留给自研部分）

`ai/`（观测/动作编码、网络、MCTS、自对弈、训练器、评估器、metrics、runtime 路径）、
`configs/`、`selfplay.py`/`train.py`/`evaluate.py`/`auto_train.py`/`monitor.py`、
`tests/test_ai_training_framework.py`、`tests/test_training_correctness.py`、
`tools/benchmark_*`。

另有一份**独立于 AI 内部**的真人对战 + 数据采集参考实现留在
`Kards-AI-main/ai/hero/` 与 `Kards-AI-main/hero_play.py`（只用模拟器接口 + 标准库），
其落盘格式（玩家视角决策流 / 裁判流分离）可直接沿用。
