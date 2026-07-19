# KARDS 原生卡牌规则解析：Agent 交接任务书

## 任务目标

让模拟器能够执行唯一可信卡牌数据源
`data/source/kards_info_cards.json` 中描述的真实卡牌规则。最终产物必须是一套经过审查、可追溯的原生规则层：将卡牌的 `text` 与 `abilities` 转为可执行动作，并通过现有模拟器完成状态变更。

不得创建第二套卡牌数据库，也不得修改 kards.info 导出的原始数据。

这是训练真实 Battle AI、MCTS 和强化学习 Agent 的最高优先级阻塞项。当前没有可执行规则的卡牌会产生错误的状态转移。

## 当前仓库状态

- 仓库：`EvanProgramming/Kards-AI`，分支：`main`。
- 唯一卡牌目录：`data/source/kards_info_cards.json`（1488 张）。
- 规则产物：`data/rules/card_rules.json`。它是模拟器派生规则数据，**不是**替代卡牌目录。
- 解析器：`simulator/rules/parser.py`，生成 `CardRule` / `RuleAction` AST。
- 规则存储：`simulator/rules/store.py`，加载逐卡审查后的 AST 覆盖项。
- 执行器：`simulator/rules/native.py` 中的 `NativeRuleEngine`，执行 AST 并分发单位触发事件。
- 动作入口：`simulator/actions/action.py`，包括出牌、攻击、移动、换牌、结束回合。
- 通用原语：`simulator/effects/resolver.py` 与 `simulator/effects/basic_effects.py`。
- 反制流程：`simulator/countermeasures.py`。
- 既有测试：`tests/test_native_rules.py`。
- 可通过以下命令重新生成派生规则覆盖：

  ```bash
  python3 tools/build_card_rules.py
  ```

解析器刻意采用“失败即显式暴露”的策略：不确定文本必须标注为 `partial` 或 `unresolved`，绝不能猜测卡牌效果。必须保留此性质。

## 不可违反的约束

1. `kards_info_cards.json` 始终是唯一的卡牌事实来源，禁止编辑它。
2. 保持原生字段：`id`、`kredits`、`type`、`text`、`abilities` 等。不得恢复旧卡牌/effects/custom-rule 数据库或兼容层。
3. 禁止在运行时用 LLM 解析文本，禁止静默跳过规则。
4. 不得为普通文本效果写单卡 Python `if`。应新增通用 AST 动作、解析模板、可复用效果原语或机制级 Handler。
5. 真正不可约的复杂机制可以使用 Handler，但 Handler 名称必须描述“机制”，而不是卡名。
6. 未知或歧义文本必须保留在 `unresolved_fragments`，并在覆盖报告中可查。
7. 不得破坏已有 Action、State 序列化或 kards.info 数据迁移测试。

## 必须保持的架构

保持以下调用链，不得重构为另一套规则体系：

```text
Card.text / Card.abilities
        -> RuleParser 或已审查的 CardRuleStore 覆盖项
        -> CardRule(triggers, RuleAction[])
        -> NativeRuleEngine
        -> EffectResolver / HandlerRegistry / State 原语
        -> GameState + event log
```

单张卡的规则优先级：

1. `data/rules/card_rules.json` 中已审查的 `CardRuleStore` 条目。
2. `RuleParser` 内确定性的通用文本模板。
3. 明确产出 `partial` 或 `unresolved`。

规则只能在对应事件被发出时执行。目前已有或正在使用的事件包括：`on_play`、`on_deploy`、`on_destroy`、`on_attack`、`on_damage_dealt`、`on_combat_damage_dealt`、`on_enemy_hq_damaged`、`on_turn_start`、`on_turn_end`、`on_card_drawn`、`on_friendly_card_played` 和目标相关事件。仅当事件触发点与先后顺序已定义并测试时，才可新增事件。

## 实施计划

### 1. 建立可复现基线

执行并记录：

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q simulator tests
python3 tools/build_card_rules.py
```

按卡牌类型和机制统计 `implemented`、`partial`、`unresolved` 条目。生成结果只是工作队列，不是“规则已等价”的结论。必须批量检查 `unresolved_fragments`，不得靠人工随意挑卡处理。

### 2. 按可复用机制分类未解析文本

创建或更新机器可读报告（例如 `docs/rule_coverage.json`）以及可读的 `RULE_IMPLEMENTATION_REPORT.md`。每个未解析片段必须归入以下一类：

- 资源/费用修改，包括“每回合第一次”限制；
- 定向或全局属性变化；
- 移动、撤退、前线控制与部署；
- 抽牌、检索、展示、Develop、生成、复制、洗牌、弃牌；
- 伤害、战斗、摧毁、修复、治疗、HQ 防御；
- 选择、目标选择、随机性以及确定性 RNG；
- 临时效果、持续时间、延迟触发与失效；
- 关键词/能力语义；
- 反制触发、取消与消耗；
- 条件、阈值、历史状态（如“本回合”）与区域判定；
- 不可约的特殊机制；或
- 源文本信息不足/语义存在歧义。

每个类别都要输出：卡牌数量、片段数量、卡牌 ID、计划使用的 Action/Handler、状态，以及未完成原因。卡牌能打出且不崩溃，不代表它的规则已经实现。

### 3. 以高复用价值为顺序实现通用机制

依据分类报告，优先实现数量大且语义确定的机制。建议顺序：

1. 费用/资源变更，以及范围与持续时间。
2. 常见区域操作：抽取、检索、生成、复制、洗牌、弃牌、回手。
3. 目标筛选与多目标选择。
4. 临时/条件效果和事件历史判定。
5. 反制的触发、取消和消耗语义。
6. 其余关键词与战斗例外。

每个机制必须做到：

- 仅在确有多卡复用时扩展 `RuleAction`，新增字段必须强类型且可序列化；禁止塞入无结构 `dict`。
- 先增加精确、确定性的解析模板，再考虑更泛化的匹配。
- 能用 `EffectResolver` 执行的效果必须复用它。
- 如需新增底层原语，必须清楚定义目标校验、状态变更、事件发射与死亡处理。
- “本回合”等临时效果必须带有效期/事件作用域，禁止永久修改卡牌或单位。
- 所有随机选择必须统一走模拟器的带种子 RNG 协议，保证 replay 与训练可复现。

### 4. 仅为例外使用审查后的覆盖项

若文本无法安全通用解析，但行为可被确认，可在 `CardRuleStore` 中增加覆盖项，必须包括：

- 精确的 `card_id` 与对应 `source_text`；
- 明确的 `triggers`；
- 强类型 `actions`；
- `status`：`implemented`、`partial` 或 `unresolved`；
- 所有未覆盖子句写入 `unresolved_fragments`。

若例外确实需要控制流，通过既有 Handler 框架注册一个通用机制 Handler。可接受名称例如：`conditional_multi_target_damage`、`counter_next_order`、`copy_modified_unit`。除非它确实是唯一的规则引擎例外且理由已文档化，否则禁止使用 `handle_firestorm` 一类按卡名命名的 Handler。

### 5. 校正完整游戏流程集成

不能只验证 AST；必须验证完整生命周期：

```text
抽牌 -> 手牌 -> 校验合法动作 -> 支付费用 -> 出牌/部署
-> 按既定顺序发射事件 -> 解析规则/效果
-> 处理死亡与触发链 -> 墓地/激活区域 -> 下一动作
```

重点检查：

- 指令卡（Order）必须在完成最终清理前解析原生规则。
- 部署、攻击、伤害、死亡、移动、抽牌、回合边界都必须恰好一次地发射支持规则所需的事件。
- 反制必须按真实规则隐藏/激活，只对匹配的敌方事件触发，取消正确的操作，并且只消耗一次。
- 目标所有者、位置、类别、费用、免疫、容量等必须在支付资源前拒绝非法动作。
- 死亡连锁使用稳定快照，且不得重复触发。

## 测试与验收标准

为每种新增 Action 和事件增加聚焦回归测试。测试必须断言最终 State 和关键 event log，而不仅是“不抛异常”。

最低必需测试集：

- 解析器：代表性文本 -> 精确 AST、触发器、状态、未解析列表；
- 执行器：每个通用 Action 与目标选择器至少一个测试；
- 生命周期：出牌/部署/攻击/摧毁/回合开始/回合结束的事件顺序；
- 费用和临时效果：范围、下限和失效时点；
- 确定性随机：相同 seed 与相同行动必须得到相同 State；
- 反制：激活、匹配、取消、消耗、不匹配；
- 覆盖：1488 张卡均可加载，规则存储 ID 唯一且存在于目录中，`implemented` 条目不得带 `unresolved_fragments`；
- 回归：已有 migration、native-rule、training-stability 测试。

每个里程碑前执行：

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q simulator tests
python3 tools/build_card_rules.py
git diff --check
```

## 完成定义

本任务不是“每张卡都有 AST”就算完成。只有以下报告真实、可复现时，才可宣称对应里程碑完成：

- 给出准确目录总数（预期 1488）且每张卡恰有一条规则记录；
- 输出完全实现、部分实现、未实现、Handler 支持的精确数量；
- 按机制与卡牌类型统计数量；
- 列出所有待处理卡：`card_id`、卡名、精确未解析片段、分类、原因；
- 明确列出所有 Handler 支持的机制及其测试；
- 禁止静默降级、`custom_action_pending` 或未报告的未知 Action；
- 有文档化、经过测试的事件顺序契约；
- 全部自动化测试通过。

只要仍有重要指令卡、反制卡、关键词规则或高频单位效果待实现，就不能宣称模拟器已可进行真实 KARDS AI 训练。可以在一个里程碑结束时保留并报告 pending；不可以用猜测语义替代它们。

## 交付格式

按可独立测试的里程碑提交 commit 并推送。每个里程碑总结必须包含：

1. Commit hash 与修改文件列表。
2. 新增机制/Action，以及对应卡牌数和文本片段数的覆盖变化。
3. 剩余 `implemented` / `partial` / `unresolved` / Handler 支持数量。
4. 实际执行的测试命令与结果。
5. 需要人工或 KARDS 官方规则确认的任何源数据歧义。
