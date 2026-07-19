# KARDS Simulator AI 训练就绪度审计

审计日期：2026-07-18  
审计范围：当前 `main` 分支（commit `e82db9f`）的源码、唯一数据库、自动化测试与官方 KARDS 规则说明。  
审计性质：只读评估；本报告不改变模拟器规则或卡牌数据。

## 结论

**总体完成度：30%（面向“真实 KARDS 对战 AI / RL 训练”的加权就绪度）。**

当前版本可用于：状态序列化、动作接口开发、基础环境集成和无规则语义保证的 smoke test。  
当前版本**不可用于训练真实 KARDS Battle AI、MCTS 或强化学习策略**。最关键原因是 1488 张卡的 `text` / `abilities` 尚未转化为可执行规则；训练会把绝大多数 Orders、Countermeasures、触发和单位能力误学为“无效果”。

## 1. 卡牌数据库

| 检查项 | 结果 | 证据 |
|---|---|---|
| 唯一卡牌来源 | 通过 | 仓库仅保留 `data/source/kards_info_cards.json`；加载器只提供 `CardDatabase.from_file()`。 |
| 加载数量 | 通过 | 1488；与 `metadata.totalCards = 1488` 一致。 |
| ID 唯一性 | 通过 | 加载器拒绝重复 ID；测试覆盖。 |
| 类型校验 | 通过 | `infantry` 459、`fighter` 131、`tank` 135、`bomber` 91、`artillery` 55、`order` 567、`countermeasure` 50。 |
| 加载错误 | 无 | 当前目录测试与直接加载均成功。 |
| 原始字段保留 | 通过 | 静态模型直接使用 `id`、`kredits`、`operationCost`、`text`、`abilities`、图片 URL 等原始字段。 |

数据源元数据为 kards.info 的 KARDS GraphQL catalog 导出，生成时间为 2026-05-13；其导出明确不包含完整 exile 集合，因此“当前游戏全部历史卡”不能据此断言。

## 2. 游戏规则层

| 领域 | 状态 | 审计发现 |
|---|---|---|
| HQ | 部分实现 | 有 20 HP、伤害、治疗、防御修正与事件日志。HQ 专属卡牌效果未实现。 |
| 胜负 | 部分实现 | HQ `<= 0` 时判定胜负/平局；未见其他终局、投降或比赛规则。 |
| 回合 | 部分实现 | 初始化 4/5 手牌、自动空换牌、每回合 Kredit 增长、抽牌、疲劳伤害、9 张手牌上限均存在。`turn_number` 每个玩家交接即递增，非完整回合计数。 |
| Kredit | 部分实现 | 上限 12、部署/移动/攻击扣费存在；卡牌减费、加费、临时资源等只在通用 Effect API 中，未由实际卡牌触发。 |
| Support Line | 部分实现 | 每方最多 4 单位、单位从手牌部署至 Support Line。符合官方“HQ + 4 单位”的容量描述。 |
| Frontline | 部分实现 | 最多 5、同一方独占、Support → Frontline、通常不可返回 Support Line。无“夺取前线”事件及其卡牌触发。 |

官方资料确认 Support Line 为 HQ 加最多 4 个单位、Frontline 最多 5 个且仅一方控制；当前容量和独占性实现与此一致，但相关卡牌触发尚未实现。[Support Line](https://support.kards.com/hc/en-us/articles/360026757171-Battlefield-elements-The-Support-Line) [Frontline](https://support.kards.com/hc/en-us/articles/360026464712-Battlefield-elements-The-Frontline)

## 3. 战斗

| 机制 | 状态 | 审计发现 |
|---|---|---|
| 攻击与操作费 | 部分实现 | 验证所属、回合、次数、部署回合、位置与操作费用；支持单位目标与 HQ 目标。 |
| 伤害/死亡 | 部分实现 | 单位互相伤害、HQ 伤害、`defense <= 0` 移出战场并进入墓地。 |
| 目标与位置规则 | 不完整 | 缺少全面目标合法性、攻击事件链、前线占领事件和卡牌例外规则。 |
| Blitz | 部分实现 | 可在部署回合攻击。 |
| Fury | 部分实现 | 每回合攻击次数为 2，并为每次攻击扣操作费。 |
| Heavy Armor | 部分实现 | 普通单位战斗伤害可减免；订单伤害未走此减免路径。 |
| Guard | 部分实现 | 只实现相邻 Guard 的基础阻挡；例外与全部目标规则未审计完成。 |
| 其他 catalog abilities | 未实现 | `ambush`、`smokescreen`、`bond`、`shock`、`alpine`、`covert`、`mobilize`、`intel*`、`salvage` 等未接入规则。 |

官方规则可验证 Fury 的双攻击/双付费、Heavy Armor 的单位战斗伤害减免等；当前仅覆盖其中一部分，不能宣称关键词完整。[Fury](https://support.kards.com/hc/en-us/articles/360026228852-Unit-ability-Fury) [Heavy Armor](https://support.kards.com/hc/en-us/articles/360026532571-Unit-ability-Heavy-Armor)

## 4. Card Effects 与 Handler 覆盖

### 实际卡牌覆盖（决定训练真实性）

| 指标 | 数量 | 说明 |
|---|---:|---|
| Catalog cards | 1488 | 全部可加载。 |
| 非空 `text` | 1488 | 每张均有规则文本。 |
| 已由卡牌自动执行的文本效果 | **0** | `RuleParser.dispatch()` 仅写入 `card_rule_dispatched` 日志，不解析或执行 `text`。 |
| 已由卡牌自动执行的 `abilities` | **0** | 除战斗代码直接检查的 Blitz/Fury/Heavy Armor/Guard 外，没有卡牌规则分派。 |
| 未实现的卡牌文本效果 | **1488 张** | 不能从自由文本可靠拆成“效果条数”；故按卡计数，而非伪造 clause 数。 |
| 真实卡牌绑定 Handler | **0** | 新数据库没有 handler 映射，`PlayCardAction` 也不调用 `HandlerRegistry`。 |

### 通用框架能力（不等于卡牌覆盖）

`EffectResolver` 可直接执行 26 个通用 action 名称/别名：伤害、治疗、buff/debuff、攻击/防御/费用修改、抽牌、弃牌、摧毁、移动、生成、压制、关键词、变形、复制、随机/选择及 HQ 防御等。它们只会在外部规则/handler 显式调用时生效；当前没有 kards.info 文本到这些动作的解析或映射。

`HandlerRegistry` 有 6 个通用 Handler：`DEPLOY`、`RANDOM`、`CHOICE`、`TRANSFORM`、`COPY`、`CONDITIONAL`。**已绑定到卡牌：0；所需 Handler 数：当前无法从自由文本精确量化。** 必须先建立可审计的规则 AST/覆盖清单后，才能统计哪些文本可 DSL 化、哪些确实需要 Handler。

Orders 目前会付费后进入墓地但不会执行其文本。Countermeasures 目前也未按“留在手牌、激活、仅于下个敌方回合自动触发、触发后弃置”的真实流程接入；这与官方机制不一致。[Orders](https://support.kards.com/hc/en-us/articles/360026754851-Orders) [Countermeasures](https://support.kards.com/hc/en-us/articles/360026404872-Countermeasure)

## 5. Deck Builder

| 要求 | 状态 | 审计发现 |
|---|---|---|
| 40 张 | 部分实现 | `DeckValidator` 检查 40；`Simulator.reset()` 仅检查 ID 存在，不强制 40。 |
| Nation 规则 | 部分实现 | `DeckValidator` 允许 main nation、可选 ally 与 Neutral；未验证真实阵营/盟友合法组合。 |
| Copy 限制 | 通过（静态） | Standard 4、Limited 3、Special 2、Elite 1；与官方卡牌说明一致。 |
| Token 过滤 | 通过（静态） | `set == "OnlySpawnable"` 被拒绝；不使用不可靠的 `spawnable` 标记。 |
| 训练环境强制合法卡组 | 未实现 | API 可以绕过 `DeckValidator`，传入任意长度、国家或复制数的已知 ID 卡组。 |

官方规则确认四种稀有度的复制上限；静态校验与其一致。[Cards](https://support.kards.com/hc/en-us/articles/360026768151-Cards)

## 6. AI Environment API

| API | 状态 | 审计发现 |
|---|---|---|
| `reset()` | 部分实现 | 建状态、洗牌、4/5 起手、自动空换牌、首回合资源；不强制合法卡组。 |
| `step(action)` | 部分实现 | 执行动作、返回 `(state, reward, done)`；reward 只基于对方 HQ HP 变化和胜局 +100。 |
| action space | 部分实现 | 产生 Pass、可支付 Play、Move、Attack；不提供 Mulligan 选择、Countermeasure 激活/取消、目标化 Orders、UseAbility。 |
| state observation | 部分实现 | 可 clone、JSON serialize/restore；直接暴露完整 `GameState`，没有固定 shape、特征编码、合法动作 mask 或玩家视角隐藏信息。 |
| 可复现性 | 部分实现 | reset 洗牌采用 seed；部分随机行为和事件/效果尚未形成完整统一 RNG 协议。 |

## 训练前必须修复（按优先级）

1. **实现可执行的 native rule layer。** 将 `Card.text` / `abilities` 解析或映射为规则 AST，并从事件分派到 `EffectResolver`/`HandlerRegistry`；建立逐卡覆盖报告。
2. **实现 Orders、Countermeasures、触发、延迟效果与目标选择。** 否则 617 张非单位牌及大量单位文本均为错误转移。
3. **补齐全部关键词与精确战斗/目标/前线规则。** 尤其是 catalog 中实际出现但未实现的 10 余类 abilities。
4. **在 `Simulator.reset()` 强制 DeckValidator。** 拒绝非 40、国家非法、超复制及 OnlySpawnable 卡组。
5. **完成动作空间。** 加入 mulligan 决策、Order target、Countermeasure activate/deactivate、UseAbility 和选择动作；输出合法 action mask。
6. **建立可重复的规则测试。** 至少逐类效果、关键词、Countermeasure、触发时序、隐藏信息、疲劳和长期对局回归；当前仅 6 个迁移测试。
7. **定义训练观测与 reward。** 固定玩家视角编码、隐藏对手手牌/已激活反制、确定性 RNG、胜负/中间 reward 的规范。

## 可开始的训练范围

| 训练目标 | 结论 |
|---|---|
| API / 模型管线联调 | 可以，明确标记为 synthetic smoke environment。 |
| Deck Builder 的静态合法性过滤 | 可以，需通过 DeckValidator 而非 Simulator.reset。 |
| 真实对战策略、MCTS、强化学习 | **不可以开始正式训练。** 当前转移函数对绝大多数卡牌不真实，会产生系统性错误策略。 |

## 验证记录

- `python3 -m unittest discover -s tests -v`：6/6 通过。
- `python3 -m compileall -q simulator tests`：通过。
- catalog 直接加载：1488 cards，无加载错误。
- 10 回合随机 smoke simulation：完成；这只证明不会崩溃，不证明规则正确性。
