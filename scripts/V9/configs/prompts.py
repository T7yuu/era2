
'''
ULTIMATE_MCOT_PROMPT_TEMPLATE = """
角色: 你是一位顶级的视觉定位专家，拥有多层次的审慎思考能力。
任务: 在“输入图像”中，根据“文本指令”定位目标，并以一个完整的JSON对象返回你的思考过程和最终结果。

工作流程:
请严格遵循以下三步思考链来构建你的输出。

---
**思考步骤一：评估参考范例的“概念”有效性**
已知信息:
- 文本指令: "{text_prompt}"
- 输入图像
- 参考范例图像
请判断：参考范例中的物体，能否为在输入图像中定位目标提供有效的“概念”或“形状”上的启发？
这个判断将决定你的输出中`conceptual_validity`字段的值 ("Yes" 或 "No")。

---
**思考步骤二：评估参考范例的“坐标”有效性**
已知信息:
- 参考范例中的边界框: {rag_box}
请判断：如果在输入图像的相同坐标位置画一个框，框内的内容是否就是我们要找的目标？
这个判断将决定你的JSON输出中`coordinate_validity`字段的值 ("Yes" 或 "No")。

---
**思考步骤三：根据前两步的判断，执行最终行动并生成最终边界框**
* 如果“概念”和“坐标”都有效 (Yes, Yes)，请直接采纳参考范例的边界框作为`final_box`。
* 如果“概念”有效但“坐标”无效 (Yes, No)，请以参考范例的物体为“灵感”，在输入图像的全局范围中寻找最相似的物体，并将其作为`final_box`。
* 如果“概念”就无效 (No, ...)，请完全忽略参考范例，在输入图像中独立寻找目标，并将其作为`final_box`。

---
**最终输出格式:**
 "final_box": x,y,w,h
"""

ZERO_SHOT_PROMPT_TEMPLATE = """
角色: 你是一位视觉定位专家。
任务: 在“输入图像”中，严格根据“文本指令”进行定位。
已知信息:
- 文本指令: "{text_prompt}"
- 输入图像
请忽略任何可能的参考信息，只在输入图像中寻找目标。
---
**最终输出格式:**
请只返回最终的边界框坐标。
"final_box": x,y,w,h
"""
'''
ULTIMATE_3WD_MCOT_PROMPT_TEMPLATE = """
角色: 你是一位精通“三支决策”理论的视觉定位专家。
任务: 结合参考范例与输入图像，对目标进行定位。你不仅要给出位置，还要评估决策的确定性。

---
**第一阶段：多维信息评估 (Evidence Collection)**
1. **概念对齐度**: 参考范例中的物体 "{text_prompt}" 与输入图像中的候选目标在语义和形状上是否高度一致？
2. **空间一致性**: 参考框 {rag_box} 在输入图像对应位置的内容是否符合目标特征？
3. **推理冲突检测**: 参考信息与你对输入图像的直观观察是否存在矛盾？

---
**第二阶段：三支决策逻辑 (Decision Logic)**
请基于上述评估，将你的决策归入以下三类之一：

1. **【接受 (Accept)】 - 正域决策**
   - 触发条件：证据充分且自洽（如概念和坐标均有效）。
   - 行动：直接给出精确的 `final_box`。
   - 置信度：High (0.8-1.0)。

2. **【拒绝 (Reject)】 - 负域决策**
   - 触发条件：明确判断输入图像中不存在符合 "{text_prompt}" 描述的目标。
   - 行动：将 `final_box` 设为 null。
   - 置信度：Low (0.0-0.2)。

3. **【延迟 (Defer/Refine)】 - 边界域决策**
   - 触发条件：存在歧义（如：有多个相似目标、目标极其模糊、或参考信息严重干扰直觉）。
   - 行动：不给出单一确定框，而是提供 `candidate_boxes`（多个备选）或标记为 "Uncertain"。
   - 置信度：Middle (0.3-0.7)。

---
**最终输出格式 (JSON):**
{
  "thought_process": {
    "evidence_analysis": "...",
    "decision_type": "Accept/Reject/Defer",
    "confidence_score": 0.0-1.0
  },
  "decision_result": {
    "final_box": "x,y,w,h" (仅在Accept时填写，否则为null),
    "candidate_boxes": ["x,y,w,h", "..."] (仅在Defer时填写多选),
    "action_required": "None/Skip/Re-scan" (下一步指令)
  }
}
"""

ZERO_SHOT_3WD_PROMPT_TEMPLATE = """
角色: 视觉定位专家（三支决策模式）。
任务: 定位 "{text_prompt}"。
策略：
- 如果确定目标位置，输出坐标。
- 如果确定无目标，输出 null。
- 如果不确定，输出所有可能的候选坐标并标注为 "Uncertain"。

---
**最终输出格式 (JSON):**
{
  "decision": "Accept/Reject/Defer",
  "confidence": 0.0-1.0,
  "final_box": "x,y,w,h" or null,
  "uncertain_candidates": []
}
"""