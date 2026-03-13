'''import logging
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scripts.V5.configs.prompts import ULTIMATE_MCOT_PROMPT_TEMPLATE, ZERO_SHOT_PROMPT_TEMPLATE


class ReasoningEngine:
    def __init__(self, config: Dict[str, Any]):
        engine_config = config['model_configs']['reasoning_engine']
        model_path = engine_config['model_path']
        logging.info("正在初始化推理引擎...")
        logging.info(f" - 从 '{model_path}' 加载Qwen2.5-VL模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            load_in_8bit=engine_config.get('use_int8', False),
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if engine_config.get('use_flash_attention_2', False) else "eager",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        logging.info("✅ 推理引擎初始化完成。")

    def _parse_box_from_string(self, raw_output: str) -> Optional[List[int]]:
        match = re.search(r'(\d+[\.,]?\d*,\s*\d+[\.,]?\d*,\s*\d+[\.,]?\d*,\s*\d+[\.,]?\d*)', raw_output)
        if match:
            try:
                cleaned_str = match.group(1).replace('，', ',')
                return [int(float(n.strip())) for n in cleaned_str.split(',')]
            except (ValueError, IndexError):
                return None
        return None

    def _run_vlm(self, prompt: str, images: List[Image.Image]) -> Optional[List[int]]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)

        raw_output = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        logging.info(f"--- [VLM原始输出] ---\n{raw_output}\n-------------------------")

        return self._parse_box_from_string(raw_output)

    def run_ultimate_mcot(self, image: Image.Image, text_prompt: str, reference_example: Dict) -> Optional[List[int]]:
        logging.info("执行带上下文的VLM推理 (MCoT)...")
        ref_image = Image.open(reference_example['data']['image_path']).convert("RGB")
        prompt = ULTIMATE_MCOT_PROMPT_TEMPLATE.format(
            text_prompt=text_prompt,
            rag_box=reference_example['data']['box']
        )
        return self._run_vlm(prompt, [image, ref_image])

    def run_zero_shot_vlm(self, image: Image.Image, text_prompt: str) -> Optional[List[int]]:
        logging.info("执行零样本VLM定位...")
        prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(text_prompt=text_prompt)
        return self._run_vlm(prompt, [image])
'''
import logging
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

# 确保路径指向你定义了 3WD Prompt 的地方
from scripts.V9.configs.prompts import ULTIMATE_3WD_MCOT_PROMPT_TEMPLATE, ZERO_SHOT_3WD_PROMPT_TEMPLATE

class ReasoningEngine:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化推理引擎。
        使用 4-bit 量化以节省显存，支持 CPU 卸载。
        """
        model_path = config['model_path']
        logging.info("正在初始化推理引擎 (4-bit 显存优化版)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 设定三支决策阈值
        self.alpha = config.get('alpha', 0.7)
        self.beta = config.get('beta', 0.3)

        # 核心优化：使用 4-bit 量化配置
        # 这将显存需求降低约 50%
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # 允许显存不足时卸载到内存
        )

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # --- 针对 16G 内存的救命参数 ---
                low_cpu_mem_usage=True,
                offload_folder="offload_cache",  # 在本地创建一个文件夹缓存放不下的权重
                offload_state_dict=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            logging.info(f"✅ 推理引擎加载成功。当前模式: 4-bit NF4。阈值: alpha={self.alpha}, beta={self.beta}")
        except Exception as e:
            logging.error(f"❌ 模型加载失败: {e}")
            raise e

    def _parse_3wd_output(self, raw_output: str) -> Dict[str, Any]:
        """解析模型输出的 JSON 字符串"""
        try:
            # 兼容模型可能输出的 markdown 代码块
            json_str = re.search(r'\{.*\}', raw_output, re.DOTALL).group()
            return json.loads(json_str)
        except Exception:
            logging.warning("JSON 解析失败，尝试从文本中恢复坐标...")
            box = self._parse_box_from_string(raw_output)
            return {
                "decision": "Accept" if box else "Reject",
                "confidence": 0.5,
                "final_box": box
            }

    def _parse_box_from_string(self, box_str: Any) -> Optional[List[int]]:
        """辅助方法：从字符串中提取 [x,y,w,h] 或 [x1,y1,x2,y2]"""
        if not box_str or box_str == "null": return None
        if isinstance(box_str, list): return box_str

        # 正则匹配 4 个数字
        match = re.search(r'(\d+[\.,]?\d*,\s*\d+[\.,]?\d*,\s*\d+[\.,]?\d*,\s*\d+[\.,]?\d*)', str(box_str))
        if match:
            try:
                cleaned_str = match.group(1).replace('，', ',')
                return [int(float(n.strip())) for n in cleaned_str.split(',')]
            except:
                return None
        return None

    def _run_vlm_3wd(self, prompt: str, images: List[Image.Image]) -> Dict[str, Any]:
        """运行 VLM 并获取三支决策结果"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)

        raw_output = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        logging.info(f"--- [VLM 3WD 输出] ---\n{raw_output}\n-------------------------")

        return self._parse_3wd_output(raw_output)

    def run_ultimate_mcot(self, image_path: Any, text_prompt: str, rag_box: Optional[List] = None) -> Tuple[Optional[List[int]], str]:
        """
        执行带上下文的三支决策推理。
        """
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"无法打开图像 {image_path}: {e}")
            return None, "Reject"

        # 构造提示词
        prompt = ULTIMATE_3WD_MCOT_PROMPT_TEMPLATE.format(
            text_prompt=text_prompt,
            rag_box=str(rag_box) if rag_box is not None else "None"
        )

        res = self._run_vlm_3wd(prompt, [image])

        # 解析决策逻辑 (优先匹配 decision_type)
        decision = res.get("decision_type", res.get("decision", "Defer"))
        confidence = float(res.get("confidence_score", res.get("confidence", 0.5)))

        if confidence >= self.alpha:
            final_box_data = res.get("decision_result", {}).get("final_box") if isinstance(res.get("decision_result"), dict) else res.get("final_box")
            return self._parse_box_from_string(final_box_data), "Accept"
        elif confidence < self.beta:
            return None, "Reject"
        else:
            # 边界域处理
            candidates = res.get("decision_result", {}).get("candidate_boxes", [])
            box = self._parse_box_from_string(candidates[0] if candidates else res.get("final_box"))
            return box, "Defer"