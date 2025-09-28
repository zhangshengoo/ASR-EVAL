"""
热词测试数据集管理器
专门处理热词测试数据
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.models import TestItem
from core.enums import TestDatasetType


@dataclass
class HotwordItem:
    """热词测试项"""
    audio_path: str
    reference_text: str
    hotwords: List[str]
    scene: str
    expected_accuracy: Optional[float] = None


class HotwordDatasetManager:
    """热词测试数据集管理器"""

    def __init__(self, base_path: str = "datasets/热词测试"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)

    def load_hotword_dataset(self, scene: str = None) -> List[TestItem]:
        """加载热词测试数据集"""
        test_items = []

        # 获取场景目录
        scenes_to_load = [scene] if scene else self._get_scenes()

        for current_scene in scenes_to_load:
            scene_path = self.base_path / current_scene
            if not scene_path.exists():
                self.logger.warning(f"场景目录不存在: {scene_path}")
                continue

            # 加载场景数据
            items = self._load_scene_data(scene_path, current_scene)
            test_items.extend(items)

        self.logger.info(f"加载热词测试数据: {len(test_items)} 个测试项")
        return test_items

    def _get_scenes(self) -> List[str]:
        """获取所有场景"""
        if not self.base_path.exists():
            return []

        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def _load_scene_data(self, scene_path: Path, scene_name: str) -> List[TestItem]:
        """加载场景数据"""
        test_items = []

        # 查找.list文件
        list_file = scene_path / f"{scene_name}.list"
        if not list_file.exists():
            self.logger.warning(f"场景文件不存在: {list_file}")
            return test_items

        # 解析.list文件
        try:
            with open(list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # 解析格式: filename | reference_text | hotwords
                parts = line.split('|')
                if len(parts) < 2:
                    self.logger.warning(f"第{line_num}行格式错误: {line}")
                    continue

                filename = parts[0].strip()
                reference_text = parts[1].strip()
                hotwords = []

                # 提取热词（如果有）
                if len(parts) > 2:
                    hotwords = [hw.strip() for hw in parts[2].split(',') if hw.strip()]

                # 构建完整音频路径
                audio_path = scene_path / f"{filename}.wav"
                if not audio_path.exists():
                    # 尝试其他格式
                    for ext in ['.mp3', '.flac', '.m4a']:
                        audio_path = scene_path / f"{filename}{ext}"
                        if audio_path.exists():
                            break
                    else:
                        self.logger.warning(f"音频文件不存在: {audio_path}")
                        continue

                test_items.append(TestItem(
                    audio_path=str(audio_path),
                    reference_text=reference_text,
                    dataset_type=TestDatasetType.ACCENT,  # 使用ACCENT作为热词测试类型
                    metadata={
                        "scene": scene_name,
                        "hotwords": hotwords,
                        "filename": filename,
                        "test_type": "hotword"
                    }
                ))

        except Exception as e:
            self.logger.error(f"加载场景数据失败 {scene_name}: {e}")

        return test_items

    def create_sample_hotword_dataset(self, scene: str = "场景1", sample_count: int = 5):
        """创建示例热词测试数据集"""
        scene_path = self.base_path / scene
        scene_path.mkdir(parents=True, exist_ok=True)

        # 示例热词数据
        sample_data = [
            {
                "filename": "test1",
                "text": "AI帮助很大",
                "hotwords": ["AI", "帮助"]
            },
            {
                "filename": "test2",
                "text": "使用LLM技术",
                "hotwords": ["LLM", "技术"]
            },
            {
                "filename": "test3",
                "text": "语音识别准确",
                "hotwords": ["语音识别", "准确"]
            },
            {
                "filename": "test4",
                "text": "热词测试场景",
                "hotwords": ["热词", "测试", "场景"]
            },
            {
                "filename": "test5",
                "text": "模型评估结果",
                "hotwords": ["模型", "评估", "结果"]
            }
        ]

        # 创建.list文件
        list_file = scene_path / f"{scene}.list"
        with open(list_file, 'w', encoding='utf-8') as f:
            for item in sample_data[:sample_count]:
                hotwords_str = ", ".join(item["hotwords"])
                f.write(f"{item['filename']} | {item['text']} | {hotwords_str}\n")

        # 创建空音频文件（实际使用时需要替换为真实音频）
        for item in sample_data[:sample_count]:
            audio_file = scene_path / f"{item['filename']}.wav"
            audio_file.touch()

        self.logger.info(f"已创建示例热词数据集: {scene} ({sample_count} 个样本)")

    def get_hotword_statistics(self, scene: str = None) -> Dict[str, Any]:
        """获取热词统计信息"""
        test_items = self.load_hotword_dataset(scene)

        if not test_items:
            return {"total_items": 0, "scenes": [], "hotwords": []}

        # 统计热词
        all_hotwords = []
        scenes = set()

        for item in test_items:
            hotwords = item.metadata.get("hotwords", [])
            all_hotwords.extend(hotwords)
            scenes.add(item.metadata.get("scene", "unknown"))

        hotword_counts = {}
        for hw in all_hotwords:
            hotword_counts[hw] = hotword_counts.get(hw, 0) + 1

        return {
            "total_items": len(test_items),
            "scenes": list(scenes),
            "hotwords": list(set(all_hotwords)),
            "hotword_counts": hotword_counts,
            "scenes_count": len(scenes)
        }

    def validate_dataset(self, scene: str = None) -> Dict[str, Any]:
        """验证数据集完整性"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }

        try:
            test_items = self.load_hotword_dataset(scene)
            validation_result["summary"]["total_items"] = len(test_items)

            # 检查音频文件是否存在
            missing_files = []
            valid_files = 0

            for item in test_items:
                if not os.path.exists(item.audio_path):
                    missing_files.append(item.audio_path)
                else:
                    valid_files += 1

            if missing_files:
                validation_result["warnings"].append(f"缺失音频文件: {len(missing_files)}个")
                validation_result["missing_files"] = missing_files

            validation_result["summary"]["valid_files"] = valid_files
            validation_result["summary"]["missing_files"] = len(missing_files)

            if len(test_items) == 0:
                validation_result["valid"] = False
                validation_result["errors"].append("没有找到有效的测试数据")

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))

        return validation_result


# 热词测试专用评估器
class HotwordEvaluator:
    """热词测试评估器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_hotword_performance(self, test_results: List[Any], test_items: List[TestItem]) -> Dict[str, Any]:
        """评估热词识别性能"""
        hotword_stats = {}

        # 获取所有热词
        all_hotwords = set()
        for item in test_items:
            all_hotwords.update(item.metadata.get("hotwords", []))

        # 初始化统计
        for hw in all_hotwords:
            hotword_stats[hw] = {
                "total_occurrences": 0,
                "correct_recognitions": 0,
                "accuracy": 0.0,
                "test_cases": []
            }

        # 统计每个热词的识别情况
        for result, item in zip(test_results, test_items):
            hotwords = item.metadata.get("hotwords", [])
            predicted_text = result.predicted_text.lower()
            reference_text = item.reference_text.lower()

            for hw in hotwords:
                hw_lower = hw.lower()

                # 统计热词在参考文本中的出现次数
                ref_count = reference_text.count(hw_lower)
                pred_count = predicted_text.count(hw_lower)

                hotword_stats[hw]["total_occurrences"] += ref_count
                hotword_stats[hw]["correct_recognitions"] += pred_count

                # 记录测试用例
                hotword_stats[hw]["test_cases"].append({
                    "audio_path": item.audio_path,
                    "reference": item.reference_text,
                    "predicted": predicted_text,
                    "expected_count": ref_count,
                    "predicted_count": pred_count,
                    "correct": pred_count >= ref_count
                })

        # 计算准确率
        for hw, stats in hotword_stats.items():
            if stats["total_occurrences"] > 0:
                stats["accuracy"] = stats["correct_recognitions"] / stats["total_occurrences"]

        return {
            "hotword_performance": hotword_stats,
            "overall_summary": {
                "total_hotwords": len(hotword_stats),
                "average_accuracy": sum(s["accuracy"] for s in hotword_stats.values()) / len(hotword_stats) if hotword_stats else 0,
                "total_test_cases": len(test_results)
            }
        }