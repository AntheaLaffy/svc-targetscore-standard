import json
import os
import math
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union


class SVCScoringSystem:
    """SVC模型评分系统（基于SVC初学者指南）"""

    def __init__(self, config_file="svc_scoring_config.json"):
        self.config_file = config_file
        self.command_parser = CommandParser()
        self.current_session = None
        self.load_system_config()

    def load_system_config(self):
        """加载系统配置文件"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.system_config = json.load(f)
        else:
            # 创建默认SVC系统配置
            self.system_config = {
                "command_priority": 1,
                "log_dir": "svc_scoring_logs",
                "auto_save": True,
                "scoring_rules": {
                    "dimensions": {
                        "timbre_similarity": {
                            "weight": 0.30,
                            "sub_dimensions": {
                                "pitch_match": "基频分布匹配度",
                                "formant_similarity": "共振峰结构相似度",
                                "spectral_balance": "谐波能量分布匹配"
                            }
                        },
                        "style_similarity": {
                            "weight": 0.20,
                            "sub_dimensions": {
                                "vibrato_consistency": "颤音特征一致性",
                                "dynamics_handling": "动态处理相似度"
                            }
                        },
                        "audio_quality": {
                            "weight": 0.25,
                            "sub_dimensions": {
                                "artifact_control": "噪声与伪影控制",
                                "spectral_smoothness": "频谱连续性",
                                "phase_coherence": "相位一致性"
                            }
                        },
                        "naturalness": {
                            "weight": 0.25,
                            "sub_dimensions": {
                                "articulation_clarity": "发音清晰度",
                                "breath_naturalness": "气息自然度"
                            }
                        }
                    },
                    "suppression_configs": {
                        "strict": {"left_valve": 0.2, "right_valve": 0.8},
                        "standard": {"left_valve": 0.1, "right_valve": 0.7},
                        "lenient": {"left_valve": 0.0, "right_valve": 0.6}
                    },
                    "default_suppression": "standard"
                },
                "song_library": {
                    "CN": {
                        "regular": [
                            {"name": "CN1(不怕，可爱积极)", "max_score": 9},
                            {"name": "CN2(嗵嗵1，节奏低声)", "max_score": 9},
                            {"name": "CN3(如果有来生，悠扬开阔)", "max_score": 9},
                            {"name": "CN4(嗵嗵2，律动婉转)", "max_score": 9},
                            {"name": "CN5(勾指起誓，可爱律动)", "max_score": 9},
                            {"name": "CN6(离音，坚定深情)", "max_score": 9},
                            {"name": "CN7(其实都没有，深情律动)", "max_score": 9},
                            {"name": "CN8(夜空星，力量高音)", "max_score": 9},
                            {"name": "CN9(晴天，温柔节奏)", "max_score": 9}
                        ],
                        "special": [
                            {"name": "CN-SP(岁月神偷，超低音)", "max_score": 4}
                        ],
                        "extra": [
                            {"name": "CN-EX1(浸春芜，特殊唱法，高音，劣质推理源)", "max_score": 5},
                            {"name": "CN-EX2(左手指月，逆天高音)", "max_score": 5},
                            {"name": "CN-EX3(世末歌者，高音转音)", "max_score": 5}
                        ]
                    },
                    "JP": {
                        "regular": [
                            {"name": "JP1(未闻花名，律动积极)", "max_score": 8},
                            {"name": "JP2(任性，温柔舒适)", "max_score": 8},
                            {"name": "JP3(摇篮曲，幼音气音)", "max_score": 8},
                            {"name": "JP4(猫病，节奏律动)", "max_score": 8},
                            {"name": "JP5(罪恶王冠，深情力量)", "max_score": 8},
                            {"name": "JP6(给我翅膀，高音长音)", "max_score": 8},
                            {"name": "JP7(少年之心，中性音，合唱劣音)", "max_score": 8},
                            {"name": "JP8(多娜多娜，可爱激昂)", "max_score": 8},
                            {"name": "JP9(mottai，语速咬字)", "max_score": 8},
                            {"name": "JP10(giligili爱，激昂高音)", "max_score": 8},
                            {"name": "JP11(没什么大不了，低音)", "max_score": 8}
                        ],
                        "extra": [
                            {"name": "JP-EX1(tellYW林莉奈，逆天高音长音)", "max_score": 6},
                            {"name": "JP-EX2(never，炸裂高亢高音)", "max_score": 6}
                        ]
                    }
                }
            }
            self.save_system_config()

        # 构建歌曲列表
        self.songs = self.get_all_songs()

    def save_system_config(self):
        """保存系统配置文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.system_config, f, ensure_ascii=False, indent=2)

    def get_all_songs(self) -> Dict[str, List[str]]:
        """获取所有歌曲的扁平列表"""
        all_songs = {"CN": [], "JP": []}

        for lang in ["CN", "JP"]:
            lang_data = self.system_config["song_library"][lang]
            for category in ["regular", "special", "extra"]:
                if category in lang_data:
                    for song in lang_data[category]:
                        all_songs[lang].append(song["name"])

        return all_songs

    def get_song_max_score(self, song_name: str, language: str) -> float:
        """获取歌曲的最大分数"""
        lang_data = self.system_config["song_library"][language]

        for category in ["regular", "special", "extra"]:
            if category in lang_data:
                for song in lang_data[category]:
                    if song["name"] == song_name:
                        return song["max_score"]

        # 默认值
        if language == "CN":
            return 9 if "EX" not in song_name and "SP" not in song_name else (5 if "EX" in song_name else 4)
        else:
            return 8 if "EX" not in song_name else 6

    def calculate_suppression_parameters(self, left_valve: float, right_valve: float) -> Tuple[float, float]:
        """根据左右阀门计算θ和k值"""
        theta = (left_valve + right_valve) / 2.0
        k = 4.0 / (right_valve - left_valve)
        return theta, k

    def calculate_suppression_factor(self, worst_score: float, left_valve: float = 0.1,
                                     right_valve: float = 0.7) -> float:
        """
        计算压制因子

        参数:
        worst_score: 最差维度的原始分数(0-10分)
        left_valve: 压制区间左阀门(归一化)
        right_valve: 压制区间右阀门(归一化)

        返回:
        压制因子(0-1)
        """
        # 计算θ和k
        theta, k = self.calculate_suppression_parameters(left_valve, right_valve)

        # 归一化处理
        x = worst_score / 10.0

        # 标准sigmoid压制
        exponent = -k * (x - theta)
        suppression = 1 / (1 + math.exp(exponent))

        return suppression

    def calculate_final_weight(self, scores_dict: Dict, config_name: str = 'standard') -> Dict:
        """
        计算最终权重分数(0-100分)
        基于SVC文档5.6节的完整评分流程
        """
        # 获取配置
        config = self.system_config["scoring_rules"]["suppression_configs"][config_name]

        # 1. 计算各维度基础分（子维度平均）
        timbre_score = (
                               scores_dict['pitch_match'] +
                               scores_dict['formant_similarity'] +
                               scores_dict['spectral_balance']
                       ) / 3.0

        style_score = (
                              scores_dict['vibrato_consistency'] +
                              scores_dict['dynamics_handling']
                      ) / 2.0

        quality_score = (
                                scores_dict['artifact_control'] +
                                scores_dict['spectral_smoothness'] +
                                scores_dict['phase_coherence']
                        ) / 3.0

        natural_score = (
                                scores_dict['articulation_clarity'] +
                                scores_dict['breath_naturalness']
                        ) / 2.0

        # 2. 计算加权基础分(0-10分)
        weights = self.system_config["scoring_rules"]["dimensions"]
        base_score = (
                timbre_score * weights["timbre_similarity"]["weight"] +
                style_score * weights["style_similarity"]["weight"] +
                quality_score * weights["audio_quality"]["weight"] +
                natural_score * weights["naturalness"]["weight"]
        )

        # 3. 找到最差维度分
        worst_score = min(timbre_score, style_score, quality_score, natural_score)

        # 4. 计算压制因子
        suppression = self.calculate_suppression_factor(worst_score, **config)

        # 5. 计算最终权重(0-100分)
        final_weight = base_score * 10 * suppression

        return {
            'config_used': config_name,
            'dimension_scores': {
                'timbre': round(timbre_score, 2),
                'style': round(style_score, 2),
                'quality': round(quality_score, 2),
                'natural': round(natural_score, 2)
            },
            'base_score': round(base_score, 2),
            'worst_score': round(worst_score, 2),
            'suppression_factor': round(suppression, 3),
            'final_weight': round(final_weight, 1)
        }

    def calculate_song_score(self, scores_dict: Dict, song_name: str, language: str,
                             suppression_config: str = 'standard') -> Dict:
        """
        计算单曲总分
        基于SVC算法：最终权重 × 歌曲满分 / 100
        """
        # 计算最终权重（0-100分）
        weight_result = self.calculate_final_weight(scores_dict, suppression_config)

        # 获取歌曲满分
        max_score = self.get_song_max_score(song_name, language)

        # 计算单曲总分
        song_total_score = weight_result['final_weight'] * max_score / 100.0

        # 添加到结果中
        weight_result['song_name'] = song_name
        weight_result['language'] = language
        weight_result['max_score'] = max_score
        weight_result['song_total_score'] = round(song_total_score, 2)

        return weight_result

    def calculate_language_total(self, scores: List[Dict], language: str) -> Dict:
        """计算语种总分"""
        language_scores = [s for s in scores if s["language"] == language]
        total = sum(s["song_total_score"] for s in language_scores)

        # 判断合格和优秀
        passed = total >= 60
        excellent = total >= 80

        return {
            "total_score": round(total, 2),
            "passed": passed,
            "excellent": excellent,
            "song_count": len(language_scores)
        }

    def calculate_overall(self, cn_total: float, jp_total: float) -> Dict:
        """计算总体评分"""
        overall_total = cn_total + jp_total
        passed = overall_total >= 160

        return {
            "total_score": round(overall_total, 2),
            "passed": passed,
            "cn_score": cn_total,
            "jp_score": jp_total
        }

    def modify_song_library(self, modifications: Dict):
        """修改曲库"""
        for lang, categories in modifications.items():
            if lang in self.system_config["song_library"]:
                for category, songs in categories.items():
                    if category in self.system_config["song_library"][lang]:
                        self.system_config["song_library"][lang][category] = songs

        self.songs = self.get_all_songs()
        self.save_system_config()

    def modify_scoring_rules(self, modifications: Dict):
        """修改评分规则"""
        rules = self.system_config["scoring_rules"]

        # 更新维度权重
        if "dimensions" in modifications:
            dimensions_mods = modifications["dimensions"]
            for dimension, settings in dimensions_mods.items():
                if dimension in rules["dimensions"]:
                    if "weight" in settings:
                        rules["dimensions"][dimension]["weight"] = settings["weight"]
                    if "sub_dimensions" in settings:
                        rules["dimensions"][dimension]["sub_dimensions"].update(settings["sub_dimensions"])

        # 更新压制配置
        if "suppression_configs" in modifications:
            suppression_mods = modifications["suppression_configs"]
            rules["suppression_configs"].update(suppression_mods)

        self.save_system_config()


# 保留原有的CommandParser类（与之前相同）
class CommandParser:
    """命令解析器"""

    def __init__(self):
        self.mode_options = {'c', 'v', 'f', 'o'}
        self.target_options = {'m', 's', 'r', 'l', 'c', 'h'}
        self.input_options = {'i', 'f', 'n', 'p', 'r'}

        # 默认参数
        self.default_params = {
            'c': 'y',  # 确认模式默认是
            'v': 'n',  # 详细模式默认否
            'f': 'n',  # 强制模式默认否
        }

    def parse_command(self, input_str: str) -> Tuple[str, Dict[str, Any]]:
        """解析命令输入"""
        input_str = input_str.strip()
        if not input_str:
            return "", {}

        # 分割命令和参数
        parts = input_str.split()
        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        # 解析选项
        options = self._parse_options(args)

        return command, options

    def _parse_options(self, args: List[str]) -> Dict[str, Any]:
        """解析选项参数"""
        options = {'mode': {}, 'target': {}, 'input': {}}
        current_section = None
        current_option = None

        i = 0
        while i < len(args):
            arg = args[i]

            # 检查是否是选项
            if arg.startswith('-'):
                option = arg[1:]

                # 确定选项类别
                if option in self.mode_options:
                    current_section = 'mode'
                elif option in self.target_options:
                    current_section = 'target'
                elif option in self.input_options:
                    current_section = 'input'
                else:
                    # 未知选项，跳过
                    i += 1
                    continue

                current_option = option

                # 检查是否有参数
                if i + 1 < len(args) and not args[i + 1].startswith('-'):
                    # 有参数
                    param = args[i + 1]
                    options[current_section][current_option] = param
                    i += 2
                else:
                    # 无参数，使用默认值
                    if current_option in self.default_params:
                        options[current_section][current_option] = self.default_params[current_option]
                    else:
                        options[current_section][current_option] = True
                    i += 1
            else:
                # 非选项参数，忽略或根据上下文处理
                i += 1

        return options


class SVCScoringSession:
    """SVC评分会话"""

    def __init__(self, model_name: str, epoch: int, log_dir: str):
        self.model_name = model_name
        self.epoch = epoch
        self.session_id = f"{model_name}_{epoch}"
        self.log_dir = log_dir

        # 会话状态
        self.current_step = 0
        self.scores = []
        self.history = []  # 用于撤销/重做
        self.future = []  # 重做栈

        # 文件路径
        self.permanent_log = os.path.join(log_dir, f"{self.session_id}_permanent.json")
        self.temp_log = os.path.join(log_dir, f"{self.session_id}_temp.json")

        self.load_session()

    def load_session(self):
        """加载会话状态"""
        # 优先加载临时日志
        if os.path.exists(self.temp_log):
            with open(self.temp_log, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.current_step = data.get("current_step", 0)
                self.scores = data.get("scores", [])
                self.history = data.get("history", [])
                self.future = data.get("future", [])
        elif os.path.exists(self.permanent_log):
            with open(self.permanent_log, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.current_step = data.get("current_step", 0)
                self.scores = data.get("scores", [])

    def save_session(self, permanent=False):
        """保存会话状态"""
        data = {
            "model_name": self.model_name,
            "epoch": self.epoch,
            "current_step": self.current_step,
            "scores": self.scores,
            "history": self.history,
            "future": self.future,
            "last_updated": datetime.now().isoformat()
        }

        # 总是保存到临时日志
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.temp_log, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 如果要求永久保存，则同步到永久日志
        if permanent:
            with open(self.permanent_log, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def add_score(self, scoring_system, song_name: str, language: str,
                  pitch_match: float, formant_similarity: float, spectral_balance: float,
                  vibrato_consistency: float, dynamics_handling: float,
                  artifact_control: float, spectral_smoothness: float, phase_coherence: float,
                  articulation_clarity: float, breath_naturalness: float,
                  suppression_config: str = 'standard'):
        """添加SVC评分"""
        # 保存当前状态到历史
        self.history.append({
            "step": self.current_step,
            "scores": self.scores.copy()
        })

        # 构建评分字典
        scores_dict = {
            'pitch_match': pitch_match,
            'formant_similarity': formant_similarity,
            'spectral_balance': spectral_balance,
            'vibrato_consistency': vibrato_consistency,
            'dynamics_handling': dynamics_handling,
            'artifact_control': artifact_control,
            'spectral_smoothness': spectral_smoothness,
            'phase_coherence': phase_coherence,
            'articulation_clarity': articulation_clarity,
            'breath_naturalness': breath_naturalness
        }

        # 计算总分
        score_result = scoring_system.calculate_song_score(scores_dict, song_name, language, suppression_config)

        # 保存完整结果
        score_data = {
            "song_name": song_name,
            "language": language,
            "scores_dict": scores_dict,
            "suppression_config": suppression_config,
            "step": self.current_step,
            "timestamp": datetime.now().isoformat()
        }
        score_data.update(score_result)

        self.scores.append(score_data)
        self.current_step += 1
        self.future.clear()  # 新的操作清空重做栈
        self.save_session()

    def undo(self):
        """撤销上一步"""
        if self.history:
            # 保存当前状态到重做栈
            self.future.append({
                "step": self.current_step,
                "scores": self.scores.copy()
            })

            # 恢复到上一个状态
            last_state = self.history.pop()
            self.current_step = last_state["step"]
            self.scores = last_state["scores"]
            self.save_session()
            return True
        return False

    def redo(self):
        """重做"""
        if self.future:
            # 保存当前状态到历史
            self.history.append({
                "step": self.current_step,
                "scores": self.scores.copy()
            })

            # 恢复到重做状态
            next_state = self.future.pop()
            self.current_step = next_state["step"]
            self.scores = next_state["scores"]
            self.save_session()
            return True
        return False

    def modify_step(self, scoring_system, step_index: int, new_scores: Dict, suppression_config: str = None):
        """修改特定步骤的评分"""
        if 0 <= step_index < len(self.scores):
            # 保存当前状态到历史
            self.history.append({
                "step": self.current_step,
                "scores": self.scores.copy()
            })

            # 更新评分
            song_data = self.scores[step_index]
            current_scores = song_data["scores_dict"].copy()
            current_scores.update(new_scores)

            config = suppression_config or song_data.get("suppression_config", "standard")

            # 重新计算总分
            score_result = scoring_system.calculate_song_score(
                current_scores,
                song_data["song_name"],
                song_data["language"],
                config
            )

            # 更新数据
            self.scores[step_index]["scores_dict"] = current_scores
            self.scores[step_index]["suppression_config"] = config
            self.scores[step_index].update(score_result)
            self.future.clear()
            self.save_session()
            return True
        return False

    def clear_scores(self):
        """清空评分数据"""
        self.history.append({
            "step": self.current_step,
            "scores": self.scores.copy()
        })

        self.scores = []
        self.current_step = 0
        self.future.clear()
        self.save_session()

    def delete_session(self):
        """删除会话"""
        if os.path.exists(self.temp_log):
            os.remove(self.temp_log)
        if os.path.exists(self.permanent_log):
            os.remove(self.permanent_log)


class SVCCommandProcessor:
    """SVC命令处理器"""

    def __init__(self, scoring_system):
        self.scoring_system = scoring_system
        self.command_parser = CommandParser()

        # 命令帮助信息
        self.command_help = {
            "help": "显示帮助信息",
            "list": "显示评分总览",
            "print": "打印系统信息",
            "change": "修改系统配置",
            "clear": "清空数据",
            "delete": "删除数据",
            "undo": "撤销操作",
            "redo": "重做操作",
            "save": "保存数据",
            "quit": "退出系统"
        }

    def execute_command(self, command: str, options: Dict, session: SVCScoringSession) -> str:
        """执行命令"""
        if command == "help":
            return self.help_command(options)
        elif command == "list":
            return self.list_command(options, session)
        elif command == "print":
            return self.print_command(options, session)
        elif command == "change":
            return self.change_command(options, session)
        elif command == "clear":
            return self.clear_command(options, session)
        elif command == "delete":
            return self.delete_command(options, session)
        elif command == "undo":
            return self.undo_command(options, session)
        elif command == "redo":
            return self.redo_command(options, session)
        elif command == "save":
            return self.save_command(options, session)
        elif command == "quit":
            return "quit"
        else:
            return f"未知命令: {command}"

    def help_command(self, options: Dict) -> str:
        """显示帮助信息"""
        mode = options.get('mode', {})
        target = options.get('target', {})

        verbose = mode.get('v', 'n') == 'y'

        if 'c' in target:  # 指定命令
            command_name = target['c']
            if command_name in self.command_help:
                return f"{command_name}: {self.command_help[command_name]}"
            else:
                return f"未知命令: {command_name}"

        # 显示所有命令
        help_text = "SVC模型评分系统命令帮助:\n\n"

        for cmd, desc in self.command_help.items():
            help_text += f"  {cmd}: {desc}\n"

        if verbose:
            help_text += "\n详细选项说明:\n"
            help_text += "模式选项 (-):\n"
            help_text += "  -c: 确认模式 (y/n)\n"
            help_text += "  -v: 详细模式 (y/n)\n"
            help_text += "  -f: 强制模式 (y/n)\n"
            help_text += "  -o: 输出模式 (文件路径)\n\n"

            help_text += "目标选项 (-):\n"
            help_text += "  -m: 模型相关\n"
            help_text += "  -s: 步骤/歌曲相关\n"
            help_text += "  -r: 规则相关\n"
            help_text += "  -l: 曲库相关\n"
            help_text += "  -c: 配置相关\n"
            help_text += "  -h: 历史相关\n\n"

            help_text += "输入选项 (-):\n"
            help_text += "  -i: 输入内容\n"
            help_text += "  -f: 文件输入\n"
            help_text += "  -n: 数字参数\n"
            help_text += "  -p: 路径参数\n"
            help_text += "  -r: 范围参数\n"

        return help_text

    def list_command(self, options: Dict, session: SVCScoringSession) -> str:
        """显示评分列表"""
        target = options.get('target', {})

        # 检查是否指定了特定模型
        if 'm' in target:
            model_spec = target['m']
            return self._list_model_scores(model_spec)

        # 显示当前会话的评分
        if not session or not session.scores:
            return "暂无评分数据"

        return self._format_session_scores(session)

    def _list_model_scores(self, model_spec: str) -> str:
        """显示指定模型的评分"""
        log_dir = self.scoring_system.system_config["log_dir"]
        try:
            model_name, epoch_str = model_spec.rsplit("_", 1)
            epoch = int(epoch_str)
            target_session = SVCScoringSession(model_name, epoch, log_dir)
            return self._format_session_scores(target_session)
        except (ValueError, IndexError):
            return f"无效的模型规格: {model_spec}"

    def _format_session_scores(self, session: SVCScoringSession) -> str:
        """格式化会话评分输出"""
        if not session.scores:
            return f"模型 {session.session_id} 暂无评分数据"

        cn_scores = [s for s in session.scores if s["language"] == "CN"]
        jp_scores = [s for s in session.scores if s["language"] == "JP"]

        cn_total = sum(s["song_total_score"] for s in cn_scores)
        jp_total = sum(s["song_total_score"] for s in jp_scores)
        overall_total = cn_total + jp_total

        result = f"模型: {session.model_name} 轮数: {session.epoch}\n"
        result += f"当前进度: {len(session.scores)} 首歌曲\n\n"

        result += "CN歌曲评分:\n"
        for score in cn_scores:
            result += f"  {score['song_name']}: 权重{score['final_weight']} → 总分{score['song_total_score']}\n"

        cn_stats = self.scoring_system.calculate_language_total(session.scores, "CN")
        result += f"CN总分: {cn_total:.2f} {'(合格)' if cn_stats['passed'] else '(不合格)'} {'(优秀)' if cn_stats['excellent'] else ''}\n\n"

        result += "JP歌曲评分:\n"
        for score in jp_scores:
            result += f"  {score['song_name']}: 权重{score['final_weight']} → 总分{score['song_total_score']}\n"

        jp_stats = self.scoring_system.calculate_language_total(session.scores, "JP")
        result += f"JP总分: {jp_total:.2f} {'(合格)' if jp_stats['passed'] else '(不合格)'} {'(优秀)' if jp_stats['excellent'] else ''}\n\n"

        overall_stats = self.scoring_system.calculate_overall(cn_total, jp_total)
        result += f"总体评分: {overall_total:.2f} {'(通过)' if overall_stats['passed'] else '(不通过)'}"

        return result

    def print_command(self, options: Dict, session: SVCScoringSession) -> str:
        """打印信息命令"""
        target = options.get('target', {})
        mode = options.get('mode', {})

        subject = target.get('s', '')
        output_to_file = mode.get('o', False)

        result = ""
        filename = None

        if subject == "song_library" or subject == "l":
            result = self._print_song_library()
            filename = "song_library.txt"
        elif subject == "scoring_rules" or subject == "r":
            result = self._print_scoring_rules()
            filename = "scoring_rules.txt"
        elif subject == "system" or subject == "s":
            result = self._print_system_info()
            filename = "system_info.txt"
        elif subject == "scores" or subject == "m":
            model_spec = target.get('m', '')
            result = self._print_scores(session, model_spec)
            filename = f"scores_{model_spec if model_spec else (session.session_id if session else 'overview')}.txt"
        elif subject == "dimensions" or subject == "d":
            result = self._print_dimensions_info()
            filename = "dimensions_info.txt"
        else:
            result = "请指定打印对象: -s song_library/scoring_rules/system/scores/dimensions"

        # 输出到文件
        if output_to_file and filename and result:
            if isinstance(output_to_file, str) and output_to_file != "True":
                filename = output_to_file

            output_dir = "print_outputs"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(result)

            result += f"\n\n已输出到文件: {filepath}"

        return result

    def _print_song_library(self) -> str:
        """打印曲库信息"""
        library = self.scoring_system.system_config["song_library"]
        result = "=== 曲库信息 ===\n\n"

        for lang, categories in library.items():
            result += f"{lang}歌曲:\n"
            for category, songs in categories.items():
                result += f"  {category}:\n"
                for song in songs:
                    result += f"    - {song['name']} (满分: {song['max_score']})\n"
            result += "\n"

        return result

    def _print_scoring_rules(self) -> str:
        """打印评分规则"""
        rules = self.scoring_system.system_config["scoring_rules"]
        result = "=== SVC评分规则 ===\n\n"

        # 维度权重
        dimensions = rules["dimensions"]
        result += "评分维度及权重:\n"
        for dim_name, dim_config in dimensions.items():
            result += f"  {dim_name}: {dim_config['weight'] * 100}%\n"
            for sub_dim, desc in dim_config['sub_dimensions'].items():
                result += f"    - {sub_dim}: {desc}\n"
            result += "\n"

        # 压制配置
        suppression_configs = rules["suppression_configs"]
        result += "压制算法配置:\n"
        for config_name, config in suppression_configs.items():
            theta, k = self.scoring_system.calculate_suppression_parameters(config['left_valve'], config['right_valve'])
            result += f"  {config_name}: 左阀门={config['left_valve']}, 右阀门={config['right_valve']}, θ={theta:.2f}, k={k:.2f}\n"

        result += f"\n默认压制配置: {rules['default_suppression']}"

        return result

    def _print_dimensions_info(self) -> str:
        """打印维度详细信息"""
        rules = self.scoring_system.system_config["scoring_rules"]
        result = "=== SVC评分维度详解 ===\n\n"

        dimensions = rules["dimensions"]
        for dim_name, dim_config in dimensions.items():
            result += f"{dim_name.upper()} ({dim_config['weight'] * 100}%):\n"
            for sub_dim, desc in dim_config['sub_dimensions'].items():
                result += f"  {sub_dim}: {desc}\n"
            result += "\n"

        return result

    def _print_system_info(self) -> str:
        """打印系统信息"""
        config = self.scoring_system.system_config
        result = "=== SVC评分系统信息 ===\n\n"

        result += f"配置文件: {self.scoring_system.config_file}\n"
        result += f"日志目录: {config['log_dir']}\n"
        result += f"自动保存: {config['auto_save']}\n"
        result += f"命令优先级: {config['command_priority']}\n"

        return result

    def _print_scores(self, session: SVCScoringSession, model_spec: str = None) -> str:
        """打印评分信息"""
        if model_spec:
            return self._list_model_scores(model_spec)
        elif session:
            return self._format_session_scores(session)
        else:
            # 打印所有模型的评分概览
            return self._print_all_scores_overview()

    def _print_all_scores_overview(self) -> str:
        """打印所有模型的评分概览"""
        log_dir = self.scoring_system.system_config["log_dir"]
        result = "=== 所有模型评分概览 ===\n\n"

        if not os.path.exists(log_dir):
            return result + "暂无评分数据"

        sessions = []
        for file in os.listdir(log_dir):
            if file.endswith("_permanent.json"):
                session_id = file.replace("_permanent.json", "")
                sessions.append(session_id)

        if not sessions:
            return result + "暂无评分数据"

        for session_id in sessions:
            try:
                model_name, epoch_str = session_id.rsplit("_", 1)
                epoch = int(epoch_str)
                session = SVCScoringSession(model_name, epoch, log_dir)

                if session.scores:
                    cn_scores = [s for s in session.scores if s["language"] == "CN"]
                    jp_scores = [s for s in session.scores if s["language"] == "JP"]

                    cn_total = sum(s["song_total_score"] for s in cn_scores)
                    jp_total = sum(s["song_total_score"] for s in jp_scores)
                    overall_total = cn_total + jp_total

                    result += f"{session_id}:\n"
                    result += f"  进度: {len(session.scores)}首, CN: {cn_total:.2f}, JP: {jp_total:.2f}, 总体: {overall_total:.2f} {'(通过)' if overall_total >= 160 else ''}\n"

            except Exception as e:
                result += f"{session_id}: 加载失败 - {str(e)}\n"

        return result

    def change_command(self, options: Dict, session: SVCScoringSession) -> str:
        """修改配置命令"""
        mode = options.get('mode', {})
        target = options.get('target', {})
        input_data = options.get('input', {})

        confirm = mode.get('c', 'y') == 'y'
        subject = target.get('s') or target.get('l') or target.get('r') or target.get('c')

        if not subject:
            return "请指定修改对象: -s song_library/scoring_rules/system"

        # 获取修改内容
        modification = input_data.get('i', '')
        if not modification:
            return "请提供修改内容: -i 'JSON数据'"

        try:
            modifications = json.loads(modification)
        except json.JSONDecodeError:
            return "修改内容必须是有效的JSON格式"

        if subject in ["song_library", "l"]:
            if confirm:
                # 显示当前曲库信息
                current = self.scoring_system.system_config["song_library"]
                result = "当前曲库:\n" + json.dumps(current, ensure_ascii=False, indent=2)
                result += "\n\n确认修改? (y/n): "
                return result + "\n[等待确认]"
            else:
                self.scoring_system.modify_song_library(modifications)
                return "曲库修改成功"

        elif subject in ["scoring_rules", "r"]:
            if confirm:
                # 显示当前规则
                current = self.scoring_system.system_config["scoring_rules"]
                result = "当前评分规则:\n" + json.dumps(current, ensure_ascii=False, indent=2)
                result += "\n\n确认修改? (y/n): "
                return result + "\n[等待确认]"
            else:
                self.scoring_system.modify_scoring_rules(modifications)
                return "评分规则修改成功"

        elif subject in ["system", "s"]:
            if confirm:
                current = self.scoring_system.system_config
                result = "当前系统配置:\n" + json.dumps(current, ensure_ascii=False, indent=2)
                result += "\n\n确认修改? (y/n): "
                return result + "\n[等待确认]"
            else:
                self.scoring_system.system_config.update(modifications)
                self.scoring_system.save_system_config()
                return "系统配置修改成功"

        else:
            return f"未知的修改对象: {subject}"

    def clear_command(self, options: Dict, session: SVCScoringSession) -> str:
        """清空数据命令"""
        mode = options.get('mode', {})
        target = options.get('target', {})

        confirm = mode.get('c', 'y') == 'y'
        subject = target.get('s') or target.get('m') or target.get('h')

        if not subject:
            return "请指定清空对象: -s scores/history"

        if not session:
            return "没有活动的评分会话"

        if subject in ["scores", "s"]:
            if confirm:
                return "确认清空所有评分数据? (y/n): \n[等待确认]"
            else:
                session.clear_scores()
                return "评分数据已清空"

        elif subject in ["history", "h"]:
            if confirm:
                return "确认清空操作历史? (y/n): \n[等待确认]"
            else:
                session.history.clear()
                session.future.clear()
                session.save_session()
                return "操作历史已清空"

        else:
            return f"未知的清空对象: {subject}"

    def delete_command(self, options: Dict, session: SVCScoringSession) -> str:
        """删除数据命令"""
        mode = options.get('mode', {})
        target = options.get('target', {})

        confirm = mode.get('c', 'y') == 'y'
        subject = target.get('s') or target.get('m')

        if not subject:
            return "请指定删除对象: -s session"

        if subject in ["session", "s"]:
            if confirm:
                return "确认删除当前评分会话? (y/n): \n[等待确认]"
            else:
                session.delete_session()
                return "评分会话已删除"

        else:
            return f"未知的删除对象: {subject}"

    def undo_command(self, options: Dict, session: SVCScoringSession) -> str:
        """撤销命令"""
        if not session:
            return "没有活动的评分会话"

        steps = 1
        if 'n' in options.get('input', {}):
            try:
                steps = int(options['input']['n'])
            except ValueError:
                pass

        success_count = 0
        for _ in range(steps):
            if session.undo():
                success_count += 1
            else:
                break

        if success_count > 0:
            return f"成功撤销 {success_count} 步操作"
        else:
            return "无法撤销，没有更多历史记录"

    def redo_command(self, options: Dict, session: SVCScoringSession) -> str:
        """重做命令"""
        if not session:
            return "没有活动的评分会话"

        steps = 1
        if 'n' in options.get('input', {}):
            try:
                steps = int(options['input']['n'])
            except ValueError:
                pass

        success_count = 0
        for _ in range(steps):
            if session.redo():
                success_count += 1
            else:
                break

        if success_count > 0:
            return f"成功重做 {success_count} 步操作"
        else:
            return "无法重做，没有更多重做记录"

    def save_command(self, options: Dict, session: SVCScoringSession) -> str:
        """保存命令"""
        if not session:
            return "没有活动的评分会话"

        session.save_session(permanent=True)
        return "已保存到永久日志"


def main():
    """主函数"""
    system = SVCScoringSystem()
    command_processor = SVCCommandProcessor(system)

    print("=== SVC模型评分系统（基于SVC初学者指南）===")

    while True:
        print("\n请选择操作:")
        print("1. 开始新评分")
        print("2. 继续现有评分")
        print("3. 系统管理")
        print("4. 退出系统")

        choice = input("请输入选择 (1-4): ").strip()

        if choice == "1":
            start_new_scoring(system, command_processor)
        elif choice == "2":
            continue_scoring(system, command_processor)
        elif choice == "3":
            system_management(system, command_processor)
        elif choice == "4":
            print("感谢使用SVC评分系统!")
            break
        else:
            print("无效选择，请重新输入")


def start_new_scoring(system: SVCScoringSystem, command_processor: SVCCommandProcessor):
    """开始新评分会话"""
    model_name = input("请输入模型名称: ").strip()
    epoch = input("请输入训练轮数: ").strip()

    try:
        epoch = int(epoch)
    except ValueError:
        print("训练轮数必须是数字!")
        return

    # 创建新会话
    log_dir = system.system_config["log_dir"]
    session = SVCScoringSession(model_name, epoch, log_dir)
    system.current_session = session

    print(f"\n开始为模型 {model_name} (轮数: {epoch}) 进行SVC评分")
    run_scoring_session(system, command_processor, session)


def continue_scoring(system: SVCScoringSystem, command_processor: SVCCommandProcessor):
    """继续现有评分"""
    log_dir = system.system_config["log_dir"]

    # 查找现有的会话文件
    sessions = []
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            if file.endswith("_temp.json"):
                session_id = file.replace("_temp.json", "")
                sessions.append(session_id)

    if not sessions:
        print("没有找到可继续的评分会话")
        return

    print("\n可继续的评分会话:")
    for i, session_id in enumerate(sessions, 1):
        print(f"{i}. {session_id}")

    try:
        choice = int(input("请选择会话 (输入编号): ")) - 1
        if 0 <= choice < len(sessions):
            session_id = sessions[choice]
            model_name, epoch_str = session_id.rsplit("_", 1)
            epoch = int(epoch_str)

            session = SVCScoringSession(model_name, epoch, log_dir)
            system.current_session = session

            print(f"\n继续评分会话: {session_id}")
            run_scoring_session(system, command_processor, session)
        else:
            print("无效选择")
    except (ValueError, IndexError):
        print("输入错误")


def system_management(system: SVCScoringSystem, command_processor: SVCCommandProcessor):
    """系统管理功能"""
    while True:
        print("\n=== SVC系统管理 ===")
        print("1. 查看系统配置")
        print("2. 备份系统配置")
        print("3. 恢复系统配置")
        print("4. 返回主菜单")

        choice = input("请选择: ").strip()

        if choice == "1":
            print("\n" + command_processor._print_system_info())
            print("\n" + command_processor._print_scoring_rules())
            print("\n" + command_processor._print_song_library())
            print("\n" + command_processor._print_dimensions_info())
        elif choice == "2":
            backup_config(system)
        elif choice == "3":
            restore_config(system)
        elif choice == "4":
            break
        else:
            print("无效选择")


def backup_config(system: SVCScoringSystem):
    """备份系统配置"""
    backup_dir = "svc_config_backups"
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"svc_scoring_config_backup_{timestamp}.json")

    shutil.copy2(system.config_file, backup_file)
    print(f"配置已备份到: {backup_file}")


def restore_config(system: SVCScoringSystem):
    """恢复系统配置"""
    backup_dir = "svc_config_backups"
    if not os.path.exists(backup_dir):
        print("没有找到备份目录")
        return

    backups = [f for f in os.listdir(backup_dir) if f.endswith(".json")]
    if not backups:
        print("没有找到备份文件")
        return

    print("\n可用的备份文件:")
    for i, backup in enumerate(backups, 1):
        print(f"{i}. {backup}")

    try:
        choice = int(input("请选择备份文件 (输入编号): ")) - 1
        if 0 <= choice < len(backups):
            backup_file = os.path.join(backup_dir, backups[choice])
            shutil.copy2(backup_file, system.config_file)
            system.load_system_config()
            print("配置已恢复")
        else:
            print("无效选择")
    except ValueError:
        print("输入错误")


def run_scoring_session(system: SVCScoringSystem, command_processor: SVCCommandProcessor, session: SVCScoringSession):
    """运行SVC评分会话"""
    songs = system.songs
    all_songs = [(lang, song) for lang in ["CN", "JP"] for song in songs[lang]]

    while True:
        # 显示当前进度
        current_index = session.current_step
        total_songs = len(all_songs)

        if current_index >= total_songs:
            print("\n所有歌曲评分完成!")
            show_final_results(system, session)
            break

        current_lang, current_song = all_songs[current_index]
        print(f"\n当前进度: {current_index + 1}/{total_songs}")
        print(f"当前歌曲: {current_song} ({current_lang})")
        print("请按以下顺序输入10个维度的评分(0-10分):")
        print("1. 基频分布匹配度 (pitch_match)")
        print("2. 共振峰结构相似度 (formant_similarity)")
        print("3. 谐波能量分布匹配 (spectral_balance)")
        print("4. 颤音特征一致性 (vibrato_consistency)")
        print("5. 动态处理相似度 (dynamics_handling)")
        print("6. 噪声与伪影控制 (artifact_control)")
        print("7. 频谱连续性 (spectral_smoothness)")
        print("8. 相位一致性 (phase_coherence)")
        print("9. 发音清晰度 (articulation_clarity)")
        print("10. 气息自然度 (breath_naturalness)")

        # 获取用户输入
        user_input = input("请输入10个评分 (用空格分隔) 或命令: ").strip()

        # 检查命令优先级
        command_priority = system.system_config.get("command_priority", 1)

        # 检查是否是命令
        command, options = command_processor.command_parser.parse_command(user_input)
        if command and command in command_processor.command_help:
            if command_priority >= 1:  # 命令优先级高于评分
                result = command_processor.execute_command(command, options, session)
                if result == "quit":
                    break
                else:
                    print(result)
                    continue
            else:
                print(f"命令优先级({command_priority})较低，优先处理评分")

        # 处理评分输入
        try:
            scores = [float(x) for x in user_input.split()]
            if len(scores) != 10:
                print("请输入10个数值，对应10个评分维度")
                continue

            # 验证分数范围
            if not all(0 <= score <= 10 for score in scores):
                print("所有分数必须在0-10之间")
                continue

            # 提取各个维度分数
            pitch_match, formant_similarity, spectral_balance, \
                vibrato_consistency, dynamics_handling, artifact_control, \
                spectral_smoothness, phase_coherence, articulation_clarity, \
                breath_naturalness = scores

            # 添加评分（使用标准压制配置）
            session.add_score(
                system, current_song, current_lang,
                pitch_match, formant_similarity, spectral_balance,
                vibrato_consistency, dynamics_handling,
                artifact_control, spectral_smoothness, phase_coherence,
                articulation_clarity, breath_naturalness,
                'standard'
            )

            last_score = session.scores[-1]
            print(f"评分已记录: {current_song}")
            print(
                f"  各维度: 音色{last_score['dimension_scores']['timbre']} 风格{last_score['dimension_scores']['style']} 质量{last_score['dimension_scores']['quality']} 自然{last_score['dimension_scores']['natural']}")
            print(f"  基础分: {last_score['base_score']}, 最差维度: {last_score['worst_score']}")
            print(f"  压制因子: {last_score['suppression_factor']}, 最终权重: {last_score['final_weight']}%")
            print(f"  单曲总分: {last_score['song_total_score']} (满分: {last_score['max_score']})")

        except ValueError:
            print("输入格式错误，请输入10个数字或有效命令")


def show_final_results(system: SVCScoringSystem, session: SVCScoringSession):
    """显示最终结果"""
    cn_scores = [s for s in session.scores if s["language"] == "CN"]
    jp_scores = [s for s in session.scores if s["language"] == "JP"]

    cn_total = system.calculate_language_total(session.scores, "CN")
    jp_total = system.calculate_language_total(session.scores, "JP")
    overall = system.calculate_overall(cn_total["total_score"], jp_total["total_score"])

    print("\n" + "=" * 60)
    print("SVC评分完成! 最终结果:")
    print("=" * 60)

    print(f"\nCN歌曲 ({cn_total['song_count']}首):")
    for score in cn_scores:
        print(f"  {score['song_name']}: 权重{score['final_weight']}% → 总分{score['song_total_score']}")
    print(
        f"CN总分: {cn_total['total_score']} - {'合格' if cn_total['passed'] else '不合格'} - {'优秀' if cn_total['excellent'] else ''}")

    print(f"\nJP歌曲 ({jp_total['song_count']}首):")
    for score in jp_scores:
        print(f"  {score['song_name']}: 权重{score['final_weight']}% → 总分{score['song_total_score']}")
    print(
        f"JP总分: {jp_total['total_score']} - {'合格' if jp_total['passed'] else '不合格'} - {'优秀' if jp_total['excellent'] else ''}")

    print(f"\n总体评分: {overall['total_score']} - {'通过' if overall['passed'] else '不通过'}")
    print("=" * 60)

    # 保存最终结果
    session.save_session(permanent=True)
    print("\n结果已保存到永久日志")


if __name__ == "__main__":
    main()