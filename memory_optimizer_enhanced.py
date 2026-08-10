import time
import heapq
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json
import os
import re
from textblob import TextBlob
import nltk

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

class MemoryOptimizerEnhanced:
    """增强版记忆优化器，扩展了基础版的功能，增加了聚类、摘要和情感分析"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.memory_vectors = {}
        self.load_config()
        self.sia = SentimentIntensityAnalyzer()
        self.cluster_model = None

    def load_config(self):
        """加载记忆优化配置"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'memory_optimizer_config.json')
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                "compression_threshold": 0.7,
                "importance_decay_rate": 0.01,
                "max_similar_vectors": 5,
                "relevance_threshold": 0.5,
                "cluster_count": 5,
                "summary_length": 3
            }

    def calculate_importance(self, memory: Dict) -> float:
        """计算记忆的重要性分数，考虑情感因素和用户反馈"""
        current_time = time.time()
        creation_time = memory.get('timestamp', current_time)
        time_diff = current_time - creation_time
        base_importance = memory.get('importance', 0.5)
        decay_factor = np.exp(-self.config['importance_decay_rate'] * time_diff)
        access_count = memory.get('access_count', 1)
        frequency_factor = min(1.5, 1 + np.log10(access_count))
        content = memory.get('content', '')
        if content:
            sentiment = self.sia.polarity_scores(content)
            sentiment_factor = 1 + 0.2 * sentiment['compound']
        else:
            sentiment_factor = 1.0
        user_importance = memory.get('user_importance', 1.0)
        importance = base_importance * decay_factor * frequency_factor * sentiment_factor * user_importance
        return importance

    def compress_memories(self, memories: Dict[str, Dict]) -> Tuple[Dict[str, Dict], List[str]]:
        """压缩相似记忆"""
        if not memories:
            return memories, []
        memory_ids = list(memories.keys())
        memory_texts = [memories[mid]['content'] for mid in memory_ids]
        try:
            vectors = self.vectorizer.fit_transform(memory_texts)
        except ValueError:
            return memories, []
        for i, mid in enumerate(memory_ids):
            self.memory_vectors[mid] = vectors[i]
        similarity_matrix = cosine_similarity(vectors)
        to_delete = set()
        compressed_memories = memories.copy()
        threshold = self.config['compression_threshold']
        for i in range(len(memory_ids)):
            if memory_ids[i] in to_delete:
                continue
            similar_indices = [j for j in range(len(memory_ids)) if i != j and similarity_matrix[i][j] > threshold]
            if not similar_indices:
                continue
            main_memory = compressed_memories[memory_ids[i]]
            for j in similar_indices:
                if memory_ids[j] in to_delete:
                    continue
                similar_memory = compressed_memories[memory_ids[j]]
                main_memory['content'] = f"{main_memory['content']}\n{similar_memory['content']}"
                main_memory['timestamp'] = max(main_memory.get('timestamp', 0), similar_memory.get('timestamp', 0))
                main_tags = set(main_memory.get('tags', []))
                main_tags.update(similar_memory.get('tags', []))
                main_memory['tags'] = list(main_tags)
                main_memory['access_count'] = main_memory.get('access_count', 1) + similar_memory.get('access_count', 1)
                to_delete.add(memory_ids[j])
                del compressed_memories[memory_ids[j]]
                if memory_ids[j] in self.memory_vectors:
                    del self.memory_vectors[memory_ids[j]]
        return compressed_memories, list(to_delete)

    def retrieve_relevant_memories(self, query: str, memories: Dict[str, Dict], top_k: int = 5) -> List[Tuple[str, float]]:
        """检索与查询相关的记忆"""
        if not memories:
            return []
        memory_ids = list(memories.keys())
        for mid in memory_ids:
            if mid not in self.memory_vectors and 'content' in memories[mid]:
                try:
                    self.memory_vectors[mid] = self.vectorizer.transform([memories[mid]['content']])
                except ValueError:
                    continue
        try:
            query_vector = self.vectorizer.transform([query])
        except ValueError:
            return []
        similarities = []
        for mid in memory_ids:
            if mid in self.memory_vectors:
                similarity = cosine_similarity(query_vector, self.memory_vectors[mid])[0][0]
                if similarity >= self.config['relevance_threshold']:
                    importance = self.calculate_importance(memories[mid])
                    similarities.append((mid, similarity * importance))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def prioritize_memories(self, memories: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """对记忆进行优先级排序"""
        priorities = []
        for mid, memory in memories.items():
            importance = self.calculate_importance(memory)
            priorities.append((mid, importance))
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def forget_low_priority_memories(self, memories: Dict[str, Dict], keep_ratio: float = 0.8) -> List[str]:
        """遗忘低优先级的记忆"""
        if not memories:
            return []
        priorities = self.prioritize_memories(memories)
        keep_count = max(1, int(len(memories) * keep_ratio))
        keep_ids = {mid for mid, _ in priorities[:keep_count]}
        to_delete = [mid for mid in memories.keys() if mid not in keep_ids]
        for mid in to_delete:
            del memories[mid]
            if mid in self.memory_vectors:
                del self.memory_vectors[mid]
        return to_delete

    def save_memories(self, memories: Dict[str, Dict], file_path: str) -> bool:
        """将记忆保存到文件"""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            serializable_memories = {}
            for mid, memory in memories.items():
                serializable_memory = memory.copy()
                for key, value in serializable_memory.items():
                    if isinstance(value, np.ndarray):
                        serializable_memory[key] = value.tolist()
                    elif isinstance(value, (np.float32, np.float64)):
                        serializable_memory[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        serializable_memory[key] = int(value)
                serializable_memories[mid] = serializable_memory
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_memories, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存记忆失败: {str(e)}")
            return False

    def load_memories(self, file_path: str) -> Dict[str, Dict]:
        """从文件加载记忆"""
        try:
            if not os.path.exists(file_path):
                print(f"记忆文件不存在: {file_path}")
                return {}
            with open(file_path, 'r', encoding='utf-8') as f:
                memories = json.load(f)
            self.memory_vectors = {}
            for mid, memory in memories.items():
                if 'content' in memory:
                    try:
                        self.memory_vectors[mid] = self.vectorizer.transform([memory['content']])
                    except ValueError:
                        continue
            if self.cluster_model is not None and len(memories) > 0:
                self.cluster_memories(memories)
            return memories
        except Exception as e:
            print(f"加载记忆失败: {str(e)}")
            return {}

    def optimize_memory_storage(self, memories: Dict[str, Dict]) -> Tuple[Dict[str, Dict], List[str]]:
        """全面优化记忆存储"""
        compressed_memories, deleted_due_to_compression = self.compress_memories(memories)
        deleted_due_to_priority = self.forget_low_priority_memories(compressed_memories)
        all_deleted = list(set(deleted_due_to_compression + deleted_due_to_priority))
        final_memories = {mid: memory for mid, memory in compressed_memories.items() if mid not in all_deleted}
        return final_memories, all_deleted

    def cluster_memories(self, memories: Dict[str, Dict]) -> Dict[int, List[str]]:
        """对记忆进行聚类"""
        if not memories:
            return {}
        memory_ids = list(memories.keys())
        memory_texts = [memories[mid]['content'] for mid in memory_ids]
        try:
            vectors = self.vectorizer.fit_transform(memory_texts)
        except ValueError:
            return {}
        self.memory_vectors = {mid: vectors[i] for i, mid in enumerate(memory_ids)}
        n_clusters = min(self.config['cluster_count'], len(memory_ids))
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.cluster_model.fit_predict(vectors)
        clusters = {i: [] for i in range(n_clusters)}
        for i, mid in enumerate(memory_ids):
            clusters[int(labels[i])].append(mid)
        return clusters

    def generate_summary(self, memory_content: str) -> str:
        """生成记忆内容的摘要"""
        if not memory_content:
            return ""
        try:
            blob = TextBlob(memory_content)
            sentences = list(blob.sentences)
        except Exception:
            sentences = [part.strip() for part in re.split(r'(?<=[.!?。！？])\s*', memory_content) if part.strip()]
        if not sentences:
            return ""
        if len(sentences) <= self.config['summary_length']:
            return memory_content.strip()
        return ' '.join(str(sent) for sent in sentences[:self.config['summary_length']])

    def auto_optimize(self, memories: Dict[str, Dict], interval: int = 3600) -> Tuple[Dict[str, Dict], List[str]]:
        """定期自动优化记忆"""
        last_optimization = getattr(self, 'last_optimization_time', 0)
        current_time = time.time()
        if current_time - last_optimization < interval:
            return memories, []
        optimized_memories, deleted = self.optimize_memory_storage(memories)
        self.last_optimization_time = current_time
        return optimized_memories, deleted

    def get_memory_connections(self, memory_id: str, memories: Dict[str, Dict], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """获取与指定记忆相关的其他记忆"""
        if memory_id not in self.memory_vectors or memory_id not in memories:
            return []
        current_vector = self.memory_vectors[memory_id]
        connections = []
        for mid, vector in self.memory_vectors.items():
            if mid != memory_id and mid in memories:
                similarity = cosine_similarity(current_vector, vector)[0][0]
                if similarity >= threshold:
                    connections.append((mid, similarity))
        connections.sort(key=lambda x: x[1], reverse=True)
        return connections

    def analyze_memory_sentiment(self, memory: Dict) -> Dict[str, float]:
        """分析记忆的情感"""
        content = memory.get('content', '')
        if not content:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        return self.sia.polarity_scores(content)

    def create_memory_index(self, memories: Dict[str, Dict]) -> None:
        """创建记忆索引以加速检索"""
        memory_texts = [memories[mid]['content'] for mid in memories.keys()]
        try:
            self.vectorizer.fit(memory_texts)
        except ValueError:
            return
        self.memory_vectors = {}
        for mid, memory in memories.items():
            try:
                self.memory_vectors[mid] = self.vectorizer.transform([memory['content']])
            except ValueError:
                continue

    def export_memory_snapshot(self, memories: Dict[str, Dict], file_path: str) -> bool:
        """导出记忆快照"""
        try:
            serializable_memories = {}
            for mid, memory in memories.items():
                serializable_memory = {
                    k: v for k, v in memory.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                serializable_memories[mid] = serializable_memory
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_memories, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"导出记忆快照失败: {e}")
            return False

    def import_memory_snapshot(self, file_path: str) -> Dict[str, Dict]:
        """导入记忆快照"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"导入记忆快照失败: {e}")
            return {}
