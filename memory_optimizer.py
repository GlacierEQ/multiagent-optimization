import time
import heapq
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import torch
from sentence_transformers import SentenceTransformer

class MemoryOptimizer:
    """记忆优化器，负责记忆的压缩、重要性排序和高效检索"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.memory_vectors = {}
        self.semantic_vectors = {}
        self.load_config()
        try:
            self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.use_semantic = True
        except Exception as e:
            print(f"警告: 无法加载语义模型: {e}")
            self.use_semantic = False

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
                "semantic_weight": 0.6,
                "tfidf_weight": 0.4,
                "semantic_threshold": 0.75
            }

    def calculate_importance(self, memory: Dict) -> float:
        """计算记忆的重要性分数"""
        current_time = time.time()
        creation_time = memory.get('timestamp', current_time)
        time_diff = current_time - creation_time
        base_importance = memory.get('importance', 0.5)
        decay_factor = np.exp(-self.config['importance_decay_rate'] * time_diff)
        access_count = memory.get('access_count', 1)
        frequency_factor = min(1.5, 1 + np.log10(access_count))
        return base_importance * decay_factor * frequency_factor

    def compute_semantic_vectors(self, texts: List[str]) -> Optional[np.ndarray]:
        """计算文本的语义向量"""
        if not self.use_semantic or not texts:
            return None
        try:
            with torch.no_grad():
                return self.semantic_model.encode(texts)
        except Exception as e:
            print(f"警告: 计算语义向量时出错: {e}")
            return None

    def compute_similarity_matrix(self, memory_ids: List[str], memory_texts: List[str]) -> np.ndarray:
        """计算记忆之间的相似度矩阵，结合TF-IDF和语义相似度"""
        try:
            tfidf_vectors = self.vectorizer.fit_transform(memory_texts)
            tfidf_similarity = cosine_similarity(tfidf_vectors)
            for i, mid in enumerate(memory_ids):
                self.memory_vectors[mid] = tfidf_vectors[i]
        except ValueError:
            tfidf_similarity = np.zeros((len(memory_ids), len(memory_ids)))

        if self.use_semantic:
            semantic_vectors = self.compute_semantic_vectors(memory_texts)
            if semantic_vectors is not None:
                for i, mid in enumerate(memory_ids):
                    self.semantic_vectors[mid] = semantic_vectors[i]
                semantic_similarity = cosine_similarity(semantic_vectors)
                return (
                    self.config['tfidf_weight'] * tfidf_similarity
                    + self.config['semantic_weight'] * semantic_similarity
                )
        return tfidf_similarity

    def compress_memories(self, memories: Dict[str, Dict]) -> Tuple[Dict[str, Dict], List[str]]:
        """压缩相似记忆"""
        if not memories:
            return memories, []
        memory_ids = list(memories.keys())
        memory_texts = [memories[mid]['content'] for mid in memory_ids]
        similarity_matrix = self.compute_similarity_matrix(memory_ids, memory_texts)
        to_delete = set()
        compressed_memories = memories.copy()
        threshold = self.config['compression_threshold']
        for i in range(len(memory_ids)):
            if memory_ids[i] in to_delete:
                continue
            similar_indices = [
                j for j in range(len(memory_ids))
                if i != j and similarity_matrix[i][j] > threshold
            ]
            if not similar_indices:
                continue
            main_memory = compressed_memories[memory_ids[i]]
            for j in similar_indices:
                if memory_ids[j] in to_delete:
                    continue
                similar_memory = compressed_memories[memory_ids[j]]
                main_memory['content'] = f"{main_memory['content']}\n{similar_memory['content']}"
                main_memory['timestamp'] = max(
                    main_memory.get('timestamp', 0), similar_memory.get('timestamp', 0)
                )
                main_tags = set(main_memory.get('tags', []))
                main_tags.update(similar_memory.get('tags', []))
                main_memory['tags'] = list(main_tags)
                main_memory['access_count'] = (
                    main_memory.get('access_count', 1) + similar_memory.get('access_count', 1)
                )
                to_delete.add(memory_ids[j])
                del compressed_memories[memory_ids[j]]
                self.memory_vectors.pop(memory_ids[j], None)
                self.semantic_vectors.pop(memory_ids[j], None)
        return compressed_memories, list(to_delete)

    def retrieve_relevant_memories(
        self, query: str, memories: Dict[str, Dict], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """检索与查询相关的记忆，使用TF-IDF和语义相似度的组合"""
        if not memories:
            return []
        memory_ids = list(memories.keys())
        if not self.memory_vectors:
            self.compute_similarity_matrix(memory_ids, [memories[mid]['content'] for mid in memory_ids])
        try:
            query_vector = self.vectorizer.transform([query])
        except ValueError:
            return []
        semantic_query_vector = self.compute_semantic_vectors([query]) if self.use_semantic else None
        similarities = []
        for mid in memory_ids:
            combined_similarity = 0.0
            tfidf_similarity = 0.0
            semantic_similarity = 0.0
            if mid in self.memory_vectors:
                tfidf_similarity = cosine_similarity(query_vector, self.memory_vectors[mid])[0][0]
                combined_similarity = tfidf_similarity
            if semantic_query_vector is not None and mid in self.semantic_vectors:
                memory_semantic_vector = self.semantic_vectors[mid].reshape(1, -1)
                semantic_similarity = cosine_similarity(
                    semantic_query_vector, memory_semantic_vector
                )[0][0]
                combined_similarity = (
                    self.config['tfidf_weight'] * tfidf_similarity
                    + self.config['semantic_weight'] * semantic_similarity
                )
            if self.use_semantic:
                relevant = (
                    tfidf_similarity >= self.config['relevance_threshold']
                    or semantic_similarity >= self.config['semantic_threshold']
                )
            else:
                relevant = tfidf_similarity >= self.config['relevance_threshold']
            if relevant:
                importance = self.calculate_importance(memories[mid])
                similarities.append((mid, combined_similarity * importance))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def prioritize_memories(self, memories: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """对记忆进行优先级排序"""
        priorities = [
            (mid, self.calculate_importance(memory)) for mid, memory in memories.items()
        ]
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def forget_low_priority_memories(
        self, memories: Dict[str, Dict], keep_ratio: float = 0.8
    ) -> List[str]:
        """遗忘低优先级的记忆"""
        if not memories:
            return []
        priorities = self.prioritize_memories(memories)
        keep_count = max(1, int(len(memories) * keep_ratio))
        keep_ids = {mid for mid, _ in priorities[:keep_count]}
        to_delete = [mid for mid in list(memories.keys()) if mid not in keep_ids]
        for mid in to_delete:
            del memories[mid]
            self.memory_vectors.pop(mid, None)
            self.semantic_vectors.pop(mid, None)
        return to_delete

    def advanced_memory_compression(self, memories: Dict[str, Dict]) -> Dict[str, Dict]:
        """高级记忆压缩算法，结合语义聚类和信息提取"""
        if not memories or len(memories) < 3:
            return memories
        memory_ids = list(memories.keys())
        memory_texts = [memories[mid]['content'] for mid in memory_ids]
        similarity_matrix = self.compute_similarity_matrix(memory_ids, memory_texts)
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        try:
            condensed_distance = squareform(distance_matrix)
            z_matrix = linkage(condensed_distance, method='ward')
            max_dist = 1 - self.config['compression_threshold']
            clusters = fcluster(z_matrix, max_dist, criterion='distance')
        except Exception as e:
            print(f"聚类过程中出错: {e}")
            return memories
        compressed_memories = {}
        cluster_to_mids = {}
        for i, cluster_id in enumerate(clusters):
            cluster_to_mids.setdefault(cluster_id, []).append(memory_ids[i])
        for cluster_mids in cluster_to_mids.values():
            if len(cluster_mids) == 1:
                mid = cluster_mids[0]
                compressed_memories[mid] = memories[mid]
                continue
            cluster_memories = [memories[mid] for mid in cluster_mids]
            importance_scores = [
                (mid, self.calculate_importance(memories[mid])) for mid in cluster_mids
            ]
            importance_scores.sort(key=lambda x: x[1], reverse=True)
            base_mid = importance_scores[0][0]
            all_tags = set()
            for memory in cluster_memories:
                all_tags.update(memory.get('tags', []))
            compressed_memories[base_mid] = {
                'content': self._merge_memory_contents(cluster_memories),
                'timestamp': max(memory.get('timestamp', 0) for memory in cluster_memories),
                'tags': list(all_tags),
                'access_count': sum(memory.get('access_count', 0) for memory in cluster_memories),
                'source_memories': cluster_mids,
            }
        return compressed_memories

    def _merge_memory_contents(self, memories: List[Dict]) -> str:
        """智能合并多个记忆的内容"""
        if not memories:
            return ""
        if len(memories) == 1:
            return memories[0]['content']
        contents = [memory['content'] for memory in memories]
        if self.use_semantic:
            try:
                import re
                all_sentences = []
                for content in contents:
                    sentences = re.split(r'(?<=[.!?。！？])\s*', content)
                    all_sentences.extend(s.strip() for s in sentences if s.strip())
                if not all_sentences:
                    return "\n\n".join(contents)
                sentence_vectors = self.compute_semantic_vectors(all_sentences)
                if sentence_vectors is None:
                    return "\n\n".join(contents)
                sentence_similarities = cosine_similarity(sentence_vectors)
                selected_indices = []
                remaining_indices = list(range(len(all_sentences)))
                while remaining_indices and len(selected_indices) < min(10, len(all_sentences)):
                    if not selected_indices:
                        next_idx = max(remaining_indices, key=lambda i: len(all_sentences[i]))
                    else:
                        next_idx = max(
                            remaining_indices,
                            key=lambda idx: min(
                                sentence_similarities[idx][sel_idx]
                                for sel_idx in selected_indices
                            ),
                        )
                    selected_indices.append(next_idx)
                    remaining_indices.remove(next_idx)
                selected_indices.sort()
                return "\n".join(all_sentences[i] for i in selected_indices)
            except Exception as e:
                print(f"智能合并记忆内容时出错: {e}")
        return "\n\n".join(contents)

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
            if memories:
                try:
                    self.vectorizer.fit([m.get('content', '') for m in memories.values()])
                except ValueError:
                    pass
            for mid, memory in memories.items():
                if 'content' in memory:
                    try:
                        self.memory_vectors[mid] = self.vectorizer.transform([memory['content']])
                    except ValueError:
                        continue
            return memories
        except Exception as e:
            print(f"加载记忆失败: {str(e)}")
            return {}

    def optimize_memory_storage(
        self, memories: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict], List[str]]:
        """全面优化记忆存储"""
        compressed_memories, deleted_due_to_compression = self.compress_memories(memories)
        deleted_due_to_priority = self.forget_low_priority_memories(compressed_memories)
        all_deleted = list(set(deleted_due_to_compression + deleted_due_to_priority))
        final_memories = {
            mid: memory for mid, memory in compressed_memories.items()
            if mid not in all_deleted
        }
        return final_memories, all_deleted
