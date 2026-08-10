import time
import uuid
import threading
from typing import Dict, List, Tuple, Optional, Callable
from queue import PriorityQueue
import json
import os
from distributed.distributed_strategy import StrategyFactory, LoadBalancingStrategy

class DistributedCoordinator:
    """分布式Agent协调器，负责管理Agent注册、任务分配和通信"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DistributedCoordinator, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """初始化协调器状态"""
        self.agents: Dict[str, Dict] = {}
        self.tasks: PriorityQueue = PriorityQueue()
        self.resource_locks: Dict[str, Tuple[str, float]] = {}
        self.message_queue: Dict[str, List[Dict]] = {}
        self.load_config()
        self.strategy_factory = StrategyFactory()
        self.load_balancing_strategy = self._create_strategy()
        self.failure_history = []
        if self.config.get('failover_config', {}).get('use_failover_strategy', True):
            from .failover_strategy import FailoverStrategy
            self.failover_strategy = FailoverStrategy(coordinator=self)
        else:
            self.failover_strategy = None
        self.monitor_thread = threading.Thread(target=self._monitor_agents, daemon=True)
        self.monitor_thread.start()

    def load_config(self):
        """加载分布式配置"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'distributed_config.json')
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                "agent_heartbeat_interval": 30,
                "task_timeout": 300,
                "load_balancing_strategy": "round_robin",
                "max_retries": 3
            }

    def _create_strategy(self) -> LoadBalancingStrategy:
        """创建负载均衡策略实例"""
        strategy_name = self.config.get('load_balancing_strategy', 'round_robin')
        return self.strategy_factory.create_strategy(strategy_name)

    def init_failover_strategy(self) -> None:
        """初始化故障转移策略"""
        from .failover_strategy import FailoverStrategy, BackupAgentStrategy
        strategy_type = self.config.get('failover_config', {}).get('failover_strategy_type', 'failover')
        if strategy_type == 'failover':
            self.failover_strategy = FailoverStrategy(coordinator=self)
        elif strategy_type == 'backup_agent':
            self.failover_strategy = BackupAgentStrategy(coordinator=self)
        else:
            self.failover_strategy = FailoverStrategy(coordinator=self)

    def set_strategy(self, strategy_name: str) -> bool:
        """设置负载均衡策略"""
        try:
            self.load_balancing_strategy = self.strategy_factory.create_strategy(strategy_name)
            self.config['load_balancing_strategy'] = strategy_name
            return True
        except ValueError:
            print(f"无效的策略名称: {strategy_name}")
            return False

    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str], endpoint: str) -> bool:
        """注册新Agent到协调器"""
        if agent_id in self.agents:
            print(f"Agent {agent_id} 已存在，更新信息...")
        self.agents[agent_id] = {
            'id': agent_id,
            'type': agent_type,
            'capabilities': capabilities,
            'endpoint': endpoint,
            'last_heartbeat': time.time(),
            'status': 'active',
            'task_count': 0
        }
        if agent_id not in self.message_queue:
            self.message_queue[agent_id] = []
        print(f"Agent {agent_id} 注册成功")
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """从协调器注销Agent"""
        if agent_id not in self.agents:
            print(f"Agent {agent_id} 不存在")
            return False
        del self.agents[agent_id]
        if agent_id in self.message_queue:
            del self.message_queue[agent_id]
        print(f"Agent {agent_id} 注销成功")
        return True

    def heartbeat(self, agent_id: str) -> bool:
        """更新Agent心跳"""
        if agent_id not in self.agents:
            return False
        self.agents[agent_id]['last_heartbeat'] = time.time()
        return True

    def assign_task(self, task_type: str, task_data: Dict, priority: int = 1, callback: Optional[Callable] = None) -> Optional[str]:
        """分配任务给合适的Agent"""
        suitable_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if task_type in agent['capabilities'] and agent['status'] == 'active'
        ]
        if not suitable_agents:
            print(f"没有找到能执行 {task_type} 类型任务的Agent")
            return None
        selected_agent = self._select_agent(suitable_agents)
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'type': task_type,
            'data': task_data,
            'priority': priority,
            'assigned_to': selected_agent,
            'assigned_at': time.time(),
            'callback': callback
        }
        self.tasks.put((priority, time.time(), task))
        self.agents[selected_agent]['task_count'] += 1
        print(f"任务 {task_id} 已分配给Agent {selected_agent}")
        return task_id

    def _select_agent(self, agent_ids: List[str]) -> str:
        """根据负载均衡策略选择Agent"""
        agent_info = {agent_id: self.agents[agent_id] for agent_id in agent_ids}
        return self.load_balancing_strategy.select_agent(agent_info)

    def send_message(self, from_agent: str, to_agent: str, message_type: str, content: Dict) -> bool:
        """在Agent之间发送消息"""
        if to_agent not in self.agents or from_agent not in self.agents:
            return False
        message = {
            'from': from_agent,
            'type': message_type,
            'content': content,
            'timestamp': time.time()
        }
        if to_agent not in self.message_queue:
            self.message_queue[to_agent] = []
        self.message_queue[to_agent].append(message)
        print(f"消息从 {from_agent} 发送到 {to_agent}")
        return True

    def get_messages(self, agent_id: str) -> List[Dict]:
        """获取Agent的消息队列"""
        if agent_id not in self.message_queue:
            return []
        messages = self.message_queue[agent_id].copy()
        self.message_queue[agent_id] = []
        return messages

    def acquire_lock(self, resource_id: str, agent_id: str, timeout: float = 10.0) -> bool:
        """获取资源锁"""
        current_time = time.time()
        if resource_id in self.resource_locks:
            holder_id, acquired_time = self.resource_locks[resource_id]
            if current_time - acquired_time < timeout:
                return False
            print(f"资源 {resource_id} 的锁已超时，释放并重新分配")
        self.resource_locks[resource_id] = (agent_id, current_time)
        print(f"Agent {agent_id} 成功获取资源 {resource_id} 的锁")
        return True

    def release_lock(self, resource_id: str, agent_id: str) -> bool:
        """释放资源锁"""
        if resource_id not in self.resource_locks:
            return False
        holder_id, _ = self.resource_locks[resource_id]
        if holder_id != agent_id:
            return False
        del self.resource_locks[resource_id]
        print(f"Agent {agent_id} 成功释放资源 {resource_id} 的锁")
        return True

    def _monitor_agents(self):
        """监控Agent状态的后台线程"""
        while True:
            current_time = time.time()
            heartbeat_interval = self.config.get('agent_heartbeat_interval', 30)
            for agent_id, agent in list(self.agents.items()):
                if current_time - agent['last_heartbeat'] > heartbeat_interval * 2:
                    print(f"Agent {agent_id} 心跳超时，标记为不活跃")
                    agent['status'] = 'inactive'
                    self._handle_timeout_agent(agent_id)
            time.sleep(heartbeat_interval)

    def _handle_timeout_agent(self, agent_id: str):
        """处理超时的Agent，实现故障转移"""
        print(f"处理超时Agent {agent_id} 的任务，启动故障转移...")
        failover_config = self.config.get('failover_config', {})
        failover_enabled = failover_config.get('enabled', True)
        priority_based = failover_config.get('priority_based_reassignment', True)
        preserve_priority = failover_config.get('preserve_task_priority', True)
        max_reassignment = failover_config.get('max_reassignment_attempts', 3)
        backup_agents = failover_config.get('backup_agents', {})
        use_failover_strategy = failover_config.get('use_failover_strategy', True)

        if not failover_enabled:
            print(f"故障转移功能已禁用，Agent {agent_id} 的任务将保持原状")
            self._log_failure_event(agent_id, "heartbeat_timeout", {"failover_disabled": True})
            return

        pending_tasks = []
        temp_queue = PriorityQueue()
        while not self.tasks.empty():
            priority, timestamp, task = self.tasks.get()
            if task['assigned_to'] == agent_id and 'completed' not in task:
                pending_tasks.append((priority, timestamp, task))
            else:
                temp_queue.put((priority, timestamp, task))
        while not temp_queue.empty():
            self.tasks.put(temp_queue.get())
        if priority_based:
            pending_tasks.sort(key=lambda x: x[0])

        for priority, timestamp, task in pending_tasks:
            print(f"重新分配任务 {task['id']} (原分配给 {agent_id})")
            task['original_agent_id'] = agent_id
            task['reassignment_count'] = task.get('reassignment_count', 0) + 1
            task.setdefault('previous_agents', []).append(agent_id)

            if task['reassignment_count'] > max_reassignment:
                task['status'] = 'failed'
                task['failure_reason'] = f"超过最大重分配次数 {max_reassignment}"
                self._log_failure_event(agent_id, "max_reassignment_exceeded", {
                    "task_id": task['id'],
                    "reassignment_count": task['reassignment_count'],
                    "previous_agents": task['previous_agents']
                })
                continue

            new_agent = None
            if use_failover_strategy and self.failover_strategy:
                available_agents = {
                    aid: agent for aid, agent in self.agents.items()
                    if aid != agent_id and agent['status'] == 'active'
                }
                new_agent = self.failover_strategy.select_agent(available_agents, task)
            else:
                backup_agent_id = backup_agents.get(agent_id)
                if (
                    backup_agent_id in self.agents
                    and self.agents[backup_agent_id]['status'] == 'active'
                ):
                    new_agent = backup_agent_id
                    print(f"使用备份Agent {new_agent} 处理任务 {task['id']}")
                if not new_agent:
                    suitable_agents = [
                        aid for aid, agent in self.agents.items()
                        if aid != agent_id
                        and agent['status'] == 'active'
                        and task['type'] in agent['capabilities']
                    ]
                    if suitable_agents:
                        new_agent = self._select_agent_for_failover(suitable_agents, task)

            if not new_agent:
                task['status'] = 'pending'
                task['pending_reason'] = "无可用Agent处理该任务"
                self.tasks.put((priority + 5, timestamp, task))
                self._log_failure_event(agent_id, "no_suitable_agent", {
                    "task_id": task['id'],
                    "task_type": task['type']
                })
                continue

            task['assigned_to'] = new_agent
            task['reassigned_at'] = time.time()
            max_retries = min(self.config.get('max_retries', 3), max_reassignment)
            if task['reassignment_count'] > max_retries:
                task['status'] = 'failed'
                task['failure_reason'] = f"超过最大重试次数 {max_retries}"
                self._log_failure_event(agent_id, "max_retries_exceeded", {
                    "task_id": task['id'],
                    "max_retries": max_retries,
                    "reassignment_count": task['reassignment_count']
                })
                if 'callback' in task and task['callback']:
                    try:
                        task['callback'](task['id'], None, "max_retries_exceeded")
                    except Exception as e:
                        print(f"执行回调时出错: {e}")
                continue

            if not preserve_priority:
                priority += 1
            self.tasks.put((priority, timestamp, task))
            self.agents[new_agent]['task_count'] += 1
            print(f"任务 {task['id']} 已重新分配给Agent {new_agent}")
            self._log_failure_event(agent_id, "task_reassigned", {
                "task_id": task['id'],
                "new_agent": new_agent,
                "reassignment_count": task['reassignment_count']
            })
            if use_failover_strategy and self.failover_strategy:
                self.failover_strategy.update_failover_history(agent_id, success=True)

        for resource_id, (holder_id, _) in list(self.resource_locks.items()):
            if holder_id == agent_id:
                del self.resource_locks[resource_id]
                print(f"释放超时Agent {agent_id} 持有的资源锁: {resource_id}")
        self._log_failure_event(agent_id, "heartbeat_timeout")

    def _select_agent_for_failover(self, agent_ids: List[str], task: Dict) -> str:
        """为故障转移选择最合适的Agent"""
        failover_config = self.config.get('failover_config', {})
        critical_agents = failover_config.get('critical_agents', [])
        previous_agents = task.get('previous_agents', [])
        available_agents = [aid for aid in agent_ids if aid not in previous_agents]
        if not available_agents:
            available_agents = agent_ids
        critical_available = [aid for aid in available_agents if aid in critical_agents]
        if critical_available:
            return min(critical_available, key=lambda aid: self.agents[aid]['task_count'])
        task_type = task['type']
        agent_scores = {}
        for aid in available_agents:
            agent = self.agents[aid]
            score = 10 if task_type in agent['capabilities'] else 0
            score -= agent['task_count']
            score -= self._count_recent_failures(aid) * 2
            agent_scores[aid] = score
        if agent_scores:
            return max(agent_scores.items(), key=lambda x: x[1])[0]
        return self._select_agent(available_agents)

    def _log_failure_event(self, agent_id: str, failure_type: str, details: Dict = None):
        """记录故障事件"""
        if details is None:
            details = {}
        failure_event = {
            'agent_id': agent_id,
            'failure_type': failure_type,
            'timestamp': time.time(),
            'details': details
        }
        if agent_id in self.agents:
            failure_event['agent_info'] = {
                'type': self.agents[agent_id]['type'],
                'capabilities': self.agents[agent_id]['capabilities'],
                'task_count': self.agents[agent_id]['task_count'],
                'last_heartbeat': self.agents[agent_id]['last_heartbeat']
            }
        if not hasattr(self, 'failure_history'):
            self.failure_history = []
        self.failure_history.append(failure_event)
        max_history = self.config.get('max_failure_history', 100)
        if len(self.failure_history) > max_history:
            self.failure_history = self.failure_history[-max_history:]
        print(f"故障事件已记录: {failure_type} - Agent {agent_id}")
        self._notify_failure(failure_event)

    def _notify_failure(self, failure_event: Dict):
        """发送故障通知"""
        notification_config = self.config.get('failure_notification', {})
        if not notification_config.get('enabled', False):
            return
        notification_types = notification_config.get('types', [])
        severity = self._determine_failure_severity(failure_event)
        severity_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_severity = notification_config.get('min_severity', 'low')
        if severity_levels.get(severity, 0) < severity_levels.get(min_severity, 0):
            return
        notification = {
            'type': 'failure_alert',
            'severity': severity,
            'timestamp': failure_event['timestamp'],
            'agent_id': failure_event['agent_id'],
            'failure_type': failure_event['failure_type'],
            'details': failure_event['details'],
            'message': f"Agent {failure_event['agent_id']} 发生 {failure_event['failure_type']} 故障"
        }
        for notification_type in notification_types:
            try:
                if notification_type == 'log':
                    self._send_log_notification(notification)
                elif notification_type == 'email':
                    self._send_email_notification(notification)
                elif notification_type == 'webhook':
                    self._send_webhook_notification(notification)
                elif notification_type == 'system':
                    self._send_system_notification(notification)
            except Exception as e:
                print(f"发送 {notification_type} 通知失败: {e}")

    def _determine_failure_severity(self, failure_event: Dict) -> str:
        """确定故障严重程度"""
        failure_type = failure_event['failure_type']
        if failure_type in ['system_crash', 'data_corruption']:
            base_severity = 'critical'
        elif failure_type in ['heartbeat_timeout', 'connection_error']:
            base_severity = 'high'
        elif failure_type in ['task_failure', 'resource_exhaustion']:
            base_severity = 'medium'
        else:
            base_severity = 'low'
        if failure_event.get('details', {}).get('is_critical_agent', False):
            if base_severity == 'medium':
                base_severity = 'high'
            elif base_severity == 'low':
                base_severity = 'medium'
        recent_failures = self._count_recent_failures(failure_event['agent_id'])
        if recent_failures > 3 and base_severity != 'critical':
            severity_levels = ['low', 'medium', 'high', 'critical']
            current_index = severity_levels.index(base_severity)
            base_severity = severity_levels[current_index + 1]
        return base_severity

    def _count_recent_failures(self, agent_id: str, time_window: int = 300) -> int:
        """计算最近一段时间内某Agent的故障次数"""
        if not hasattr(self, 'failure_history'):
            return 0
        current_time = time.time()
        return sum(
            1 for event in self.failure_history
            if event['agent_id'] == agent_id
            and current_time - event['timestamp'] <= time_window
        )

    def _send_log_notification(self, notification: Dict):
        """发送日志通知"""
        severity = notification['severity']
        message = notification['message']
        details = json.dumps(notification['details'], indent=2)
        print(f"[{severity.upper()}] {message}\n{details}")

    def _send_email_notification(self, notification: Dict):
        """发送邮件通知"""
        recipients = self.config.get('email_notification', {}).get('recipients', [])
        if not recipients:
            print("未配置邮件接收者，跳过邮件通知")
            return
        print(f"将发送邮件通知到: {', '.join(recipients)}")

    def _send_webhook_notification(self, notification: Dict):
        """发送Webhook通知"""
        webhook_url = self.config.get('webhook_notification', {}).get('url')
        if not webhook_url:
            print("未配置Webhook URL，跳过Webhook通知")
            return
        print(f"将发送Webhook通知到: {webhook_url}")

    def _send_system_notification(self, notification: Dict):
        """发送系统通知"""
        for agent_id, agent in self.agents.items():
            if agent['status'] == 'active':
                self.send_message(
                    from_agent="coordinator",
                    to_agent=agent_id,
                    message_type="system_notification",
                    content={
                        "notification_type": "failure_alert",
                        "severity": notification['severity'],
                        "message": notification['message'],
                        "affected_agent": notification['agent_id']
                    }
                )

    def get_failure_history(self, filters: Dict = None) -> List[Dict]:
        """获取故障历史记录"""
        if not hasattr(self, 'failure_history'):
            return []
        if filters is None:
            return self.failure_history
        filtered_history = []
        for event in self.failure_history:
            if all(event.get(key) == value for key, value in filters.items()):
                filtered_history.append(event)
        return filtered_history

    def get_cluster_status(self) -> Dict:
        """获取集群状态"""
        active_agents = [agent for agent in self.agents.values() if agent['status'] == 'active']
        inactive_agents = [agent for agent in self.agents.values() if agent['status'] == 'inactive']
        recent_failures = []
        if hasattr(self, 'failure_history'):
            current_time = time.time()
            recent_failures = [
                f for f in self.failure_history if current_time - f['timestamp'] <= 86400
            ]
        return {
            'total_agents': len(self.agents),
            'active_agents': len(active_agents),
            'inactive_agents': len(inactive_agents),
            'pending_tasks': self.tasks.qsize(),
            'active_agents_details': active_agents,
            'locked_resources': list(self.resource_locks.keys()),
            'recent_failures': recent_failures,
            'failure_count': len(recent_failures)
        }

coordinator = DistributedCoordinator()
