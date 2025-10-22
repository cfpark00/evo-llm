"""Task-specific logic for different ES fine-tuning tasks."""

from src.tasks.conciseness import ConcisenessTask
from src.tasks.countdown import CountdownTask

TASK_REGISTRY = {
    "conciseness": ConcisenessTask,
    "countdown": CountdownTask,
}

def get_task(task_name, task_config):
    """
    Get task instance for a given task name.

    Args:
        task_name: Name of the task
        task_config: Task configuration dict

    Returns:
        Task instance with load_data() and compute_reward() methods
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRY.keys())}")

    task_class = TASK_REGISTRY[task_name]
    return task_class(task_config)
