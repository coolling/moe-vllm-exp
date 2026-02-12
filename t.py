# list_tasks_v04.py
from lm_eval.tasks import TaskManager
from lm_eval.tasks.natural_qa import NaturalQATask
print('Success!')
tm = TaskManager()
all_tasks = tm.task_index  # 这是一个 dict，key 是任务名

for name in sorted(all_tasks.keys()):
    if "natural" in name:
        print(name)