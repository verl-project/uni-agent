# from uni_agent.sandbox.modal import ModalSandbox
# from uni_agent.sandbox.vefaas import VefaasSandbox


# async with VefaasSandbox() as sandbox:
#     result = await sandbox.exec(["tmux", "-V"])
#     print(result)

from uni_agent.tasks import get_task


task_config = {
    "name": "swe_bench",
    "sandbox": {
        "provider": "modal",
        "image": "python:3.12",
    },
    "run_gold_patch": True,
}

task = get_task(task_config)
print(task)