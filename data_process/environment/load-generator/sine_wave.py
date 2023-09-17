from locust import HttpUser, TaskSet, task, constant
from locust import LoadTestShape
import base64
from random import choice

import math

class UserTasks(TaskSet):
    @task(5)
    def get_root(self):
        self.client.get("/")

    @task(10)
    def buy(self):
        base64string = base64.encodebytes(
            ('%s:%s' % ('user', 'password')).encode()).decode().strip()
        catalogue = self.client.get("/catalogue").json()
        category_item = choice(catalogue)
        item_id = category_item["id"]

        self.client.get("/")
        self.client.get(
            "/login", headers={"Authorization": "Basic %s" % base64string})
        self.client.get("/category.html")
        self.client.get("/detail.html?id={}".format(item_id))
        self.client.delete("/cart")
        self.client.post("/cart", json={"id": item_id, "quantity": 1})
        self.client.get("/basket.html")
        self.client.post("/orders")


class WebsiteUser(HttpUser):
    wait_time = constant(0.5)
    tasks = [UserTasks]


class SineWaveShape(LoadTestShape):
    """
    A simply load test shape class that has different user and spawn_rate at
    different stages.
    """

    def __init__(self):
        super(SineWaveShape, self).__init__()
        self.basic_users = 1000

        # generate a stage list which users are in a sine wave
        self.stages = [{"duration": 600, "users":  math.ceil(self.basic_users * (1 + 0.5 * math.sin(math.pi * 2 * i / 10))), "spawn_rate": 10} for i in range(10)]
        self.whole_time = sum([stage["duration"] for stage in self.stages])
        self.end_times = [sum([stage["duration"] for stage in self.stages[:i + 1]]) for i in range(len(self.stages))]
        print(self.stages)


    def tick(self):
        run_time = self.get_run_time()
        run_time = run_time % self.whole_time

        for i, stage in enumerate(self.stages):
            if run_time < self.end_times[i]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None
