import base64

from locust import HttpUser, task, between, LoadTestShape
from random import randint, choice

class WebUser(HttpUser):
    wait_time = between(0.5, 10)

    @task(10)
    def browse(self):
        self.client.get("/")

    @task(5)
    def buy(self):
        base64string = base64.encodebytes(('%s:%s' % ('user','password')).encode()).decode().strip()
        catalogue = self.client.get("/catalogue").json()
        category_item = choice(catalogue)
        item_id = category_item["id"]
        self.client.get("/")
        self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})
        self.client.get("/category.html")
        self.client.get("/detail.html?id={}".format(item_id))
        self.client.delete("/cart")
        self.client.post("/cart", json={"id": item_id, "quantity": 1})
        self.client.get("/basket.html")
        self.client.post("/orders")
