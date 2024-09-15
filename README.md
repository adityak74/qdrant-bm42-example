### How to Run

1. Run the qdrant app in docker
```
$ docker compose up
```
2. Create a virtual environment and setup deps.
3. Run the follwing commands to populate/create the collection (it might take a while, so keep the terminal runnning)
```
$ python bm42-populate-async-batch.py
```
4. Start another terminal and you can query the collection. (modify the query parameter inside the code)
```
$ python bm42.py
```
