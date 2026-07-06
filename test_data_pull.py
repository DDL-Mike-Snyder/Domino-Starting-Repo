from domino_data.datasets import DatasetClient, DatasetConfig

# instantiate a client
token = request.headers.get('Authorization', '')[7:] if 'request' in vars() or 'request' in globals() else None
dataset = DatasetClient(token = token).get_dataset("dataset-images_dataset-6a4be722ce6de705f76dd012")

# select a specific snapshot, if not the read/write snapshot is used
# dataset.update(config=DatasetConfig(snapshot_id="68016814a206e098e0dd5884"))

# list files in the dataset
dataset.list_files()