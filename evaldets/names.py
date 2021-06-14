from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata


class Names:
    def __init__(self, meta: Metadata) -> None:
        self.meta = meta
        self.idx_to_id = {
            v: k for k, v in self.meta.thing_dataset_id_to_contiguous_id.items()
        }
        self.name_to_i = {v: i for i, v in enumerate(self.meta.thing_classes)}

    def get(self, id: int) -> str:
        return self.meta.thing_classes[self.meta.thing_dataset_id_to_contiguous_id[id]]

    def name_to_id(self, name):
        return self.idx_to_id.get(self.name_to_idx(name))

    def name_to_idx(self, name):
        return self.name_to_i.get(name)

    @property
    def all(self):
        """Return all class names."""
        return self.meta.thing_classes

    @classmethod
    def for_dataset(cls, dataset):
        return cls(MetadataCatalog.get(dataset))
