from Mapping.ContentTypeMapping import ContentTypeMapping
from Mapping.CreatorMapping import CreatorMapping
from Mapping.KmersMapping import KmersMapping
from Mapping.EmbeddingMapping import EmbeddingMapping

from typing import Dict, Any


class MappingFactory:

    division_type: str

    def __init__(self, division_type: str):
        self.division_type = division_type

    def get_cluster(self, args: Dict[str, Any]) -> ContentTypeMapping:
        try:
            if self.division_type == "kmers":
                return KmersMapping(args)
            elif self.division_type == "embedding":
                return EmbeddingMapping(args)
            elif self.division_type == "creator":
                return CreatorMapping(args)
            else:
                raise ValueError
        except KeyError:
            print("args given doesn't match")
        except ValueError:
            print(f"invalid cluster type `{self.division_type}`")
        except:
            print("something else goes wrong")