from typing import Callable, Optional
from pyserini.search.lucene import LuceneSearcher
from . import AbstractEncoder
from string import Template


class WikipediaEncoder(AbstractEncoder):

    NOT_FOUND: str = "Wikipedia: NOT FOUND"

    def __init__(
        self,
        query_builder_fn: Callable,
        filter_fn: Optional[Callable] = None,
    ):
        
        self.searcher = LuceneSearcher(
            '/home/user/projects/data/kilt/indexes/kilt_document'
        )

        self.query_builder: Callable = query_builder_fn
        self.filter_fn: Optional[Callable] = filter_fn

    def __call__(self, obj):
        query = self.query_builder(obj)

        docs = self.searcher.search(query)
        if len(docs) == 0:
            return self.NOT_FOUND
            
        doc = docs[0]
        # Filter out the Page Title
        passage = "\n".join(doc.contents.split("\n")[1:])
        passage = f"Wikipedia: {passage}"

        # Adds the option to filter out passages that do 
        # not relate to the topic that is been searched
        if self.filter_fn and not self.filter_fn(obj, passage):
            return self.NOT_FOUND
        else:
            return self._clean_text(passage)


class WikipediaAuthorEncoder(WikipediaEncoder):

    def __init__(self):
        query_template = Template("""$author_name""")

        super().__init__(
            query_builder_fn = lambda obj: query_template.substitute(obj),
            # filter_fn = lambda obj, passage: obj['author_name'] in passage,
        )