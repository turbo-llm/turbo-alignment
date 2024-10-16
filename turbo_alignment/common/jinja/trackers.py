import jinja2
from jinja2 import nodes
from jinja2.ext import Extension


class AssistantTracker(Extension):
    tags = {"generation"}

    def __init__(self, environment):
        super().__init__(environment)
        environment.extend(new_generation_trackers=self.new_generation_trackers)
        self.rendered_blocks = []
        self.generation_indices = []

    def new_generation_trackers(self):
        self.rendered_blocks = []
        self.generation_indices = []
        return self.rendered_blocks, self.generation_indices

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

    @jinja2.pass_eval_context
    def _generation_support(self, context, caller):
        rv = caller()
        start_index = len("".join(self.rendered_blocks))
        end_index = start_index + len(rv)
        self.generation_indices.append((start_index, end_index))
        return rv