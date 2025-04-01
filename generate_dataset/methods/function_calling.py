from generate_dataset.methods.method_manager import MethodManager, BaseMethod


@MethodManager.register("function-calling")
class FunctionCalling(BaseMethod):
    def __init__(self, input: str, output: str, global_rank):
        super().__init__(input, output, global_rank)


    def messages(self, index):
        return [
            {"role": "system", "content": "content"},
            {"role": "user", "content": "content"}
        ]
    
    def is_done(self, index):
        return False