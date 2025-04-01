from generate_dataset.methods.method_manager import MethodManager, BaseMethod


@MethodManager.register("default")
class Default(BaseMethod):
    def __init__(self, input: str, output: str, global_rank):
        super().__init__(input, output, global_rank)


    def messages(self, index):
        return [
            {"role": "system", "content": "you can think"},
            {"role": "user", "content": "tell me what is 2 sqrt 89 and give me expation"}
        ]
    
    def is_done(self, index):
        return False