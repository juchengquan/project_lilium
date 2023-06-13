class PostProcessorMixin:
    @classmethod
    def post_process(cls, texts):
        print("pre process")
        return texts
