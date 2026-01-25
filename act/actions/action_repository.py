from debug.aa import ActionFactory, Action

@ActionFactory.register("dummy_action")
class DummyAction(Action):
    def execute(self, agent, prompt: str) -> str:
        return "DO NOTHING for TEST"
    @classmethod
    def content(cls):
        return "DO NOTHING for TEST"