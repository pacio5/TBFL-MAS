import logging
from spade.behaviour import FSMBehaviour


fsm_logger = logging.getLogger("FSM")


class StartAndEndFSMBehaviour(FSMBehaviour):
    async def on_start(self):
        print(f"{self.agent.name} starting at initial state {self.current_state}")
        fsm_logger.info(self.agent.name + " started")

    async def on_end(self):
        print(f"{self.agent.name} finished at state {self.current_state}")
        fsm_logger.info(self.agent.name + " stopped")
        await self.agent.stop()
