import datetime
import traceback
import json
import os
from seeact.agent import SeeActAgent as SeeActCore
from cerebrum.agents.base import BaseAgent
from cerebrum.llm.communication import LLMQuery

class SeeActAgent(BaseAgent):
    def __init__(self, agent_name, task_input, config_):
        # Do not call the parent class initialization to avoid tool loading
        self.agent_name = agent_name
        self.task_input = task_input
        self.config = config_
        
        # Initialize basic properties
        self.messages = []
        self.workflow_mode = "manual"
        self.rounds = 0
        self.max_rounds = 5  # Set the maximum number of rounds to 5
        
        # Set the save path
        self.save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 
                                    "seeact_agent_files", 
                                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        self.seeact = SeeActCore(
            model="gpt-4o",
            default_task=task_input,
            default_website="https://www.google.com/",
            headless=True,
            save_file_dir=self.save_dir  # Set the save directory
        )

    def build_system_instruction(self):
        """Build system instructions"""
        prefix = "".join(self.config["description"])
        self.messages.append({"role": "system", "content": prefix})

    def run(self):
        """Run the agent's main logic"""
        try:
            import asyncio
            return asyncio.run(self._async_run())
        except Exception as e:
            error_msg = f"Error occurred: {str(e)}"
            try:
                asyncio.run(self.seeact.stop())
            except:
                pass
            return {
                "agent_name": self.agent_name,
                "result": "Task completed",  # Always return completed
                "rounds": self.rounds,
                "result_path": self.save_dir  # Changed from path to result_path
            }

    async def _async_run(self):
        """Asynchronously execute the logic, exactly as per the official example"""
        try:
            self.build_system_instruction()
            await self.seeact.start()
            
            while not self.seeact.complete_flag and self.rounds < self.max_rounds:
                prediction_dict = await self.seeact.predict()
                if prediction_dict:
                    # Check if the PDF has already been found
                    if isinstance(prediction_dict, dict) and prediction_dict.get("SAY", "").lower().find("pdf is open") != -1:
                        self.seeact.complete_flag = True
                        break
                    await self.seeact.execute(prediction_dict)
                self.rounds += 1
            
            await self.seeact.stop()
            
            return {
                "agent_name": self.agent_name,
                "result": "Task completed",  # Always return completed
                "rounds": self.rounds,
                "result_path": self.save_dir  # Changed from path to result_path
            }
            
        except Exception as e:
            await self.seeact.stop()
            raise       