from langchain_core.prompts import PromptTemplate

class TaskTemplate:
    def __init__(self, instruction, query_last=True, query_word="Question"):
        self.instruction = instruction
        self.query_last = query_last
        self.query_word = query_word
        self.prompt_template_str = f"{self.instruction}\n"
        self.setup_template()

    def setup_template(self):
        if self.query_last:
            prompt_str = "Documents:\n{context}\n" + self.query_word + ": {query}\nAnswer:"
        else:
            prompt_str = self.query_word + ": {query}\nDocuments:\n{context}\nAnswer:"
        self.prompt_template_str += prompt_str

    def create_prompt_template(self):
        return PromptTemplate.from_template(template=self.prompt_template_str)


class QueryOnlyTaskTemplate(TaskTemplate):
    # Override setup_template
    def setup_template(self):
        self.prompt_template_str = f"{self.instruction}\n"
        self.prompt_template_str += self.query_word + ": {query}\nAnswer:"


def apply_chat_task_template(
    chat_task_template_str: str, 
    task_instruction: str,
    is_query_only_task: bool = False    
):
    # Insert the task instruction in the chat template of the model
    chat_task_template = PromptTemplate.from_template(
        template=chat_task_template_str,
        partial_variables={"task_instruction": task_instruction}
    )

    # Create the template of the context with an empty instruction, since it was passed in the chat template 
    if is_query_only_task:
        context_template = QueryOnlyTaskTemplate("").create_prompt_template()
    else:
        context_template = TaskTemplate("").create_prompt_template()
    complete_task_template_str = chat_task_template.format(
        context_prompt=context_template.template
    )

    return PromptTemplate.from_template(template=complete_task_template_str)


task_instructions = {
    "query_only": "You are given a question and you must respond based on the provided documents. You must always provide an answer.",
    "nq": "You are given a question and you must respond based on the provided documents. You must always provide an answer.",
    "qa_proof": { 
        "nq": "You are given a question and you must respond based on the provided documents. You must always provide an answer. If none of the documents contain the answer, respond with NO-RES. In addition, you must report the portion of the document (Proof) containing the answer.\nSTART example\nDocument [20970787](Title: Ancient Egyptian technology) Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty . Chariots , however , are only believed to have been introduced by the invasion of the Hyksos in the Second Intermediate period ; during the New Kingdom era , chariotry became central to Egypt 's military .\nQuestion: when was the potter's wheel first used in egypt\nAnswer: 4th Dynasty\nProof: Evidence indicates that Egyptians made use of potter 's wheels in the manufacturing of pottery from as early as the 4th Dynasty .\nEND example\n", 

    },
    
}


task_templates = {
    "query_only": QueryOnlyTaskTemplate(task_instructions['query_only']),
    "nq": TaskTemplate(task_instructions['nq']),
    "qa_proof": {
        "nq": TaskTemplate(task_instructions['qa_proof']['nq']),
    },
}


chat_task_templates = {
    'google/gemma-2-2b-it': {
        "template": "<bos><start_of_turn>user\n{task_instruction}{context_prompt}<end_of_turn>\n<start_of_turn>model",
        "answer_prefix": "Answer:\nmodel",
    },
}