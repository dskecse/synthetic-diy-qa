from failure_modes import FailureMode, FailureModeDefinitions
from models import DIYRepairQA
from openai_client import get_openai_client

class LLMJudge:
    """
    LLM-as-a-Judge system for evaluating Q&A pairs against failure modes
    """

    def __init__(self, model: str = "gpt-5.4-nano"):
        self.client = get_openai_client()
        self.model = model
        self.failure_modes = FailureModeDefinitions.failure_modes()

    def _evaluate_against_single_failure_mode(self, qa_pair: DIYRepairQA, failure_mode: FailureMode) -> tuple[int, str]:
        """
        Evaluate a Q&A pair against a single failure mode

        Return a tuple of (failure_score, raw_response) where failure_score is 0 or 1.
        """
        try:
            # Format the evaluation prompt with Q&A data
            prompt = failure_mode.evaluation_prompt.format(
                question=qa_pair.question,
                answer=qa_pair.answer,
                equipment_problem=qa_pair.equipment_problem,
                tools_required=qa_pair.tools_required,
                steps=qa_pair.steps,
                safety_info=qa_pair.safety_info,
                tips=qa_pair.tips
            )

            response = self.client.responses.create(
                model=self.model,
                instructions="You are an expert DIY repair evaluator. Respond with ONLY 0 or 1.", # system prompt
                input=prompt,
                temperature=0.1, # Low temperature for consistent evaluation
                max_output_tokens=16 # Minimum value can't be <16 tokens
            )
            raw_response = response.output_text.strip()

            # Parse the response (should be 0 or 1)
            try:
                failure_score = int(raw_response)
                if failure_score not in [0, 1]:
                    failure_score = 1 # Default to failure if unclear
            except ValueError:
                failure_score = 1 # Default to failure if can't parse

            return failure_score, raw_response
        except Exception as e:
            print(f"Error evaluating {failure_mode.name}: {str(e)}")
            return 1, f"Error: {str(e)}" # Default to failure on error

    def evaluate(self, qa_pair: DIYRepairQA, trace_id: str) -> dict[str, any]:
        """
        Evaluate a Q&A pair against all failure modes
        """
        results = {
            "trace_id": trace_id,
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "equipment_problem": qa_pair.equipment_problem,
            "tools_required": qa_pair.tools_required,
            "steps": qa_pair.steps,
            "safety_info": qa_pair.safety_info,
            "tips": qa_pair.tips
        }

        # Evaluate against each failure mode
        failure_count = 0
        for failure_mode in self.failure_modes:
            failure_score, raw_response = self._evaluate_against_single_failure_mode(qa_pair, failure_mode)
            results[failure_mode.name] = failure_score
            results[f"{failure_mode.name}_response"] = raw_response
            failure_count += failure_score

        # Overall failure label (1 if any failure mode is 1, 0 if all are 0)
        results["overall_failure"] = 1 if failure_count > 0 else 0
        results["failure_count"] = failure_count

        return results
