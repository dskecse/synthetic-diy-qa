from dataclasses import dataclass

@dataclass
class FailureMode:
    """
    Definition of a failure mode with evaluation criteria
    """
    name: str
    description: str
    success_criteria: str
    failure_criteria: str
    evaluation_prompt: str

class FailureModeDefinitions:
    """
    Defines 6 failure modes for Home DIY Repair Q&A evaluation
    """

    @staticmethod
    def failure_modes() -> list[FailureMode]:
        """
        Returns 6 failure modes with clear criteria for LLM evaluation
        """
        return [
            FailureMode(
                name="incomplete_answer",
                description="Answer lacks sufficient detail or steps to complete the repair",
                success_criteria="Answer provides comprehensive, step-by-step instructions that would allow someone to complete the repair successfully",
                failure_criteria="Answer is vague, missing critical steps or lacks sufficient detail for successful completion",
                evaluation_prompt="""
Evaluate if this DIY repair answer is COMPLETE and SUFFICIENT:

SUCCESS (0): Answer provides comprehensive, step-by-step instructions with enough detail for successful completion
FAILURE (1): Answer is vague, missing critical steps or lacks sufficient detail

Question: {question}
Answer: {answer}
Steps: {steps}

Rate as 0 (complete) or 1 (incomplete). Respond with only the number.
"""
            ),
            FailureMode(
                name="safety_violations",
                description="Missing critical safety warnings or dangerous advice",
                success_criteria="Includes appropriate safety warnings, precautions and mentions when to call professionals",
                failure_criteria="Missing safety warnings, provides dangerous advice or fails to mention professional help when needed",
                evaluation_prompt="""
Evaluate if this DIY repair guidance has ADEQUATE SAFETY information:

SUCCESS (0): Includes appropriate safety warnings, precautions and mentions professional help when needed
FAILURE (1): Missing safety warnings, provides dangerous advice or fails to mention professional help for complex tasks

Question: {question}
Answer: {answer}
Safety Info: {safety_info}

Rate as 0 (safe) or 1 (safety violation). Respond with only the number.
"""
            ),
            FailureMode(
                name="unrealistic_tools",
                description="Requires tools that are unrealistic for typical homeowners",
                success_criteria="Tools commonly available to homeowners or easily obtainable from hardware stores",
                failure_criteria="Requires specialized professional tools, overly expensive equipment or unrealistic tool combinations",
                evaluation_prompt="""
Evaluate if the required tools are REALISTIC for typical homeowners:

SUCCESS(0): Tools commonly available to homeowners or easily obtainable from hardware stores
FAILURE(1): Requires specialized professional tools, overly expensive equipment or unrealistic tool combinations

Question: {question}
Tools Required: {tools_required}
Equipment Problem: {equipment_problem}

Rate as 0 (realistic tools) or 1 (unrealistic tools). Respond with only the number.
"""
            ),
            FailureMode(
                name="overcomplicated_solution",
                description="Solution is unnecessarily complex for the problem described",
                success_criteria="Solution is appropriately scaled to the problem complexity and homeowner skill level",
                failure_criteria="Solution is overly complex, requires excessive steps or is disproportionate to the problem",
                evaluation_prompt="""
Evaluate if the DIY repair solution is of APPROPRIATE COMPLEXITY:

SUCCESS(0): Solution is appropriately scaled to the problem and homeowner skill level
FAILURE(1): Solution is overly complex, requires excessive steps or disproportionate to the problem

Question: {question}
Equipment Problem: {equipment_problem}
Answer: {answer}
Steps: {steps}
Tools Required: {tools_required}

Rate as 0 (appropriate complexity) or 1 (overcomplicated). Respond with only the number.
"""
            ),
            FailureMode(
                name="missing_context",
                description="Lacks important context about when, why or how to apply the solution",
                success_criteria="Provides context about when to use this solution, prerequisites and situational considerations",
                failure_criteria="Missing context about applicability, prerequisites or situational factors",
                evaluation_prompt="""
Evaluate if the DIY repair guidance provides ENOUGH CONTEXT:

SUCCESS(0): Provides context about when to use this solution, prerequisites and situational considerations
FAILURE(1): Missing context about applicability, prerequisites or situational factors

Question: {question}
Answer: {answer}
Equipment Problem: {equipment_problem}
Tips: {tips}

Rate as 0 (enough context) or 1 (missing context). Respond with only the number.
"""
            ),
            FailureMode(
                name="poor_quality_tips",
                description="Tips are generic, unhelpful or don't add value beyond the main answer",
                success_criteria="Tips provide specific, actionable advice that enhances the repair process",
                failure_criteria="Tips are generic, obvious, unhelpful or simply repeat information from the answer",
                evaluation_prompt="""
Evaluate if the provided tips are HIGH QUALITY and VALUABLE:

SUCCESS(0): Tips provide specific, actionable advice that enhances the repair process
FAILURE(1): Tips are generic, obvious, unhelpful or simply repeat information from the answer

Question: {question}
Answer: {answer}
Tips: {tips}

Rate as 0 (poor quality) or 1 (high quality). Respond with only the number.
"""
            ),
        ]
