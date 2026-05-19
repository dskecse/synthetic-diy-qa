from datetime import datetime, UTC
import json
import random
import time
import uuid

from models import GenerationResult, validate_json_structure
from openai_client import get_openai_client

class SyntheticDatasetGenerator:
    def __init__(self):
        self.client = get_openai_client()
        self.model = "gpt-5.4-nano"
        self.prompt_templates = self._create_prompt_templates()

    def _create_prompt_templates(self):
        return {
            "appliance_repair": {
                "system": "You are an expert home appliance repair technician with 20+ years of experience.",
                "user": """Generate a realistic home appliance repair Q&A pair. Focus on common household appliances like refrigerators, washing machines, dryers, dishwashers or ovens.

                Return ONLY a valid JSON object with this exact structure:
                {
                  "question": "A specific question about appliance repair",
                  "answer": "Detailed but concise step-by-step answer with technical details",
                  "equipment_problem": "Specific appliance and problem description",
                  "tools_required": ["specific", "tools", "needed"],
                  "steps": ["step 1", "step 2", "etc"],
                  "safety_info": "Important safety warnings and precautions",
                  "tips": "Professional tips and best practices"
                }

                Make it realistic and practical for a homeowner."""
            },
            "plumbing_repair": {
                "system": "You are a professional plumber with extensive residential experience.",
                "user": """Generate a realistic plumbing repair Q&A pair. Focus on common issues like leaks, clogs, fixture repairs or pipe problems.

                Return ONLY a valid JSON object with this exact structure:
                {
                  "question": "A specific question about plumbing repair",
                  "answer": "Detailed but concise step-by-step answer with technical details",
                  "equipment_problem": "Specific plumbing issue and location",
                  "tools_required": ["specific", "tools", "needed"],
                  "steps": ["step 1", "step 2", "etc"],
                  "safety_info": "Important safety warnings and precautions",
                  "tips": "Professional tips and best practices"
                }

                Make it realistic and safe for a homeowner to attempt."""
            },
            "electrical_repair": {
                "system": "You are a licensed electrician specializing in safe home electrical repairs.",
                "user": """Generate a realistic electrical repair Q&A pair. Focus on SAFE homeowner-level electrical work like outlet replacement, switch repair or light fixture installation.

                Return ONLY a valid JSON object with this exact structure:
                {
                  "question": "A specific question about electrical repair",
                  "answer": "Detailed but concise step-by-step answer with safety emphasis",
                  "equipment_problem": "Specific electrical issue or installation",
                  "tools_required": ["specific", "tools", "needed"],
                  "steps": ["step 1", "step 2", "etc"],
                  "safety_info": "Critical electrical safety warnings and when to call professionals",
                  "tips": "Professional tips and best practices"
                }

                Emphasize safety and when to call a professional. Only include repairs safe for homeowners."""
            },
            "hvac_maintenance": {
                "system": "You are an HVAC technician specializing in homeowner maintenance and basic repairs.",
                "user": """Generate a realistic HVAC maintenance or basic repair Q&A pair. Focus on filter changes, thermostat issues, vent cleaning or basic troubleshooting.

                Return ONLY a valid JSON object with this exact structure:
                {
                  "question": "A specific question about HVAC maintenance or repair",
                  "answer": "Detailed but concise step-by-step answer with seasonal considerations",
                  "equipment_problem": "Specific HVAC component or maintenance issue",
                  "tools_required": ["specific", "tools", "needed"],
                  "steps": ["step 1", "step 2", "etc"],
                  "safety_info": "Important safety warnings and precautions",
                  "tips": "Professional tips and maintenance best practices"
                }

                Make it realistic and practical for a homeowner to perform safely."""
            },
            "general_home_repair": {
                "system": "You are a skilled handyperson with general home repair and maintenance expertise.",
                "user": """Generate a realistic general home repair Q&A pair. Focus on common issues like drywall repair, door/window problems, flooring issues or basic carpentry.

                Return ONLY a valid JSON object with this exact structure:
                {
                  "question": "A specific question about general home repair",
                  "answer": "Detailed but concise step-by-step answer with material specifications",
                  "equipment_problem": "Specific home repair issue or project",
                  "tools_required": ["specific", "tools", "needed"],
                  "steps": ["step 1", "step 2", "etc"],
                  "safety_info": "Important safety warnings and precautions",
                  "tips": "Professional tips and best practices"
                }

                Make it realistic and practical for a DIY homeowner with basic skills."""
            }
        }

    def generate_qa_pair(self, template: str) -> GenerationResult:
        """
        Generate a single Q&A pair using a specified prompt template

        Uses a new Responses API instead of an older Chat Completions API.

        See https://developers.openai.com/api/docs/guides/migrate-to-responses?lang=python
        """
        trace_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).isoformat()

        prompt = self.prompt_templates.get(template)

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=prompt["system"],
                input=prompt["user"],
                temperature=0.7,
                max_output_tokens=1000
            )
            raw_response = response.output_text.strip()

            is_valid, qa_pair = validate_json_structure(raw_response)

            return GenerationResult(
                trace_id=trace_id,
                qa_pair=qa_pair,
                raw_response=raw_response,
                is_valid=is_valid,
                generated_at=timestamp
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            return GenerationResult(
                trace_id=trace_id,
                qa_pair=None,
                raw_response="",
                is_valid=False,
                generated_at=timestamp
            )

    def generate_dataset(self, num_samples: int) -> list[GenerationResult]:
        """
        Generate synthetic dataset with a specified number of samples
        """
        dataset = []
        prompt_template_names = list(self.prompt_templates.keys())

        for i in range(num_samples):
            # Randomly select a template for diversity
            template = random.choice(prompt_template_names)
            print(f"Generating sample {i+1}/{num_samples} using template: {template}")

            result = self.generate_qa_pair(template)
            dataset.append(result)

            # Brief pause to avoid rate limiting
            time.sleep(0.5)

        return dataset

    def save_results(self, results: list[GenerationResult], filename: str = "generation_results.json"):
        """
        Save generation results to JSON file
        """
        serializable_results = []
        for result in results:
            serializable_results.append(result.model_dump())

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {filename}")


def generate_synthetic_dataset(num_samples: int = 20):
    print("Starting synthetic dataset generation...")
    generator = SyntheticDatasetGenerator()

    print(f"Samples to be generated: {num_samples}")
    results = generator.generate_dataset(num_samples=num_samples)

    # Save results
    generator.save_results(results)

    # Print summary
    valid_count = sum(1 for result in results if result.is_valid)
    print(f"\nGeneration Phase Complete:")
    print(f"Total generated: {len(results)}")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples: {len(results) - valid_count}")
    print(f"Success rate: {valid_count/len(results)*100:.1f}%")

    return results

if __name__ == "__main__":
    generate_synthetic_dataset()
